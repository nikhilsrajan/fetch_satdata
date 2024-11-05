import numpy as np
import s2cloudless
import copy
import tqdm
import datetime
import pandas as pd
import numba
import multiprocessing as mp

import create_datacube
import exceptions


# https://stackoverflow.com/questions/77783414/using-s2cloudless-to-generate-cloud-mask-using-sentinel-2-l1c-raw-data
# https://forum.step.esa.int/t/info-introduction-of-additional-radiometric-offset-in-pb04-00-products/35431
# https://sentiwiki.copernicus.eu/web/s2-processing
QUANTIFICATION_VALUE = 10000
RADIO_ADD_OFFSET = -1000


"""
A datacube is defined here as a 4d numpy array - dimensions being (timestamps, height, width, bands)
Datacube is the output of `create_datacube` function in the module of the same name.
"""


def run_datacube_ops(
    datacube:np.ndarray,
    metadata:dict,
    sequence:list,
    print_messages:bool=True,
):
    for func, kwargs in sequence:
        if print_messages:
            print(f'Running {func.__qualname__} -- kwargs={kwargs}')
        datacube, metadata = func(datacube=datacube, metadata=metadata, **kwargs)
    return datacube, metadata


def _run_s2cloudless_core(
    datacube:np.ndarray,
):
    """
    Assumes only the 10 bands required to run the model are passed
    after normalisation
    """
    cloud_detector = s2cloudless.S2PixelCloudDetector(
        threshold=0.4, average_over=4, dilation_size=2, all_bands=False
    )
    cmk = cloud_detector.get_cloud_probability_maps(data=datacube)
    int_cmk = (cmk * 10000).astype(int)
    return int_cmk


def _run_s2cloudless_core_chunkwise(
    datacube:np.ndarray,
    chunksize:int = 50,
    njobs:int = 8,
    print_messages:bool = True,
):
    """
    Assumes only the 10 bands required to run the model are passed
    after normalisation
    """
    N = datacube.shape[0]

    datacube_chunks = [datacube[i:i+chunksize] for i in range(0, N, chunksize)]

    with mp.Pool(njobs) as p:
        if print_messages:
            int_cmks = list(tqdm.tqdm(
                p.imap(_run_s2cloudless_core, datacube_chunks), 
                total=len(datacube_chunks)
            ))
        else:
            int_cmks = list(p.imap(_run_s2cloudless_core, datacube_chunks))
    
    return np.concatenate(int_cmks, axis=0)

    
def run_s2cloudless(
    datacube:np.ndarray,
    metadata:dict,
    chunksize:int = 10,
    njobs:int = 8,
    print_messages:bool = True,
):
    MODEL_BANDS = s2cloudless.utils.MODEL_BANDS

    missing_bands = set(MODEL_BANDS) - set(metadata['bands'])
    if len(missing_bands) > 0:
        raise exceptions.DatacubeException(
            f'Bands required by s2cloudless missing: {list(missing_bands)}'
        )

    band_indices = {_band : _index for _index, _band in enumerate(metadata['bands'])}
    model_band_indexes = [band_indices[_model_band] for _model_band in MODEL_BANDS]

    int_cmk = _run_s2cloudless_core_chunkwise(
        datacube = (datacube[:,:,:,model_band_indexes] + RADIO_ADD_OFFSET) / QUANTIFICATION_VALUE,
        chunksize = chunksize,
        njobs = njobs,
        print_messages = print_messages,
    )

    datacube = np.concatenate([datacube, np.expand_dims(int_cmk, axis=3)], axis=3)

    _metadata = copy.deepcopy(metadata)
    _metadata['bands'].append('CMK')

    return datacube, _metadata


def is_list_sorted(l:list):
    # https://stackoverflow.com/questions/3755136/pythonic-way-to-check-if-a-list-is-sorted-or-not
    return all(l[i] <= l[i+1] for i in range(len(l) - 1))


def get_mosaic_ts_index_ranges(
    timestamps:list[pd.Timestamp],
    startdate:datetime.datetime,
    enddate:datetime.datetime,
    mosaic_days:int = 20,
):
    startdate = create_datacube.dt2ts(startdate)
    enddate = create_datacube.dt2ts(enddate)

    if not is_list_sorted(timestamps):
        raise exceptions.DatacubeException('timestamps is not sorted.')
    if startdate > timestamps[0]:
        raise exceptions.DatacubeException('startdate must be before on on the first timestamp')
    if enddate < timestamps[-1]:
        raise exceptions.DatacubeException('enddate must be after or on the last timestamp')

    mosaic_buckets = []

    cur_date = startdate
    while cur_date < enddate:
        next_date = cur_date + datetime.timedelta(days = mosaic_days)
        mosaic_buckets.append((cur_date, next_date))
        cur_date = next_date
    
    cur_interval_index = 0
    ts_indexes = [[]]
    for index, ts in enumerate(timestamps):
        # gte_date = mosaic_buckets[cur_interval_index][0]
        lte_date = mosaic_buckets[cur_interval_index][1]
        if ts <= lte_date:
            ts_indexes[cur_interval_index].append(index)
        else:
            ts_indexes.append([index])
            cur_interval_index += 1

    ts_index_ranges = [(min(index_list), max(index_list)) for index_list in ts_indexes if len(index_list) > 0]

    return ts_index_ranges


@numba.njit()
def _median_mosaic_core(
    datacube:np.ndarray,
    ts_index_ranges:np.ndarray,
):
    n_ts, height, width, n_bands = datacube.shape
    n_mosaiced_ts = ts_index_ranges.shape[0]

    mosaiced_datacube = np.full(
        shape=(n_mosaiced_ts, height, width, n_bands), 
        fill_value=np.nan,
    )

    for t in numba.prange(n_mosaiced_ts):
        for h in numba.prange(height):
            for w in numba.prange(width):
                for b in numba.prange(n_bands):
                    mosaiced_datacube[t,h,w,b] = np.nanmedian(
                        datacube[ts_index_ranges[t][0] : ts_index_ranges[t][1] + 1, h, w, b]
                    )
    
    return mosaiced_datacube
    

def median_mosaic(
    datacube:np.ndarray,
    metadata:dict,
    startdate:datetime,
    enddate:datetime,
    mosaic_days:int = 20,
    mask_value:int = 0,
) -> tuple[np.ndarray, dict]:
    ts_index_ranges = get_mosaic_ts_index_ranges(
        timestamps = metadata['timestamps'],
        startdate = startdate,
        enddate = enddate,
        mosaic_days = mosaic_days,
    )

    dtype = datacube.dtype
    datacube = datacube.astype(float)
    datacube[np.where(datacube==mask_value)] = np.nan

    mosaiced_datacube = _median_mosaic_core(
        datacube = datacube, ts_index_ranges = np.array(ts_index_ranges)
    )
    
    mosaiced_datacube[np.isnan(mosaiced_datacube)] = mask_value
    mosaiced_datacube = mosaiced_datacube.astype(dtype)

    mosaiced_metadata = copy.deepcopy(metadata)
    mosaiced_metadata['mosaic_index_intervals'] = ts_index_ranges
    mosaiced_metadata['previous_timestamps'] = metadata['timestamps']
    mosaiced_metadata['timestamps'] = [
        metadata['timestamps'][ts_index_range[0]] for ts_index_range in ts_index_ranges
    ]
    mosaiced_metadata['data_shape_desc'] = ('timestamps', 'height', 'width', 'bands')

    return mosaiced_datacube, mosaiced_metadata


def apply_cloud_mask(
    datacube:np.ndarray,
    metadata:dict,
    cloud_threshold:float,
    bands_to_modify:list[str]=None,
    mask_value = 0,
):
    band_indices = {band:index for index,band in enumerate(metadata['bands'])}

    if 'CMK' not in band_indices.keys():
        raise exceptions.DatacubeException(f'CMK band not present in datacube')
    
    if bands_to_modify is None:
        bands_to_modify = list(band_indices.keys())
        bands_to_modify.remove('CMK')
    
    present_bands_to_modify = set(bands_to_modify) & set(band_indices.keys())

    band_indices_to_modify = [
        band_indices[bandname] for bandname in present_bands_to_modify
    ]

    cmk_index = band_indices['CMK']

    cmk = datacube[:,:,:,cmk_index]
    selected_bands = datacube[:,:,:,band_indices_to_modify]

    selected_bands[np.where(cmk >= cloud_threshold)] = mask_value

    datacube[:,:,:,band_indices_to_modify] = selected_bands

    return datacube, metadata


def apply_cloud_mask_scl(
    datacube:np.ndarray,
    metadata:dict,
    mask_classes:list[int],
    bands_to_modify:list[str]=None,
    mask_value = 0,
):
    band_indices = {band:index for index,band in enumerate(metadata['bands'])}

    if 'SCL' not in band_indices.keys():
        raise exceptions.DatacubeException(f'SCL band not present in datacube')
    
    if bands_to_modify is None:
        bands_to_modify = list(band_indices.keys())
        bands_to_modify.remove('SCL')
    
    present_bands_to_modify = set(bands_to_modify) & set(band_indices.keys())

    band_indices_to_modify = [
        band_indices[bandname] for bandname in present_bands_to_modify
    ]

    scl_index = band_indices['SCL']

    scl = datacube[:,:,:,scl_index]
    selected_bands = datacube[:,:,:,band_indices_to_modify]

    selected_bands[np.where(np.isin(scl, mask_classes))] = mask_value

    datacube[:,:,:,band_indices_to_modify] = selected_bands

    return datacube, metadata


def drop_bands(
    datacube:np.ndarray,
    metadata:dict,
    bands_to_drop:list[str],
):
    band_indices = {band:index for index,band in enumerate(metadata['bands'])}

    bands_to_keep = [band for band in band_indices.keys() if band not in bands_to_drop]
    band_indices_to_keep = [index for band, index in band_indices.items() if band in bands_to_keep]
    datacube = datacube[:,:,:,band_indices_to_keep]
    metadata['bands'] = bands_to_keep

    return datacube, metadata


def area_median(
    datacube:np.ndarray,
    metadata:dict,
    mask_value:int = 0,
):
    dtype = datacube.dtype

    n_ts, height, width, n_b = datacube.shape

    datacube = datacube.astype(float)
    datacube[np.where(datacube==mask_value)] = np.nan

    mosaiced_datacube = np.expand_dims(np.nanmedian(datacube, axis=(1,2)), axis=(1,2))
    mosaiced_datacube[np.isnan(mosaiced_datacube)] = mask_value
    mosaiced_datacube = mosaiced_datacube.astype(dtype)

    mosaiced_metadata = copy.deepcopy(metadata)
    mosaiced_metadata['previous_height_width'] = (height, width)

    return mosaiced_datacube, mosaiced_metadata
