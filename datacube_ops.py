import numpy as np
import s2cloudless
import copy
import tqdm
import datetime
import pandas as pd
import numba
import multiprocessing as mp


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
    bands:np.ndarray,
    metadata:dict,
    sequence:list,
    print_messages:bool=True,
):
    for func, kwargs in sequence:
        if print_messages:
            print(f'Running {func.__qualname__}')
        bands, metadata = func(bands=bands, metadata=metadata, **kwargs)
    return bands, metadata


def _run_s2cloudless_core(
    bands:np.ndarray,
):
    """
    Assumes only the 10 bands required to run the model are passed
    after normalisation
    """
    cloud_detector = s2cloudless.S2PixelCloudDetector(
        threshold=0.4, average_over=4, dilation_size=2, all_bands=False
    )
    cmk = cloud_detector.get_cloud_probability_maps(data=bands)
    int_cmk = (cmk * 10000).astype(int)
    return int_cmk


def _run_s2cloudless_core_chunkwise(
    bands:np.ndarray,
    chunksize:int = 50,
    njobs:int = 8,
    print_messages:bool = True,
):
    """
    Assumes only the 10 bands required to run the model are passed
    after normalisation
    """
    N = bands.shape[0]

    bands_chunks = [bands[i:i+chunksize] for i in range(0, N, chunksize)]

    with mp.Pool(njobs) as p:
        if print_messages:
            int_cmks = list(tqdm.tqdm(
                p.imap(_run_s2cloudless_core, bands_chunks), 
                total=len(bands_chunks)
            ))
        else:
            int_cmks = list(p.imap(_run_s2cloudless_core, bands_chunks))
    
    return np.concatenate(int_cmks, axis=0)

    
def run_s2cloudless(
    bands:np.ndarray,
    metadata:dict,
    chunksize:int = 10,
    njobs:int = 8,
    print_messages:bool = True,
):
    MODEL_BANDS = s2cloudless.utils.MODEL_BANDS

    missing_bands = set(MODEL_BANDS) - set(metadata['bands'])
    if len(missing_bands) > 0:
        raise ValueError(f'Bands required by s2cloudless missing: {list(missing_bands)}')

    band_indices = {_band : _index for _index, _band in enumerate(metadata['bands'])}
    model_band_indexes = [band_indices[_model_band] for _model_band in MODEL_BANDS]

    int_cmk = _run_s2cloudless_core_chunkwise(
        bands = (bands[:,:,:,model_band_indexes] + RADIO_ADD_OFFSET) / QUANTIFICATION_VALUE,
        chunksize = chunksize,
        njobs = njobs,
        print_messages = print_messages,
    )

    bands = np.concatenate([bands, np.expand_dims(int_cmk, axis=3)], axis=3)

    _metadata = copy.deepcopy(metadata)
    _metadata['bands'].append('CMK')

    return bands, _metadata


def _dt2ts(
    dt:datetime.datetime, 
    tz='UTC',
):
    return pd.Timestamp(dt, tz=tz)


def is_list_sorted(l:list):
    # https://stackoverflow.com/questions/3755136/pythonic-way-to-check-if-a-list-is-sorted-or-not
    return all(l[i] <= l[i+1] for i in range(len(l) - 1))


def get_mosaic_ts_index_ranges(
    timestamps:list[pd.Timestamp],
    startdate:datetime.datetime,
    enddate:datetime.datetime,
    mosaic_days:int = 20,
):
    startdate = _dt2ts(startdate)
    enddate = _dt2ts(enddate)

    if not is_list_sorted(timestamps):
        raise ValueError('timestamps is not sorted.')
    if startdate > timestamps[0]:
        raise ValueError('startdate must be before on on the first timestamp')
    if enddate < timestamps[-1]:
        raise ValueError('enddate must be after or on the last timestamp')

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
    bands:np.ndarray,
    ts_index_ranges:np.ndarray,
):
    n_ts, height, width, n_bands = bands.shape
    n_mosaiced_ts = ts_index_ranges.shape[0]

    mosaiced_bands = np.full(
        shape=(n_mosaiced_ts, height, width, n_bands), 
        fill_value=np.nan,
    )

    for t in numba.prange(n_mosaiced_ts):
        for h in numba.prange(height):
            for w in numba.prange(width):
                for b in numba.prange(n_bands):
                    mosaiced_bands[t,h,w,b] = np.nanmedian(
                        bands[ts_index_ranges[t][0] : ts_index_ranges[t][1] + 1, h, w, b]
                    )
    
    return mosaiced_bands
    

def median_mosaic(
    bands:np.ndarray,
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

    dtype = bands.dtype
    bands = bands.astype(float)
    bands[np.where(bands==mask_value)] = np.nan

    mosaiced_bands = _median_mosaic_core(
        bands = bands, ts_index_ranges = np.array(ts_index_ranges)
    )
    
    mosaiced_bands[np.isnan(mosaiced_bands)] = mask_value
    mosaiced_bands = mosaiced_bands.astype(dtype)

    mosaiced_metadata = copy.deepcopy(metadata)
    mosaiced_metadata['mosaic_index_intervals'] = ts_index_ranges
    mosaiced_metadata['previous_timestamps'] = metadata['timestamps']
    mosaiced_metadata['timestamps'] = [
        metadata['timestamps'][ts_index_range[0]] for ts_index_range in ts_index_ranges
    ]
    mosaiced_metadata['data_shape_desc'] = ('timestamps', 'height', 'width', 'bands')

    return mosaiced_bands, mosaiced_metadata


def apply_cloud_mask(
    bands:np.ndarray,
    metadata:dict,
    cloud_threshold:float,
    bands_to_modify:list[str]=None,
    mask_value = 0,
):
    band_indices = {band:index for index,band in enumerate(metadata['bands'])}

    if 'CMK' not in band_indices.keys():
        raise ValueError(f'CMK band not present in datacube')
    
    if bands_to_modify is None:
        bands_to_modify = list(band_indices.keys())
        bands_to_modify.remove('CMK')
    
    present_bands_to_modify = set(bands_to_modify) & set(band_indices.keys())

    band_indices_to_modify = [
        band_indices[bandname] for bandname in present_bands_to_modify
    ]

    cmk_index = band_indices['CMK']

    cmk = bands[:,:,:,cmk_index]
    selected_bands = bands[:,:,:,band_indices_to_modify]

    selected_bands[np.where(cmk >= cloud_threshold)] = mask_value

    bands[:,:,:,band_indices_to_modify] = selected_bands

    return bands, metadata


def drop_bands(
    bands:np.ndarray,
    metadata:dict,
    bands_to_drop:list[str],
):
    band_indices = {band:index for index,band in enumerate(metadata['bands'])}

    bands_to_keep = [band for band in band_indices.keys() if band not in bands_to_drop]
    band_indices_to_keep = [index for band, index in band_indices.items() if band in bands_to_keep]
    bands = bands[:,:,:,band_indices_to_keep]
    metadata['bands'] = bands_to_keep

    return bands, metadata
