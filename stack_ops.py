import numpy as np
import s2cloudless
import copy
import tqdm
import datetime
import pandas as pd
import numba


# https://stackoverflow.com/questions/77783414/using-s2cloudless-to-generate-cloud-mask-using-sentinel-2-l1c-raw-data
# https://forum.step.esa.int/t/info-introduction-of-additional-radiometric-offset-in-pb04-00-products/35431
# https://sentiwiki.copernicus.eu/web/s2-processing
QUANTIFICATION_VALUE = 10000
RADIO_ADD_OFFSET = -1000


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
):
    """
    Assumes only the 10 bands required to run the model are passed
    after normalisation
    """
    N = bands.shape[0]
    
    int_cmks = []
    for i in tqdm.tqdm(range(0, N, chunksize)):
        _int_cmk = _run_s2cloudless_core(
            bands = bands[i:i+chunksize]
        )
        int_cmks.append(_int_cmk)
        del _int_cmk
    
    return np.concatenate(int_cmks, axis=0)

    
def run_s2cloudless(
    bands:np.ndarray,
    metadata:dict,
    chunksize:int = 10,
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

    ts_index_ranges = [(min(index_list), max(index_list)) for index_list in ts_indexes]

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
    n_ts, height, width, n_bands = bands.shape

    ts_index_ranges = get_mosaic_ts_index_ranges(
        timestamps = metadata['timestamps'],
        startdate = startdate,
        enddate = enddate,
        mosaic_days = mosaic_days,
    )

    n_mosaiced_ts = len(ts_index_ranges)

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

    return mosaiced_bands, mosaiced_metadata
