import numpy as np
import s2cloudless
import copy
import tqdm


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
