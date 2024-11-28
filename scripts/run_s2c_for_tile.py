import s2cloudless
import rasterio.warp
import multiprocessing as mp
import numpy as np
import os
import functools
import time

import sys
sys.path.append('..')

import config
import argparse
import rsutils.utils
from init_logger import get_logger


logger = get_logger()


# https://stackoverflow.com/questions/77783414/using-s2cloudless-to-generate-cloud-mask-using-sentinel-2-l1c-raw-data
# https://forum.step.esa.int/t/info-introduction-of-additional-radiometric-offset-in-pb04-00-products/35431
# https://sentiwiki.copernicus.eu/web/s2-processing
QUANTIFICATION_VALUE = 10000
RADIO_ADD_OFFSET = -1000

REF_BAND = 'B08'
BANDS_TO_RESAMPLE = ['B01', 'B05', 'B8A', 'B09', 'B10', 'B11', 'B12']
WORKING_BANDS = ['B01', 'B02', 'B04', 'B05', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']


def single_resample_tif(
    tif_filepath:str,
    ref_meta:dict,
):
    resampled_image, _ = rsutils.utils.resample_tif_inplace(
        src_filepath = tif_filepath,
        ref_meta = ref_meta,
        resampling = rasterio.warp.Resampling.nearest,
    )
    return resampled_image


def parallel_resample_tif(
    tif_filepaths:str,
    ref_meta:dict,
    njobs:int,
):  
    _single_resample_tif_partial = functools.partial(
        single_resample_tif,
        ref_meta = ref_meta,
    )
        
    with mp.Pool(njobs) as p:
        resampled_images = list(p.imap(_single_resample_tif_partial, tif_filepaths))
    
    return resampled_images


def resample_tiles_and_create_datacube(
    filepaths:dict,
    njobs:int,
):
    bands_to_keep_as_is = [
        band for band in WORKING_BANDS 
        if band not in BANDS_TO_RESAMPLE
    ]

    band_image_dict = {}

    ref_meta = None
    for band in bands_to_keep_as_is:
        with rasterio.open(filepaths[band]) as src:
            band_image_dict[band] = src.read()
            ref_meta = src.meta.copy()
    
    images_resampled = parallel_resample_tif(
        tif_filepaths = [filepaths[band] for band in BANDS_TO_RESAMPLE],
        ref_meta = ref_meta,
        njobs = njobs,
    )

    for i, band in enumerate(BANDS_TO_RESAMPLE):
        band_image_dict[band] = images_resampled[i]
    
    del images_resampled

    ordered_bands = [band_image_dict[band] for band in WORKING_BANDS]

    del band_image_dict

    datacube = np.stack(ordered_bands, axis=-1)

    del ordered_bands

    return datacube, ref_meta


def run_s2c(
    filepaths:dict,
    out_filepath:str,
    njobs:int,
):
    logger.info('Resample rasters and create datacube.')
    datacube, ref_meta = \
    resample_tiles_and_create_datacube(
        filepaths = filepaths,
        njobs = njobs,
    )

    cloud_detector = s2cloudless.S2PixelCloudDetector(
        threshold=0.4, average_over=4, dilation_size=2, all_bands=False
    )

    logger.info('Running s2cloudless.')
    cloud_prob = cloud_detector.get_cloud_probability_maps(
        data = (datacube + RADIO_ADD_OFFSET) / QUANTIFICATION_VALUE, 
        num_threads = njobs,
    )

    del datacube

    int_cloud_prob = (cloud_prob * QUANTIFICATION_VALUE).astype(int)

    out_folderpath = os.path.split(out_filepath)[0]

    if len(out_folderpath) == 0:
        os.makedirs(os.path.split(out_filepath)[0], exist_ok=True)

    logger.info('Saving cloud probability raster')
    with rasterio.open(out_filepath, 'w', **ref_meta) as dst:
        dst.write(int_cloud_prob)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog = 'python run_s2c_for_tile.py',
        description = (
            'Script to run s2cloudless and create a cloud probability tif for a tile. '
            'It is important to note that this script is done for a tile as the bounds of '
            'tile image across bands is the same even when they have different spatial '
            'resolutions.'
        ),
        epilog = f"--- Send your complaints to {','.join(config.MAINTAINERS)} ---",
    )
    parser.add_argument('--b01', action='store', required=True, help='Filepath to B01.')
    parser.add_argument('--b02', action='store', required=True, help='Filepath to B02.')
    parser.add_argument('--b04', action='store', required=True, help='Filepath to B04.')
    parser.add_argument('--b05', action='store', required=True, help='Filepath to B05.')
    parser.add_argument('--b08', action='store', required=True, help='Filepath to B08.')
    parser.add_argument('--b8a', action='store', required=True, help='Filepath to B8A.')
    parser.add_argument('--b09', action='store', required=True, help='Filepath to B09.')
    parser.add_argument('--b10', action='store', required=True, help='Filepath to B10.')
    parser.add_argument('--b11', action='store', required=True, help='Filepath to B11.')
    parser.add_argument('--b12', action='store', required=True, help='Filepath to B12.')
    parser.add_argument('--export', action='store', required=True, help='Filepath to export to.')
    parser.add_argument('-j', '--njobs', default=1, action='store', required=False, help='[default = 1] Number of cores to use.')

    args = parser.parse_args()

    filepaths = {
        'B01': str(args.b01),
        'B02': str(args.b02),
        'B04': str(args.b04),
        'B05': str(args.b05),
        'B08': str(args.b08),
        'B8A': str(args.b8a),
        'B09': str(args.b09),
        'B10': str(args.b10),
        'B11': str(args.b11),
        'B12': str(args.b12),
    }

    out_filepath = str(args.export)

    njobs = int(args.njobs)


    start_time = time.time()

    run_s2c(
        filepaths = filepaths,
        out_filepath = out_filepath,
        njobs = njobs,
    )

    end_time = time.time()

    logger.info(f'--- t_elapsed: {round(end_time - start_time, 2)} secs ---')
