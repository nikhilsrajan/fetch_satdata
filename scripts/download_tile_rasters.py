import os
import argparse
import json
import copy

import config
from init_logger import get_logger

import sys
sys.path.append('..')

import cdseutils.constants
import cdseutils.sentinel2
import run_s2c_for_tile


VALID_SATELLITES = [
    cdseutils.constants.Bands.S2L1C.NAME,
    cdseutils.constants.Bands.S2L2A.NAME,
]


logger = get_logger()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog = 'python download_tile_rasters.py',
        description = (
            'Script to fetch band rasters for a given tile. Requires the satellite, s3 url, '
            'and download root folderpath as inputs. The script downloads the rasters and '
            'outputs a json file containing the local filepaths of the band rasters.'
        ),
        epilog = f"--- Send your complaints to {','.join(config.MAINTAINERS)} ---",
    )

    parser.add_argument('cdse_creds_json', help='Path to cdse_credentials.json')
    parser.add_argument('satellite', help=f'Satellite source to make the query for. Options: {VALID_SATELLITES}')
    parser.add_argument('s3url', help='s3 path where the tile rasters are present')
    parser.add_argument('root_download_folderpath', help='Path to the root folder where the downloads are to be saved. This folder is common across satellites.')
    parser.add_argument('filepaths_json', help='Path to filepaths json file. This file contains info of the files that were downloaded using this script.')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-d', '--download-jobs', metavar='N_DOWNLOAD_JOBS', required=False, default=1, action='store', help=f'[default = 1] Number of parallel downloads. Max allowed: {cdseutils.utils.MAX_CONCURRENT_CONNECTIONS}')
    # parser.add_argument('-s2c', '--s2-cloudless', metavar='N_CORES', required=False, default=None, action='store', help=f'Option to run s2cloudless. Number of cores to use while running s2cloudless.')

    args = parser.parse_args()

    cdse_creds = cdseutils.utils.cdse_credentials_from_json(str(args.cdse_creds_json))

    satellite = str(args.satellite)

    s3url = str(args.s3url)

    if satellite not in VALID_SATELLITES:
        raise ValueError(f"Invalid satellite='{satellite}'. Valid options: {VALID_SATELLITES}")
    
    bands = {
        cdseutils.constants.Bands.S2L1C.NAME: cdseutils.constants.Bands.S2L1C.ALL,
        cdseutils.constants.Bands.S2L2A.NAME: cdseutils.constants.Bands.S2L2A.ALL,
    }[satellite]
    
    root_download_folderpath = str(args.root_download_folderpath)

    download_jobs = int(args.download_jobs)
    if download_jobs > cdseutils.utils.MAX_CONCURRENT_CONNECTIONS:
        njobs = cdseutils.utils.MAX_CONCURRENT_CONNECTIONS
        logger.warning(
            f'MAX_CONCURRENT_CONNECTIONS = {cdseutils.utils.MAX_CONCURRENT_CONNECTIONS}. '
            f'Limiting njobs to {cdseutils.utils.MAX_CONCURRENT_CONNECTIONS}.'
        )
    elif download_jobs < 1:
        download_jobs = 1

    filepaths_json = str(args.filepaths_json)

    run_s2c = False
    if satellite == cdseutils.constants.Bands.S2L1C.NAME and args.s2_cloudless is not None:
        run_s2c = True
        s2c_cores = int(args.s2_cloudless)

    if args.verbose:
        logger.setLevel('INFO')

    if args.overwrite:
        logger.warning(
            'using option --overwrite, use this option sparingly to '
            'avoid using up download quota.'
        )
    logger.info('Fetching band raster s3paths')
    s3paths, download_filepaths = \
    cdseutils.sentinel2.get_s3paths_single_url(
        s3url = s3url,
        s3_creds = cdse_creds.s3_creds,
        root_folderpath = root_download_folderpath,
        bands = bands,
        satellite = satellite,
    )

    logger.info('Downloading raster bands')
    download_statuses = cdseutils.utils.download_s3_files(
        s3_creds = cdse_creds.s3_creds,
        s3paths = s3paths,
        download_filepaths = download_filepaths,
        overwrite = args.overwrite,
        logger = logger,
        njobs = download_jobs,
    )

    logger.info(f'Downloaded {sum([status != cdseutils.utils.S3_DOWNLOAD_FAILED for status in download_statuses])} files.')

    info = {
        'id': None,
        'files': [],
    }

    file_info_template = {
        'filename': None,
        's3url': None,
        'filepath': None,
        'is_band': False,
        'band': None,
        'downloaded': False,
    }

    for s3path, download_filepath, download_status in zip(s3paths, download_filepaths, download_statuses):
        s3url = cdseutils.utils.s3path_to_s3url(s3path = s3path)
        parsed_s3_url = cdseutils.sentinel2.parse_s3url(s3url = s3url)

        if info['id'] is None:
            info['id'] = parsed_s3_url['id']
        elif info['id'] != parsed_s3_url['id']:
            raise ValueError('This should have not happened: id not same across the urls.')

        file_info = copy.deepcopy(file_info_template)

        if parsed_s3_url['band_filename'] is not None:
            band_filename = parsed_s3_url['band_filename']
            parsed_band_filename = cdseutils.sentinel2.parse_band_filename(
                sentinel2_band_filename = band_filename,
                satellite = satellite,
            )
            local_filename = parsed_band_filename['band'] + parsed_band_filename['ext']
            file_info['is_band'] = True
            file_info['band'] = parsed_band_filename['band']

        elif parsed_s3_url['xml_filename'] is not None:
            local_filename = parsed_s3_url['xml_filename']
        
        file_info['filename'] = local_filename
        file_info['filepath'] = os.path.abspath(download_filepath)
        file_info['s3url'] = s3url
        file_info['downloaded'] = download_status != cdseutils.utils.S3_DOWNLOAD_FAILED

        info['files'].append(file_info)

    ## The time it takes to run s2cloudless at a tile level is not really worth it.
    ## At this point it is just better to run when the datacube creation takes place.
    # if run_s2c:
    #     band_filepaths = {}
    #     for file_info in info['files']:
    #         band = file_info['band']
    #         if band in run_s2c_for_tile.WORKING_BANDS:
    #             band_filepaths[band] = file_info['filepath']
    #             if not file_info['downloaded']:
    #                 msg = f'Download of the band {band} failed - critical to run s2cloudless.'
    #                 logger.error(msg)
    #                 raise ValueError(msg)

    #     cmk_filepath = band_filepaths['B08'].replace('B08', 'CMK')
            
    #     run_s2c_for_tile.run_s2c(
    #         filepaths = band_filepaths,
    #         out_filepath = cmk_filepath,
    #         njobs = s2c_cores,
    #     )

    #     file_info = copy.deepcopy(file_info_template)
    #     file_info['band'] = 'CMK'
    #     file_info['downloaded'] = True
    #     file_info['filepath'] = os.path.abspath(cmk_filepath)
    #     file_info['is_band'] = True

    #     info['files'].append(file_info)
 
    os.makedirs(os.path.split(filepaths_json)[0], exist_ok=True)
    with open(filepaths_json, 'w') as h:
        json.dump(info, h, indent=4)
    
    logger.info(f'Filepaths info json saved at: {os.path.abspath(filepaths_json)}')
