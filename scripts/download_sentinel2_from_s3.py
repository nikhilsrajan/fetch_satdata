import os
import datetime
import argparse
import sentinelhub

import sys
sys.path.append('..')

import config
import sentinel2_via_s3
import cdseutils.utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog = 'python download_sentinel2_from_s3.py',
        description = (
            'Script to download Sentinel-2 tiles from S3.'    
        ),
        epilog = f"--- Send your complaints to {','.join(config.MAINTAINERS)} ---",
    )

    VALID_LEVELS = ['l1c', 'l2a']

    parser.add_argument('cdse_creds_json', help='/path/to/cdse_credentials.json')
    parser.add_argument('roi_filepath', help='/path/to/shapefile')
    parser.add_argument('startdate', help='YYYY-MM-DD')
    parser.add_argument('enddate', help='YYYY-MM-DD')
    parser.add_argument('-F', '--folderpath', action='store', required=False, default=config.FOLDERPATH_SATELLITE, help='/path/to/download-to')
    parser.add_argument('-l', '--level', action='store', required=False, default='l2a', help=f'[default = l2a] Level of the sentinel-2 images to download. Options: {VALID_LEVELS}')
    parser.add_argument('--tile-limit', action='store', required=False, default=1, help='[default = 1] This is to make a user conscious about how many tiles will get downloaded. Size of each tile is about 700 MB.')
    parser.add_argument('--log-chunk', action='store', required=False, default=100, help='[default = 100] Number of files after which the catalog will be updated. Number of files is different from number of tiles as each tile contains multiple files.')

    args = parser.parse_args()

    cdse_creds = cdseutils.utils.cdse_credentials_from_json(args.cdse_creds_json)
    roi_filepath = args.roi_filepath
    startdate = datetime.datetime.strptime(args.startdate, '%Y-%m-%d')
    enddate = datetime.datetime.strptime(args.enddate, '%Y-%m-%d')
    upper_limit_for_number_of_tiles = int(args.tile_limit)
    chunksize_for_download_and_update_catalog = int(args.log_chunk)

    root_folderpath = args.folderpath

    sentinel2_local_catalog_filepath = os.path.join(root_folderpath, 'catalog_sentinel-2.geojson')

    level = str(args.level).lower()
    if level not in VALID_LEVELS:
        raise ValueError(f'Invalid level = {level}. level must be from {VALID_LEVELS}')
    
    if level == 'l1c':
        satellite = cdseutils.constants.Bands.S2L1C.NAME
        bands = cdseutils.constants.Bands.S2L1C.ALL
        collection = sentinelhub.DataCollection.SENTINEL2_L1C
    elif level == 'l2a':
        satellite = cdseutils.constants.Bands.S2L2A.NAME
        bands = cdseutils.constants.Bands.S2L2A.ALL
        # bands = ['B04', 'B08', 'SCL']
        collection = sentinelhub.DataCollection.SENTINEL2_L2A
    else:
        raise NotImplementedError(f'level = {level}')
    

    successful_download_count, total_download_count \
    = sentinel2_via_s3.download_sentinel2_tiles(
        cdse_creds = cdse_creds,
        collection = collection,
        satellite = satellite,
        bands = bands,
        sentinel2_local_catalog_filepath = sentinel2_local_catalog_filepath,
        catalog_save_folderpath = config.FOLDERPATH_FETCH_CATALOG_CACHE,
        root_download_folderpath = root_folderpath,
        roi_filepath = roi_filepath,
        startdate = startdate,
        enddate = enddate,
        upper_limit_for_number_of_tiles = upper_limit_for_number_of_tiles,
        chunksize_for_download_and_update_catalog = chunksize_for_download_and_update_catalog,
    )
