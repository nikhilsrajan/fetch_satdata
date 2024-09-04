import datetime
import argparse

import sys
sys.path.append('..')

import config
import s2l1c_via_s3
import cdseutils.utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog = 'python download_s2l1c_from_s3.py',
        description = (
            'Script to download S2L1C tiles from S3.'    
        ),
        epilog = '--- Send your complaints to nsasiraj@umd.edu ---',
    )

    parser.add_argument('cdse_creds_json', help='/path/to/cdse_credentials.json')
    parser.add_argument('roi_filepath', help='/path/to/shapefile')
    parser.add_argument('startdate', help='YYYY-MM-DD')
    parser.add_argument('enddate', help='YYYY-MM-DD')
    parser.add_argument('--tile-limit', action='store', required=False, default=1, help='[default = 1] This is to make a user conscious about how many tiles will get downloaded. Size of each tile is about 700 MB.')
    parser.add_argument('--log-chunk', action='store', required=False, default=100, help='[default = 100] Number of files after which the catalog will be updated. Number of files is different from number of tiles as each tile contains multiple files.')

    args = parser.parse_args()

    cdse_creds = cdseutils.utils.cdse_credentials_from_json(args.cdse_creds_json)
    roi_filepath = args.roi_filepath
    startdate = datetime.datetime.strptime(args.startdate, '%Y-%m-%d')
    enddate = datetime.datetime.strptime(args.enddate, '%Y-%m-%d')
    upper_limit_for_number_of_tiles = int(args.tile_limit)
    chunksize_for_download_and_update_catalog = int(args.log_chunk)

    successful_download_count, total_download_count \
    = s2l1c_via_s3.download_sentinel2_l1c_tiles(
        cdse_creds = cdse_creds,
        sentinel2_local_catalog_filepath = config.FILEPATH_SENTINEL2_LOCAL_CATALOG,
        catalog_save_folderpath = config.FOLDERPATH_FETCH_CATALOG_CACHE,
        root_download_folderpath = config.FOLDERPATH_SATELLITE,
        roi_filepath = roi_filepath,
        startdate = startdate,
        enddate = enddate,
        upper_limit_for_number_of_tiles = upper_limit_for_number_of_tiles,
        chunksize_for_download_and_update_catalog = chunksize_for_download_and_update_catalog,
    )
