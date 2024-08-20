import datetime

import sys
sys.path.append('..')

import config
import s2l1c_via_s3
import cdseutils.utils


if __name__ == '__main__':
    """
    Arguments:
    - filepath to cdse credentials json
    - ROI shape filepath
    - startdate (YYYY-MM-DD)
    - enddate (YYYY-MM-DD)
    - upper_limit_for_number_of_tiles
    - chunksize_for_download_and_update_catalog
    """

    cdse_creds = cdseutils.utils.cdse_credentials_from_json(sys.argv[1])
    roi_filepath = sys.argv[2]
    startdate = datetime.datetime.strptime(sys.argv[3], '%Y-%m-%d')
    enddate = datetime.datetime.strptime(sys.argv[4], '%Y-%m-%d')
    upper_limit_for_number_of_tiles = int(sys.argv[5])
    chunksize_for_download_and_update_catalog = int(sys.argv[6])

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
