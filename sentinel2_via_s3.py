import os
import geopandas as gpd
import datetime
import sentinelhub
import logging

import sys
sys.path.append('..')

import cdseutils.mydataclasses
import cdseutils.utils
import cdseutils.constants
import cdseutils.sentinel2
import catalogmanager_old as cm


COL_ID = cm.COL_ID
COL_TIMESTAMP = 'timestamp'
COL_LOCAL_FOLDERPATH = cm.COL_LOCAL_FOLDERPATH
COL_GEOMETRY = 'geometry'
COL_LAST_UPDATE = cm.COL_LAST_UPDATE
COL_FILES = cm.COL_FILES
COL_S3URL = 's3url'
COL_CLOUDCOVER = 'cloud_cover'
COL_SATELLITE = 'satellite'


def get_catalog_with_manager(
    sentinel2_local_catalog_filepath:str,
):
    catalog_manager = cm.CatalogManager(
        catalog_filepath = sentinel2_local_catalog_filepath,
        cols_dtype_dict = {
                          COL_ID: cm.DTYPE_STR,
                   COL_SATELLITE: cm.DTYPE_STR,
                   COL_TIMESTAMP: cm.DTYPE_TIMESTAMP,
                       COL_S3URL: cm.DTYPE_STR,
            COL_LOCAL_FOLDERPATH: cm.DTYPE_STR,
                       COL_FILES: cm.DTYPE_LIST_STR,
                 COL_LAST_UPDATE: cm.DTYPE_TIMESTAMP,
                  COL_CLOUDCOVER: cm.DTYPE_FLOAT,
                    COL_GEOMETRY: cm.DTYPE_MULTIPOLYGON,
        }
    )
    return catalog_manager


def update_catalog(
    catalog_gdf:gpd.GeoDataFrame, # contains other info that needs to be stored in local catalog
    s3paths:list[cdseutils.mydataclasses.S3Path],
    download_filepaths:list[str],
    download_successes:list[bool],
    sentinel2_local_catalog_filepath:str,
    satellite:str,
):
    catalog_manager = get_catalog_with_manager(
        sentinel2_local_catalog_filepath = sentinel2_local_catalog_filepath,
    )

    id_to_download_files = {}
    id_to_download_folderpath = {}
    for s3path, download_filepath, download_success in zip(s3paths, 
                                                           download_filepaths, 
                                                           download_successes):
        if not download_success:
            continue
        
        parsed_s3_url = cdseutils.sentinel2.parse_s3url(
            s3url = cdseutils.utils.s3path_to_s3url(s3path = s3path)
        )
        _id = parsed_s3_url['id']

        if _id not in id_to_download_files.keys():
            id_to_download_files[_id] = []
        
        if parsed_s3_url['band_filename'] is not None:
            band_filename = parsed_s3_url['band_filename']
            parsed_band_filename = cdseutils.sentinel2.parse_band_filename(
                sentinel2_band_filename=band_filename,
                satellite = satellite,
            )
            local_filename = parsed_band_filename['band'] + parsed_band_filename['ext']
        elif parsed_s3_url['xml_filename'] is not None:
            local_filename = parsed_s3_url['xml_filename']

        id_to_download_files[_id].append(local_filename)
        id_to_download_folderpath[_id] = os.path.abspath(os.path.split(download_filepath)[0])

    _ids = id_to_download_files.keys()
    selected_catalog_gdf = catalog_gdf[catalog_gdf[COL_ID].isin(_ids)]

    id_to_timestamps = dict(zip(selected_catalog_gdf[COL_ID], selected_catalog_gdf[COL_TIMESTAMP]))
    id_to_s3urls = dict(zip(selected_catalog_gdf[COL_ID], selected_catalog_gdf[COL_S3URL]))
    id_to_cloudcover = dict(zip(selected_catalog_gdf[COL_ID], selected_catalog_gdf[COL_CLOUDCOVER]))
    id_to_geometry = dict(zip(selected_catalog_gdf[COL_ID], selected_catalog_gdf[COL_GEOMETRY]))

    for _id in _ids:
        catalog_manager.add(
            entry = {
                              COL_ID: _id,
                       COL_TIMESTAMP: id_to_timestamps[_id],
                           COL_S3URL: id_to_s3urls[_id],
                COL_LOCAL_FOLDERPATH: id_to_download_folderpath[_id],
                           COL_FILES: id_to_download_files[_id],
                      COL_CLOUDCOVER: id_to_cloudcover[_id],
                        COL_GEOMETRY: id_to_geometry[_id],
                       COL_SATELLITE: satellite,
            }
        )
    
    catalog_manager.save()


def chunkwise_download_files_and_update_catalog(
    catalog_gdf:gpd.GeoDataFrame,
    s3_creds:cdseutils.mydataclasses.S3Credentials,
    s3paths:list[cdseutils.mydataclasses.S3Path],
    download_filepaths:list[str],
    sentinel2_local_catalog_filepath:str,
    satellite:str,
    chunksize = 4 * 25,
    overwrite:bool = False,
    logger:logging.Logger = None,
):
    N = len(s3paths)
    download_success_count = 0
    download_failed_count = 0
    download_skipped_count = 0
    download_overwrite_count = 0
    
    downloads_done = 0
    for i in range(0, N, chunksize):
        _s3paths = s3paths[i:i+chunksize]
        _download_filepaths = download_filepaths[i:i+chunksize]
        _download_statuses = \
        cdseutils.utils.download_s3_files(
            s3_creds = s3_creds,
            s3paths = _s3paths,
            download_filepaths = _download_filepaths,
            overwrite = overwrite,
            logger = logger,
        )
        _download_successes = []
        for _download_status in _download_statuses:
            downloads_done += 1
            if _download_status in [
                cdseutils.utils.S3_DOWNLOAD_SUCCESS,
                cdseutils.utils.S3_DOWNLOAD_SKIPPED,
                cdseutils.utils.S3_DOWNLOAD_OVERWRITE,
            ]:
                _download_successes.append(True)
                if _download_status == cdseutils.utils.S3_DOWNLOAD_SUCCESS:
                    download_success_count += 1
                elif _download_status == cdseutils.utils.S3_DOWNLOAD_SKIPPED:
                    download_skipped_count += 1
                elif _download_status == cdseutils.utils.S3_DOWNLOAD_OVERWRITE:
                    download_overwrite_count += 1
            elif _download_status == cdseutils.utils.S3_DOWNLOAD_FAILED:
                _download_successes.append(False)
                download_failed_count += 1
        update_catalog(
            catalog_gdf = catalog_gdf,
            s3paths = _s3paths,
            download_filepaths = _download_filepaths,
            download_successes = _download_successes,
            sentinel2_local_catalog_filepath = sentinel2_local_catalog_filepath,
            satellite = satellite,
        )
        print(
            f'Download status: {downloads_done} / {N}\n'
            f'- Success: {download_success_count}\n'
            f'- Skipped: {download_skipped_count}\n'
            f'- Overwrite: {download_overwrite_count}\n'
            f'- Failed: {download_failed_count}'
        )


def download_sentinel2_tiles(
    cdse_creds:cdseutils.mydataclasses.Credentials,
    catalog_save_folderpath:str,
    root_download_folderpath:str,
    sentinel2_local_catalog_filepath:str,
    collection:sentinelhub.DataCollection,
    satellite:str,
    bands:list[str],
    catalog_gdf:gpd.GeoDataFrame = None,
    roi_filepath:str = None,
    startdate:datetime.datetime = None,
    enddate:datetime.datetime = None,
    max_cloudcover_threshold:float = None,
    upper_limit_for_number_of_tiles:int = 1,
    chunksize_for_download_and_update_catalog:int = 4 * 25,
):
    if collection.catalog_id != satellite:
        raise ValueError(
            "When this code was written (2024-08-19), the assumption was that "
            "sentinelhub.DataCollection's catalog_id member variable always remain static. "
            "That does not hold true anymore. Please refactor the code to take this dynamic "
            "nature of catalog_id variable into account."
        )

    os.makedirs(root_download_folderpath, exist_ok=True)
    if not os.access(root_download_folderpath, os.W_OK):
        raise PermissionError(
            f'Change root_download_folderpath as user does not '
            f'have permission to write to {root_download_folderpath}.'
        )
    
    if satellite == cdseutils.constants.Bands.S2L1C.NAME:
        all_bands = cdseutils.constants.Bands.S2L1C.ALL
    elif satellite == cdseutils.constants.Bands.S2L2A.NAME:
        all_bands = cdseutils.constants.Bands.S2L2A.ALL
    else:
        raise NotImplementedError(f'satellite = {satellite}')

    invalid_sentinel2_bands = set(bands) - set(all_bands)
    if len(invalid_sentinel2_bands) > 0:
        raise ValueError(f'Invalid {satellite} bands requested to be downloaded: {list(invalid_sentinel2_bands)}')

    MUST_PRESENT_CATALOG_COLS = [
        COL_ID, COL_SATELLITE, COL_TIMESTAMP, COL_S3URL, COL_GEOMETRY,
    ]

    if max_cloudcover_threshold is not None:
        MUST_PRESENT_CATALOG_COLS.append(COL_CLOUDCOVER)

    if not cdse_creds.is_sh_creds_defined():
        raise ValueError('cdse_creds does not have sh_clientid and sh_clientsecret defined.')
    if not cdse_creds.is_s3_creds_defined():
        raise ValueError('cdse_creds does not have s3_access_key and s3_secret_key defined.')

    fetch_catalog_params_present = roi_filepath is not None \
                                    and startdate is not None \
                                    and enddate is not None
    
    if catalog_gdf is None and not fetch_catalog_params_present:
        raise ValueError('If catalog_gdf is None, roi_filepath, startdate, enddate can not be None.')

    if catalog_gdf is not None:
        for col in MUST_PRESENT_CATALOG_COLS:
            if col not in catalog_gdf.columns:
                raise ValueError(
                    f'catalog_gdf does not have {col} column. '
                    f'catalog_gdf must have following columns: {MUST_PRESENT_CATALOG_COLS}'
                )
        if catalog_gdf['id'].value_counts().max() > 1:
            raise ValueError('catalog_gdf contains duplicate ids.')
    else:
        shapes_gdf = gpd.read_file(roi_filepath)
        catalog_gdf = cdseutils.utils.query_catalog(
            shapes_gdf = shapes_gdf,
            sh_creds = cdse_creds.sh_creds,
            collection = collection,
            startdate = startdate,
            enddate = enddate,
            cache_folderpath = catalog_save_folderpath,
        )
    
    if max_cloudcover_threshold is not None:
        catalog_gdf = catalog_gdf[catalog_gdf['cloud_cover'] <= max_cloudcover_threshold]

    number_of_tiles_to_download = catalog_gdf.shape[0]
    if number_of_tiles_to_download > upper_limit_for_number_of_tiles:
        raise ValueError(
            f'Are you sure you wish to download {number_of_tiles_to_download} image sets?\n'
            f'This is roughly {number_of_tiles_to_download * 725 / 1000} GB of download. '
            f'This exceeds upper_limit_for_number_of_tiles={upper_limit_for_number_of_tiles}.\n'
            'Kindly discuss with your team before you go ahead.'
        )

    print('Fetching s3paths for band files:')
    s3paths, download_filepaths = \
    cdseutils.sentinel2.get_s3paths(
        s3urls = catalog_gdf['s3url'],
        s3_creds = cdse_creds.s3_creds,
        root_folderpath = root_download_folderpath,
        bands = bands,
        satellite = satellite,
    )

    print(f'Downloading {len(s3paths)} files:')
    successful_download_count = \
    chunkwise_download_files_and_update_catalog(
        catalog_gdf = catalog_gdf,
        s3_creds = cdse_creds.s3_creds,
        s3paths = s3paths,
        download_filepaths = download_filepaths,
        sentinel2_local_catalog_filepath = sentinel2_local_catalog_filepath,
        satellite = satellite,
        chunksize = chunksize_for_download_and_update_catalog,
        overwrite = False,
        logger = None,
    )

    return successful_download_count, len(s3paths)


def download_sentinel2_l1c_tiles(
    cdse_creds:cdseutils.mydataclasses.Credentials,
    catalog_save_folderpath:str,
    root_download_folderpath:str,
    sentinel2_local_catalog_filepath:str,
    bands = cdseutils.constants.Bands.S2L1C.ALL, # changes with satellite
    catalog_gdf:gpd.GeoDataFrame = None,
    roi_filepath:str = None,
    startdate:datetime.datetime = None,
    enddate:datetime.datetime = None,
    max_cloudcover_threshold:float = None,
    upper_limit_for_number_of_tiles:int = 1,
    chunksize_for_download_and_update_catalog:int = 4 * 25,
):
    # changes with satellite
    SATELLITE = cdseutils.constants.Bands.S2L1C.NAME
    COLLECTION = sentinelhub.DataCollection.SENTINEL2_L1C

    return download_sentinel2_tiles(
        cdse_creds = cdse_creds,
        catalog_save_folderpath = catalog_save_folderpath,
        root_download_folderpath = root_download_folderpath,
        sentinel2_local_catalog_filepath = sentinel2_local_catalog_filepath,
        collection = COLLECTION,
        satellite = SATELLITE,
        bands = bands,
        catalog_gdf = catalog_gdf,
        roi_filepath = roi_filepath,
        startdate = startdate,
        enddate = enddate,
        max_cloudcover_threshold = max_cloudcover_threshold,
        upper_limit_for_number_of_tiles = upper_limit_for_number_of_tiles,
        chunksize_for_download_and_update_catalog = chunksize_for_download_and_update_catalog,
    )


def download_sentinel2_l2a_tiles(
    cdse_creds:cdseutils.mydataclasses.Credentials,
    catalog_save_folderpath:str,
    root_download_folderpath:str,
    sentinel2_local_catalog_filepath:str,
    bands = cdseutils.constants.Bands.S2L2A.ALL, # changes with satellite
    catalog_gdf:gpd.GeoDataFrame = None,
    roi_filepath:str = None,
    startdate:datetime.datetime = None,
    enddate:datetime.datetime = None,
    max_cloudcover_threshold:float = None,
    upper_limit_for_number_of_tiles:int = 1,
    chunksize_for_download_and_update_catalog:int = 4 * 25,
):
    # changes with satellite
    SATELLITE = cdseutils.constants.Bands.S2L2A.NAME
    COLLECTION = sentinelhub.DataCollection.SENTINEL2_L2A

    return download_sentinel2_tiles(
        cdse_creds = cdse_creds,
        catalog_save_folderpath = catalog_save_folderpath,
        root_download_folderpath = root_download_folderpath,
        sentinel2_local_catalog_filepath = sentinel2_local_catalog_filepath,
        collection = COLLECTION,
        satellite = SATELLITE,
        bands = bands,
        catalog_gdf = catalog_gdf,
        roi_filepath = roi_filepath,
        startdate = startdate,
        enddate = enddate,
        max_cloudcover_threshold = max_cloudcover_threshold,
        upper_limit_for_number_of_tiles = upper_limit_for_number_of_tiles,
        chunksize_for_download_and_update_catalog = chunksize_for_download_and_update_catalog,
    )

