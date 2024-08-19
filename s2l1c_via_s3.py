import os
import geopandas as gpd
import datetime
import sentinelhub

import sys
sys.path.append('..')

import cdseutils.mydataclasses
import cdseutils.utils


def download_sentinel2_l1c_tiles(
    cdse_creds:cdseutils.mydataclasses.Credentials,
    catalog_save_folderpath:str,
    root_download_folderpath:str,
    catalog_gdf:gpd.GeoDataFrame = None,
    roi_filepath:str = None,
    startdate:datetime.datetime = None,
    enddate:datetime.datetime = None,
    max_cloudcover_threshold:float = None,
    upper_limit_for_number_of_tiles:int = 300,
):
    """
    Function to search the catalog for sentinel-2-l1c data for a give shape file saved
    at roi_filepath, for a given daterange - startdate to enddate, and download the tiles
    to a specified folderpath.

    If catalog_gdf is specified, fetch_catalog operation is bypassed.

    If max_cloudcover_threshold is not None, an additional mandatory column is added 'cloud_cover'.
    
    fetch_catalog function caches the results thus catalog_save_folderpath is a critical
    parameter to reduce the number of api calls.
    """

    os.makedirs(root_download_folderpath, exist_ok=True)
    if not os.access(root_download_folderpath, os.W_OK):
        raise PermissionError(
            f'Change root_download_folderpath as user does not '
            f'have permission to write to {root_download_folderpath}.'
        )

    MUST_PRESENT_CATALOG_COLS = [
        'id', 'timestamp', 's3url', 'geometry', 
    ]

    if max_cloudcover_threshold is not None:
        MUST_PRESENT_CATALOG_COLS.append('cloud_cover')

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
    else:
        shapes_gdf = gpd.read_file(roi_filepath)
        bboxes = cdseutils.utils.get_bboxes(shapes_gdf=shapes_gdf)
        # fetch_catalog is caching the results, cache_folderpath is a
        # critical parameter to reduce api calls.
        catalog_gdf, results = cdseutils.utils.fetch_catalog(
            bboxes = bboxes,
            sh_creds = cdse_creds.sh_creds,
            collection = sentinelhub.DataCollection.SENTINEL2_L1C,
            startdate = startdate,
            enddate = enddate,
            cache_folderpath = catalog_save_folderpath,
        )
        catalog_gdf['cloud_cover'] = [x['properties']['eo:cloud_cover'] for x in results]
    
    if max_cloudcover_threshold is not None:
        catalog_gdf = catalog_gdf[catalog_gdf['cloud_cover'] <= max_cloudcover_threshold]

    number_of_tiles_to_download = catalog_gdf.shape[0]
    if number_of_tiles_to_download > upper_limit_for_number_of_tiles:
        raise ValueError(
            f'Are you sure you wish to download {number_of_tiles_to_download} image sets?\n'
            f'This is roughly {number_of_tiles_to_download * 700 / 1000} GB of download. '
            f'This exceeds upper_limit_for_number_of_tiles={upper_limit_for_number_of_tiles}.\n'
            'Kindly discuss with your team before you go ahead.'
        )

    
    

    


    

