import os
import pandas as pd
import geopandas as gpd
import datetime
import sentinelhub
import shapely
import logging

import sys
sys.path.append('..')

import cdseutils.mydataclasses
import cdseutils.utils
import cdseutils.constants
import cdseutils.sentinel2


class CatalogManager(object):
    COLUMNS = [
        'id', 'timestamp', 's3url', 'local_folderpath', 
        'files', 'last_update', 'cloud_cover', 'geometry',
    ]
    EPSG_4326 = 'epsg:4326'


    def __init__(self, catalog_filepath:str):
        self.catalog_filepath = catalog_filepath

        folderpath = os.path.split(catalog_filepath)[0]
        os.makedirs(folderpath, exist_ok=True)
        if not os.access(folderpath, os.W_OK):
            raise ValueError(
                f'User does not have the permission to write catalog '
                f'in folderpath={folderpath}. Set a new catalog_filepath.'
            )

        if os.path.exists(self.catalog_filepath):
            self.catalog_gdf:gpd.GeoDataFrame = gpd.read_file(self.catalog_filepath)
            self.catalog_gdf = self.catalog_gdf.set_index('id')
        else:
            # it was difficult to figure out how to set the dtypes for each columns correctly
            # the work around implemented is to initialise with placeholders and let geopandas
            # figure out the dtype by itself.
            self.catalog_gdf = gpd.GeoDataFrame(
                data = {
                    'id': ['blah'],
                    'timestamp': [pd.Timestamp('2021-01-28 07:57:53.635000+0000', tz='UTC')],
                    's3url': ['blah'],
                    'local_folderpath': ['blah'],
                    'files': ['blah'],
                    'last_update': [pd.Timestamp('2021-01-28 07:57:53.635000+0000', tz='UTC')],
                    'cloud_cover': [0.0],
                    'geometry': [
                        shapely.MultiPolygon([
                            shapely.Polygon([
                                (38.09192035575927, 8.141009568663828), 
                                (39.08858871535991, 8.142015435187918), 
                                (39.088384163693355, 7.148816474425373), 
                                (38.09401676842014, 7.147934625992314), 
                                (38.09192035575927, 8.141009568663828),
                            ])
                        ])
                    ],
                },
                crs = CatalogManager.EPSG_4326
            ).set_index('id').drop(index='blah')

    
    @staticmethod
    def get_current_timestamp():
        return pd.Timestamp(datetime.datetime.now(), tz='UTC')
    

    def add(
        self, 
        _id:str, 
        timestamp:pd.Timestamp = None,
        s3url:str = None,
        local_folderpath:str = None,
        files:list[str] = None,
        cloud_cover:float = None,
        geometry_epsg_4326:shapely.Geometry = None,
    ):
        performed_update = False

        entry_modified_dict = {}

        first_entry = _id not in self.catalog_gdf.index

        for col, value in [
            ('timestamp', timestamp),
            ('s3url', s3url),
            ('local_folderpath', local_folderpath),
            ('files', files),
            ('cloud_cover', cloud_cover),
            ('geometry', geometry_epsg_4326),
        ]:
            entry_modified_dict[col] = False
            if value is None:
                if first_entry:
                    raise ValueError(
                        f'First entry for col={col} can not be None for id={_id}'
                    )
                else:
                    continue
            else:
                if col == 'files':
                    updated_files = set(value)
                    if not first_entry:
                        current_files = self.catalog_gdf.loc[_id, col].split(',')
                        if current_files != value:
                            entry_modified_dict[col] = True
                        updated_files = set(current_files) | set(value)
                    else:
                        entry_modified_dict[col] = True
                    self.catalog_gdf.loc[_id, col] = ','.join(updated_files)
                else:
                    if not first_entry:
                        if self.catalog_gdf.loc[_id, col] != value:
                            entry_modified_dict[col] = True
                    else:
                        entry_modified_dict[col] = True
                    self.catalog_gdf.loc[_id, col] = value
                        
        
        performed_update = all(entry_modified_dict.values())

        if performed_update:
            self.catalog_gdf.loc[_id, 'last_update'] = CatalogManager.get_current_timestamp()
        
        return performed_update
    

    def modify_files(self, _id:str, delete_files:list[str]=None, add_files:list[str]=None):
        if pd.isna(self.catalog_gdf.loc[_id, 'files']):
            raise KeyError(
                f'id={_id} not present in the catalog. Can not perform delete_files.'
            )
    
        if delete_files is None:
            delete_files = set()
        else:
            delete_files = set(delete_files)

        if add_files is None:
            add_files = set()
        else:
            add_files = set(add_files)
        
        current_files = set(self.catalog_gdf.loc[_id, 'files'].split(','))
        
        conflicting_files = delete_files & add_files
        absent_files_to_delete = delete_files - current_files
        present_files_to_add = current_files & add_files

        if len(conflicting_files) > 0:
            raise ValueError(f'Conflicting requests. Following files being added and deleted: {list(conflicting_files)}')

        if len(absent_files_to_delete) > 0:
            raise ValueError(f'Non-existent files requested to be deleted: {list(absent_files_to_delete)}')
        
        if len(present_files_to_add) > 0:
            raise ValueError(f'Pre-existent files requested to be added: {list(present_files_to_add)}')

        updated_files = (current_files | add_files) - delete_files

        if updated_files != current_files:
            self.catalog_gdf.loc[_id, 'last_update'] = CatalogManager.get_current_timestamp()

        self.catalog_gdf.loc[_id, 'files'] = ','.join(updated_files)
        
        return len(updated_files)

    
    def delete_entry(self, _id:str):
        if _id not in self.catalog_gdf.index:
            raise KeyError(f'id={_id} not present in the catalog. Can not perform delete_entry.')
        self.catalog_gdf = self.catalog_gdf.drop(index=_id)
    

    def save(self):
        self.catalog_gdf.to_file(self.catalog_filepath)


def update_catalog(
    catalog_gdf:gpd.GeoDataFrame, # contains other info that needs to be stored in local catalog
    s3paths:list[cdseutils.mydataclasses.S3Path],
    download_filepaths:list[str],
    download_successes:list[bool],
    sentinel2_local_catalog_filepath:str,
):
    catalog_manager = CatalogManager(
        catalog_filepath = sentinel2_local_catalog_filepath,
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
        
        band_filename = parsed_s3_url['band_filename']
        parsed_band_filename = cdseutils.sentinel2.parse_band_filename(
            sentinel2_band_filename=band_filename
        )
        local_band_filename = parsed_band_filename['band'] + parsed_band_filename['ext']

        id_to_download_files[_id].append(local_band_filename)
        id_to_download_folderpath[_id] = os.path.split(download_filepath)[0]

    _ids = id_to_download_files.keys()
    selected_catalog_gdf = catalog_gdf[catalog_gdf['id'].isin(_ids)]

    id_to_timestamps = dict(zip(selected_catalog_gdf['id'], selected_catalog_gdf['timestamp']))
    id_to_s3urls = dict(zip(selected_catalog_gdf['id'], selected_catalog_gdf['s3url']))
    id_to_cloudcover = dict(zip(selected_catalog_gdf['id'], selected_catalog_gdf['cloud_cover']))
    id_to_geometry = dict(zip(selected_catalog_gdf['id'], selected_catalog_gdf['geometry']))

    for _id in _ids:
        catalog_manager.add(
            _id = _id,
            timestamp = id_to_timestamps[_id],
            s3url = id_to_s3urls[_id],
            local_folderpath = id_to_download_folderpath[_id],
            files = id_to_download_files[_id],
            cloud_cover = id_to_cloudcover[_id],
            geometry_epsg_4326 = id_to_geometry[_id],
        )
    
    catalog_manager.save()


def chunkwise_download_files_and_update_catalog(
    catalog_gdf:gpd.GeoDataFrame,
    s3_creds:cdseutils.mydataclasses.S3Credentials,
    s3paths:list[cdseutils.mydataclasses.S3Path],
    download_filepaths:list[str],
    sentinel2_local_catalog_filepath:str,
    chunksize = 4 * 25,
    overwrite:bool = False,
    logger:logging.Logger = None,
):
    N = len(s3paths)
    successful_download_count = 0
    for i in range(0, N, chunksize):
        _s3paths = s3paths[i:i+chunksize]
        _download_filepaths = download_filepaths[i:i+chunksize]
        _download_successes = \
        cdseutils.utils.download_s3_files(
            s3_creds = s3_creds,
            s3paths = _s3paths,
            download_filepaths = _download_filepaths,
            overwrite = overwrite,
            logger = logger,
        )
        successful_download_count += sum(_download_successes)
        update_catalog(
            catalog_gdf = catalog_gdf,
            s3paths = _s3paths,
            download_filepaths = _download_filepaths,
            download_successes = _download_successes,
            sentinel2_local_catalog_filepath = sentinel2_local_catalog_filepath,
        )
    return successful_download_count
    

def download_sentinel2_l1c_tiles(
    cdse_creds:cdseutils.mydataclasses.Credentials,
    catalog_save_folderpath:str,
    root_download_folderpath:str,
    sentinel2_local_catalog_filepath:str,
    bands = cdseutils.constants.Bands.Sentinel2.ALL,
    catalog_gdf:gpd.GeoDataFrame = None,
    roi_filepath:str = None,
    startdate:datetime.datetime = None,
    enddate:datetime.datetime = None,
    max_cloudcover_threshold:float = None,
    upper_limit_for_number_of_tiles:int = 300,
    chunksize_for_download_and_update_catalog:int = 4 * 25,
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

    S2L1C = 'sentinel-2-l1c'

    if sentinelhub.DataCollection.SENTINEL2_L1C.catalog_id != S2L1C:
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
    
    invalid_sentinel2_bands = set(bands) - set(cdseutils.constants.Bands.Sentinel2.ALL)
    if len(invalid_sentinel2_bands) > 0:
        raise ValueError(f'Invalid Sentinel-2 L1C bands requested to be downloaded: {list(invalid_sentinel2_bands)}')

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
        if catalog_gdf['id'].value_counts().max() > 1:
            raise ValueError('catalog_gdf contains duplicate ids.')
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
        
        intersecting_ids = gpd.sjoin(
            catalog_gdf, shapes_gdf[['geometry']].to_crs(catalog_gdf.crs)
        )['id'].unique()

        catalog_gdf = catalog_gdf[catalog_gdf['id'].isin(intersecting_ids)].reset_index(drop=True)

        if catalog_gdf['id'].value_counts().max() > 1:
            raise ValueError(
                "CRITICAL ERROR: When this code was written (2024-08-19), the assumption "
                "was that id obtained from sentinelhub catalog was always going to be unique. "
                "This does not hold true anymore. This is a critical issue as the folder "
                "structure for locally storing sentinel-2 tiles were based on the assumption "
                "that the id would be unique. Heavy refactoring needs to be performed to "
                "take into consideration this new found bug."
            )
    
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

    s3paths, download_filepaths = \
    cdseutils.sentinel2.get_s3paths(
        s3urls = catalog_gdf['s3url'],
        s3_creds = cdse_creds.s3_creds,
        root_folderpath = root_download_folderpath,
        bands = bands,
    )

    print('Downloading files:')
    successful_download_count = \
    chunkwise_download_files_and_update_catalog(
        catalog_gdf = catalog_gdf,
        s3_creds = cdse_creds.s3_creds,
        s3paths = s3paths,
        download_filepaths = download_filepaths,
        sentinel2_local_catalog_filepath = sentinel2_local_catalog_filepath,
        chunksize = chunksize_for_download_and_update_catalog,
        overwrite = False,
        logger = None,
    )

    print(f'Successful downloads: {successful_download_count} / {len(s3paths)}')

    return successful_download_count, len(s3paths)

