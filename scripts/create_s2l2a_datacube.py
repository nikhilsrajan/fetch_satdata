import os
import argparse
import pandas as pd
import geopandas as gpd
import time

import config
from init_logger import get_logger

import sys
sys.path.append('..')

import sqlite_db_utils
import register_configuration as rc
import register_polygons as rp
import create_sentinel2_datacube as cs2d
import cdseutils.constants
import rsutils.s2_grid_utils


logger = get_logger()


COL_ID = 'id'
COL_ROI = 'roi'
COL_STARTDATE = 'startdate'
COL_ENDDATE = 'enddate'
COL_CONFIGID = 'config_id'
COL_LOCALFOLDERPATH = 'local_folderpath'
COL_FILES = 'files'
COL_LASTUPDATE = 'last_update'


def init_db():
    sqlite_db_utils.init_db(
        database = config.FILEPATH_DATACUBE_CATALOG_DB,
        db_structure = {
            config.S2L2A_TABLE: {
                COL_ID: 'TEXT UNIQUE',
                COL_ROI: 'TEXT',
                COL_STARTDATE: 'TEXT',
                COL_ENDDATE: 'TEXT',
                COL_CONFIGID: 'INTEGER',
                COL_LOCALFOLDERPATH: 'TEXT',
                COL_FILES: 'TEXT',
                COL_LASTUPDATE: 'TEXT',
            }
        },
        id_col = COL_ID,
    )


def get_datacube_id(
    roi:str,
    startdate:pd.Timestamp,
    enddate:pd.Timestamp,
    config_id:int,
):
    '_'.join([
        roi,
        startdate.strftime('%Y%m%dT%H%M%S'),
        enddate.strftime('%Y%m%dT%H%M%S'),
        str(config_id)
    ])


def check_if_datacube_exists_in_record(
    roi:str,
    startdate:pd.Timestamp,
    enddate:pd.Timestamp,
    config_id:int,
):
    init_db()
    
    datacube_id = get_datacube_id(
        roi = roi,
        startdate = startdate,
        enddate = enddate,
        config_id = config_id,
    )

    fetch_df = sqlite_db_utils.fetch_rows_from_db(
        database = config.FILEPATH_DATACUBE_CATALOG_DB,
        table = config.S2L2A_TABLE,
        columns = [COL_ID, COL_LOCALFOLDERPATH],
        ids = [datacube_id],
        id_col = COL_ID,
    )

    if fetch_df.shape[0] == 0:
        return False, None
    
    return True, fetch_df[COL_LOCALFOLDERPATH][0]


def record_datacube(
    roi:str,
    startdate:pd.Timestamp,
    enddate:pd.Timestamp,
    config_id:int,
    export_folderpath:str,
    overwrite:bool,
):
    init_db()

    datacube_id = get_datacube_id(
        roi = roi,
        startdate = startdate,
        enddate = enddate,
        config_id = config_id,
    )
    last_update = sqlite_db_utils.ts_to_str(ts=pd.Timestamp.now(tz='UTC'))
    startdate = sqlite_db_utils.ts_to_str(startdate)
    enddate = sqlite_db_utils.ts_to_str(enddate)
    files = ','.join([
        'datacube.npy',
        'metadata.pickle.npy',
        'mean_sun_angle.csv',
    ])
    

    if check_if_datacube_exists_in_record(
        roi = roi,
        startdate = startdate,
        enddate = enddate,
        config_id = config_id,
    )[0] and overwrite:
        sqlite_db_utils.update_value_in_db(
            database = config.FILEPATH_DATACUBE_CATALOG_DB,
            table = config.S2L2A_TABLE,
            id = datacube_id,
            id_col = COL_ID,
            col = COL_LASTUPDATE,
            update_value = last_update,
        )
    
    else:
        sqlite_db_utils.insert_rows_to_db(
            database = config.FILEPATH_DATACUBE_CATALOG_DB,
            table = config.S2L2A_TABLE,
            data_dicts = [{
                COL_ID: datacube_id,
                COL_ROI: roi,
                COL_STARTDATE: startdate,
                COL_ENDDATE: enddate,
                COL_CONFIGID: config_id,
                COL_LOCALFOLDERPATH: export_folderpath,
                COL_FILES: files,
                COL_LASTUPDATE: last_update,
            }]
        )


def get_shapes_gdf(roi:str):
    geom_type, geom_id = roi.split('=')
    if geom_type not in ['s2grid', 'geom']:
        raise ValueError(f'Invalid roi={roi}. roi must start with s2grid= or geom=')
    
    if geom_type == 's2grid':
        geom = rsutils.s2_grid_utils.get_grid_geometry_from_id(grid_id = geom_id)
    else:
        geom = sqlite_db_utils.fetch_value_in_db(
            database = config.FILEPATH_POLYGONS_REGISTRY_DB,
            table = config.GEOMETRIES_TABLE,
            id = geom_id,
            id_col = rp.COL_ID,
            col = rp.COL_GEOMETRY,
        )

    shapes_gdf = gpd.GeoDataFrame(
        data = {'geometry': [geom]},
        crs = 'epsg:4326',
    )

    return shapes_gdf


if __name__ == '__main__':
    start_time = time.time()

    parser = argparse.ArgumentParser(
        prog = 'python create_s2l2a_datacube.py',
        description = (
            'Script to create sentinel-2-l2a datacube.'
        ),
        epilog = f"--- Send your complaints to {','.join(config.MAINTAINERS)} ---",
    )
    parser.add_argument('roi', help="If ROI is an s2grid then pass 's2grid=GRID_ID'. Otherwise have the shape registered using register_polygon.py and pass 'geom=GEOM_ID'.")
    parser.add_argument('startdate', help='style: 2024-08-20 or 2024-08-20T13:54:40.022Z')
    parser.add_argument('enddate', help='style: 2024-08-20 or 2024-08-20T13:54:40.022Z')
    parser.add_argument('config_id', help='Make sure to have configuration registered using the script register_configurations.py - the ID that is printed is what needs to be passed.')
    parser.add_argument('export_folderpath', help='Folderpath where the datacube would be saved to.')
    parser.add_argument('-j', '--njobs', required=False, action='store', default=1, help='[default = 1] Number of parallel jobs')
    parser.add_argument('-v', '--verbose', required=False, action='store_true')
    parser.add_argument('--overwrite', required=False, action='store_true')
    parser.add_argument('--record', required=False, action='store_true', help='Whether to record the creation of the datacube to datacube catalog')

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel('INFO')

    roi = str(args.roi)
    startdate = pd.to_datetime(str(args.startdate))
    enddate = pd.to_datetime(str(args.enddate))
    config_id = int(args.config_id)
    export_folderpath = str(args.export_folderpath)
    njobs = int(args.njobs)

    if args.record:
        datacube_id = get_datacube_id(
            roi = roi,
            startdate = startdate,
            enddate = enddate,
            config_id = config_id,
        )

        exists_in_record, existing_folderpath = \
        check_if_datacube_exists_in_record(
            roi = roi,
            startdate = startdate,
            enddate = enddate,
            config_id = config_id,
        )

        if exists_in_record:
            if not os.path.samefile(existing_folderpath, export_folderpath):
                raise ValueError(
                    f'datacube_id={datacube_id} exists at {existing_folderpath} and is recorded. '
                    'Use the same export_folderpath as where it already exists.'
                )

        if not args.overwrite:
            exit(0)


    config_desc = rc.fetch_s2l2a_config(config_id = config_id)[config_id]

    mosaic_days = config_desc[rc.COL_MOSAIC_DAYS]
    scl_masks = rc.str_to_list_int(config_desc[rc.COL_SCL_MASKS])

    shapes_gdf = get_shapes_gdf(roi = roi)

    cs2d.create_s2l2a_datacube(
        shapes_gdf = shapes_gdf,
        export_folderpath = export_folderpath,
        catalog_db_filepath = config.FILEPATH_SATELLITE_CATALOG_DB,
        table_name = config.S2L2A_TABLE,
        startdate = startdate,
        enddate = enddate,
        bands = cdseutils.constants.Bands.S2L2A.ALL ,
        njobs = njobs,
        mosaic_days = mosaic_days,
        scl_mask_classes = scl_masks,
        print_messages = False,
        logger = logger,
        overwrite = args.overwrite,
    )

    end_time = time.time()

    if args.record:
        logger.info('Inserting record into DB')
        record_datacube(
            roi = roi,
            startdate = startdate,
            enddate = enddate,
            config_id = config_id,
            overwrite = args.overwrite,
            export_folderpath = export_folderpath,
        )

    logger.info(f'--- t_elapsed: {round(end_time - start_time, 2)} s ---')
