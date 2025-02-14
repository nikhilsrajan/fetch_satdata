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

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel('INFO')

    roi = str(args.roi)
    startdate = pd.to_datetime(str(args.startdate))
    enddate = pd.to_datetime(str(args.enddate))
    config_id = int(args.config_id)
    export_folderpath = str(args.export_folderpath)
    njobs = int(args.njobs)

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
        print_messages = True,
        logger = logger,
        # if_missing_files = ...,
        overwrite = args.overwrite,
    )

    end_time = time.time()

    logger.info(f'--- t_elapsed: {round(end_time - start_time, 2)} s ---')
