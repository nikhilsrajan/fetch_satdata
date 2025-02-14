import os
import geopandas as gpd
import argparse

import config
from init_logger import get_logger

import sys
sys.path.append('..')

import sqlite_db_utils


logger = get_logger()


TOLERANCE = 1e-6

COL_ID = 'id'
COL_GEOMETRY = 'geometry'


def check_sanity_of_polygons_gdf(
    polygons_gdf:gpd.GeoDataFrame,
    id_col:str,
    tolerance:float = TOLERANCE,
):
    id_counts_df = polygons_gdf[id_col].value_counts().reset_index()
    duplicate_ids = list(id_counts_df[id_counts_df['count'] > 1][id_col])

    if len(duplicate_ids) > 0:
        msg = f'{len(duplicate_ids)} duplicate ids found'
        msg += ' : ' + ', '.join(duplicate_ids[:3])
        if len(duplicate_ids) > 3:
            msg += ', ...'
        logger.error(msg)
        raise ValueError(msg)

    geom_id_list_dict = polygons_gdf.groupby('geometry')[id_col].apply(list).to_dict()
    conflicting_ids = []
    for geom, id_list in geom_id_list_dict.items():
        if len(id_list) > 1:
            conflicting_ids.append(tuple(id_list))
    
    if len(conflicting_ids) > 0:
        msg = f'{len(conflicting_ids)} conflicing ids for same geometries found'
        msg += ' : ' + ', '.join(conflicting_ids[:3])
        if len(conflicting_ids) > 3:
            msg += ', ...'
        logger.error(msg)
        raise ValueError(msg)
    

def fetch_polygons_db():
    # if doesn't exists then creates and empty one.
    if not os.path.exists(config.FILEPATH_POLYGONS_REGISTRY_DB):
        sqlite_db_utils.create_db(
            database = config.FILEPATH_POLYGONS_REGISTRY_DB,
            db_structure = {
                config.GEOMETRIES_TABLE : {
                    COL_ID: 'TEXT UNIQUE',
                    COL_GEOMETRY: 'TEXT',
                }
            },
            id_col = COL_ID,
        )
    
    return sqlite_db_utils.fetch_rows_from_db(
        database = config.FILEPATH_POLYGONS_REGISTRY_DB,
        table = config.GEOMETRIES_TABLE,
    )


def check_conflict_in_db(
    polygons_gdf:gpd.GeoDataFrame,
    id_col:str,
    tolerance:float = TOLERANCE,
) -> gpd.GeoDataFrame:
    """
    Checks for conflicts in ids and geometries and drops the ids that are already present.
    """

    registry_gdf = fetch_polygons_db()

    # checking ids that are already in the registry
    registry_ids = set(registry_gdf[COL_ID])
    new_ids = set(polygons_gdf[id_col])

    common_ids = registry_ids & new_ids

    polygons_common_ids_gdf = polygons_gdf[polygons_gdf[id_col].isin(common_ids)].sort_values(by=id_col).reset_index(drop=True)
    registry_common_ids_gdf = registry_gdf[registry_gdf[COL_ID].isin(common_ids)].sort_values(by=COL_ID).reset_index(drop=True)

    geom_match = [   
        not incoming_poly.equals_exact(registry_poly, tolerance) for incoming_poly, registry_poly in 
        zip(polygons_common_ids_gdf[COL_GEOMETRY], registry_common_ids_gdf[COL_GEOMETRY])
    ]

    ids_that_exist_under_different_geometries = []
    if registry_common_ids_gdf[geom_match].shape[0] > 0:
        ids_that_exist_under_different_geometries = list(registry_common_ids_gdf[geom_match][COL_ID])

    if len(ids_that_exist_under_different_geometries) > 0:
        msg = f'{len(ids_that_exist_under_different_geometries)} duplicate ids for different geometries found in registry'
        msg += ' : ' + ', '.join(ids_that_exist_under_different_geometries[:3])
        if len(ids_that_exist_under_different_geometries) > 3:
            msg += ', ...'
        logger.error(msg)
        raise ValueError(msg)

    # checking if geometries are already in registry
    polygons_new_ids_gdf = polygons_gdf[~polygons_gdf[id_col].isin(common_ids)].sort_values(by=id_col)
    registry_unchecked_ids_gdf = registry_gdf[~registry_gdf[COL_ID].isin(common_ids)].sort_values(by=COL_ID)

    id_polygon_dict = dict(zip(
        polygons_new_ids_gdf[id_col], polygons_new_ids_gdf[COL_GEOMETRY],
    ))
    id_polygon_dict.update(dict(zip(
        registry_unchecked_ids_gdf[COL_ID], registry_unchecked_ids_gdf[COL_GEOMETRY],
    )))

    incoming_and_registry_sjoin_gdf = gpd.sjoin(
        polygons_new_ids_gdf[[id_col, COL_GEOMETRY]].rename(columns={id_col: 'incoming_id'}), 
        registry_unchecked_ids_gdf[[COL_ID, COL_GEOMETRY]].rename(columns={COL_ID: 'registered_id'}),
    )[['incoming_id', 'registered_id']]

    conflicting_ids = []

    if incoming_and_registry_sjoin_gdf.shape[0] > 0:
        for index, row in incoming_and_registry_sjoin_gdf.iterrows():
            incoming_id = row['incoming_id']
            registered_id = row['registered_id']

            if id_polygon_dict[incoming_id] == id_polygon_dict[registered_id]:
                conflicting_ids.append((incoming_id, registered_id))

    if len(conflicting_ids) > 0:
        msg = f"{len(conflicting_ids)} conflicing ids for same geometries found in registry"
        msg += ' : ' + ', '.join([str(item) for item in conflicting_ids[:3]])
        if len(conflicting_ids) > 3:
            msg += ', ...'
        logger.error(msg)
        raise ValueError(msg)

    return polygons_gdf[~polygons_gdf[id_col].isin(common_ids)]


def register_polygons_to_db(
    polygons_gdf:gpd.GeoDataFrame,
    id_col:str,
):
    insert_batches = [[]]

    current_batch_size = 0
    for index, row in polygons_gdf.iterrows():
        sql_row = sqlite_db_utils.gpd_to_sql_row(
            row_dict = {
                COL_ID: row[id_col],
                COL_GEOMETRY: row[COL_GEOMETRY],
            }
        )
        insert_batches[-1].append(sql_row)
        current_batch_size += 1

        if current_batch_size == 1000:
            insert_batches.append([])
            current_batch_size = 0


    for insert_batch in insert_batches:
        sqlite_db_utils.insert_rows_to_db(
            database = config.FILEPATH_POLYGONS_REGISTRY_DB,
            table = config.GEOMETRIES_TABLE,
            data_dicts = insert_batch,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog = 'python register_polygons.py',
        description = (
            'Script to register polygon. Creates / updates polygons.db that is present in the '
            'root folder of datacubes folder. This is not to register standard polygons like s2 grids.'
        ),
        epilog = f"--- Send your complaints to {','.join(config.MAINTAINERS)} ---",
    )
    parser.add_argument('polygons_filepath', help="Path to file containing polygons and names.")
    parser.add_argument('--id-col', required=False, action='store', help="[default = 'id'] Column containing ids for the polygons")
    parser.add_argument('-v', '--verbose', action='store_true')
    
    args = parser.parse_args()

    polygons_filepath = str(args.polygons_filepath)
    id_col = str(args.id_col)

    polygons_gdf = gpd.read_file(polygons_filepath).to_crs('epsg:4326')

    if args.verbose:
        logger.setLevel('INFO')

    original_size = polygons_gdf.shape[0]
    
    logger.info(f'Checking id and geometry sanity within the input polygons file: {polygons_filepath}')
    check_sanity_of_polygons_gdf(polygons_gdf = polygons_gdf, id_col = id_col)

    logger.info(f'Checking id and geometry sanity between the input polygons file and polygons registry')
    polygons_gdf = check_conflict_in_db(polygons_gdf = polygons_gdf, id_col = id_col)

    new_size = polygons_gdf.shape[0]

    dropped_size = original_size - new_size

    if dropped_size > 0:
        logger.info(f'Dropped the ids that were already registered. Drop count = {dropped_size}')
    
    if new_size > 0:
        logger.info(f'Registering {new_size} polygons.')
        register_polygons_to_db(polygons_gdf = polygons_gdf, id_col = id_col)
    else:
        logger.info(f'No new polygons left to register.')
