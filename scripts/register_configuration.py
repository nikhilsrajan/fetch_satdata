import os
import argparse

import config
from init_logger import get_logger

import sys
sys.path.append('..')

import sqlite_db_utils


logger = get_logger()


SATELLITE_S2L2A = 'sentinel-2-l2a'

COL_ID = 'id'
COL_MOSAIC_DAYS = 'mosaic_days'
COL_SCL_MASKS = 'scl_masks'

DB_STRUCTURE = {
    # table: {col : type}
    SATELLITE_S2L2A: {
        COL_ID: 'INTEGER UNIQUE',
        COL_MOSAIC_DAYS: 'INTEGER',
        COL_SCL_MASKS: 'TEXT'
    },
}

VALID_SATELLITES = [
    SATELLITE_S2L2A,
]

DEFAULT_S2L2A_MASKS = '0,1,3,7,8,9,10'


def str_to_list_int(list_int_str:str):
    list_int = [int(v) for v in list_int_str.split(',')]
    list_int.sort()
    return list_int


def list_int_to_str(list_int:list[int]):
    list_int.sort()
    list_int_str = ','.join([str(v) for v in list_int])
    return list_int_str


def fetch_configs(
    table:str,
    query:str,
):
    config_df = sqlite_db_utils.fetch_rows_from_db(
        database = config.FILEPATH_CONFIG_REGISTRY_DB,
        table = table,
        query = query,
    )

    configs = {}
    for index, row in config_df.iterrows():
        configs[row[COL_ID]] = dict(row)
        del configs[row[COL_ID]][COL_ID]
    
    return configs


def generate_query(
    table:str,
    query_dict:dict,
):
    query = f"SELECT * FROM '{table}'"
    condition = ''
    for col, q_val in query_dict.items():
        if q_val is None:
            continue
        if len(condition) == 0:
            condition += " WHERE"
        else:
            condition += " AND"
        condition += f" {col} == {q_val}"

    query += condition

    return query


def fetch_s2l2a_config(
    config_id:int = None,
    mosaic_days:int = None,
    scl_masks:list[int] = None,
):
    if scl_masks is not None:
        scl_masks = list_int_to_str(scl_masks)

    return fetch_configs(
        table = SATELLITE_S2L2A,
        query = generate_query(
            table = SATELLITE_S2L2A,
            query_dict = {
                COL_ID: config_id,
                COL_MOSAIC_DAYS: mosaic_days,
                COL_SCL_MASKS: scl_masks,
            },
        )
    )


def register_s2l2a_config(
    mosaic_days:int,
    scl_masks:list[int],
):
    configs = fetch_s2l2a_config()

    config_already_registered = False

    scl_masks = list_int_to_str(scl_masks)

    if len(configs) == 0:
        config_id = 1
    else:
        for _id, _config in configs.items():
            if _config == {COL_MOSAIC_DAYS: mosaic_days, COL_SCL_MASKS: scl_masks}:
                config_already_registered = True
                config_id = _id
                break
        if not config_already_registered:
            config_id = max(configs.keys()) + 1

    config_to_register = {
        COL_ID: config_id,
        COL_MOSAIC_DAYS: mosaic_days,
        COL_SCL_MASKS: scl_masks,
    }

    if not config_already_registered:
        sqlite_db_utils.insert_rows_to_db(
            database = config.FILEPATH_CONFIG_REGISTRY_DB,
            table = SATELLITE_S2L2A,
            data_dicts = [config_to_register],
        )
    
    return config_id, config_to_register, config_already_registered


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog = 'python register_configuration.py',
        description = (
            'Script to register configuration for a satellite. Creates / updates configurations.db '
            'that is present in the root folder of datacubes folder.'
        ),
        epilog = f"--- Send your complaints to {','.join(config.MAINTAINERS)} ---",
    )
    parser.add_argument('satellite', help=f"options: {VALID_SATELLITES}.")
    parser.add_argument('--mosaic', required=True, action='store', help='mosaicing days')
    parser.add_argument('--s2l2a-masks', required=False, action='store', default='0,1,3,7,8,9,10', help=f'[default = {DEFAULT_S2L2A_MASKS}] SCL classes that would be masked.')

    args = parser.parse_args()

    satellite = str(args.satellite).lower()
    if satellite not in VALID_SATELLITES:
        msg = f'Invalid satellite={satellite}. satellite must be from {VALID_SATELLITES}'
        logger.error(msg)
        raise ValueError(msg)

    mosaic = int(args.mosaic)
    if mosaic < 0:
        msg = f'Invalid mosaic={mosaic}. mosaic must be > 0.'
        logger.error(msg)
        raise ValueError(msg)
    
    sqlite_db_utils.init_db(
        database = config.FILEPATH_CONFIG_REGISTRY_DB,
        db_structure = DB_STRUCTURE,
        id_col = COL_ID,
    )
    
    if satellite == SATELLITE_S2L2A:
        scl_masks = str_to_list_int(str(args.s2l2a_masks))
        config_id, config_registered, \
        config_already_registered = \
        register_s2l2a_config(
            mosaic_days = mosaic,
            scl_masks = scl_masks,
        )

    if config_already_registered:
        print('Config already registered.')
    print(f'Config registered: {config_registered}')
    print(f'Config ID: {config_id}')
