import os
import sqlite3
import geopandas as gpd
import pandas as pd
import shapely
import logging


def create_db(
    db_path:str,
    table_name:str,
    col_type_dict:dict,
    id_col:str,
    overwrite:bool = False,
):
    if os.path.exists(db_path):
        if overwrite:
            os.remove(db_path)
        else:
            raise FileExistsError(f"File already exists in the specified path: {db_path}")
    
    script = f"CREATE TABLE '{table_name}' ("
    if id_col not in col_type_dict.keys():
        raise ValueError(f'id_col={id_col} not present in col_type_dict')

    for col_name, type_name in col_type_dict.items():
        script += f"'{col_name}' {type_name},"
    
    script += f"PRIMARY KEY('{id_col}') );"

    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    cursor.execute(script)
    cursor.close()


def ts_to_str(ts:pd.Timestamp)->str:
    return ts.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + 'Z'


def str_to_ts(ts_str:str)->pd.Timestamp:
    return pd.to_datetime(ts_str)


def gpd_to_sql_row(row_dict:dict)->dict:
    reformatted_dict = {}
    for k, v in row_dict.items():
        if k in ['timestamp', 'last_update']:
            v = ts_to_str(v)
        elif k == 'geometry':
            v = shapely.to_wkt(v)
        reformatted_dict[k] = v
    return reformatted_dict


def sql_to_gpd_row(row_dict:dict)->dict:
    reformatted_dict = {}
    for k, v in row_dict.items():
        if k in ['timestamp', 'last_update']:
            v = str_to_ts(v)
        elif k == 'geometry':
            v = shapely.from_wkt(v)
        reformatted_dict[k] = v
    return reformatted_dict


def insert_row_to_db(
    database:str, 
    table:str, 
    data_dict:dict, 
    use_WAL:bool = True, 
    logger:logging.Logger = None,
):
    connection = sqlite3.connect(database)
    cursor = connection.cursor()
    if use_WAL:
        connection.execute('pragma journal_mode=wal')
        
    cols = []
    data = []
    for k, v in data_dict.items():
        cols.append(k)
        data.append(v)

    insertion_success = True

    try:
        cursor.execute(f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join(['?' for i in range(len(data))])})", data)
    except sqlite3.IntegrityError as e:
        insertion_success = False
        msg = f'Insertion to DB failed -- {e}'
        if logger is not None:
            logger.error(msg)
        else:
            print(msg)

    connection.commit()
    connection.close()

    return insertion_success


def generate_query(
    table:str,
    columns:list[str] = None,
    ids:list[str] = None, 
    id_col:str = None,
):
    query = "SELECT "

    if columns is None:
        query += "* "
    else:
        query += ', '.join(columns) + ' '
    
    query += f"FROM {table} "

    if ids is not None and id_col is not None:
        query += f"WHERE {id_col} IN "
        query += '(' + ', '.join([f"'{_id}'" for _id in ids]) + ')'
    
    return query


def fetch_rows_from_db(
    database:str, 
    table:str,
    use_WAL:bool = True, 
    columns:list[str] = None, 
    ids:list[str] = None, 
    id_col:str = None,
    query:str = None,
):
    if query is None:
        query = generate_query(
            table = table,
            columns = columns,
            ids = ids,
            id_col = id_col,
        )

    connection = sqlite3.connect(database)
    cursor = connection.cursor()
    if use_WAL:
        connection.execute('pragma journal_mode=wal')

    cursor.execute(query)
    results = cursor.fetchall()

    if columns is None:
        columns = [description[0] for description in cursor.description]

    connection.close()
    
    data = {}

    for row in results:
        row_dict = dict(zip(columns, row))
        parsed_row_dict = sql_to_gpd_row(row_dict)
        for k, v in parsed_row_dict.items():
            if k not in data.keys():
                data[k] = []
            data[k].append(v)
    
    if 'geometry' in columns:
        fetched_gdf = gpd.GeoDataFrame(data=data, crs='epsg:4326')
    else:
        fetched_gdf = gpd.GeoDataFrame(data=data)
    
    return fetched_gdf
