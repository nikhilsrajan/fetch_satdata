import os
import sqlite3
import geopandas as gpd
import pandas as pd
import shapely
import logging
import datetime


TIMESTAMP_COLS = ['timestamp', 'last_update', 'startdate', 'enddate']
GEOMETRY_COLS = ['geometry']


def create_db(
    database:str,
    db_structure:dict[str, dict[str, str]], # {table : {col : type}}
    id_col:str,
):
    connection = sqlite3.connect(database)
    cursor = connection.cursor()

    try:
        for table_name, col_type_dict in db_structure.items():
            script = f"CREATE TABLE '{table_name}' ("
            if id_col not in col_type_dict.keys():
                raise ValueError(f'id_col={id_col} not present in col_type_dict')

            for col_name, type_name in col_type_dict.items():
                script += f"'{col_name}' {type_name},"
            
            script += f"PRIMARY KEY('{id_col}') );"

            cursor.execute(script)

        cursor.close()

    except Exception as e:
        cursor.close()
        raise e
    

def add_columns_to_table(
    database:str,
    table:str,
    col_type_dict:dict
):
    connection = sqlite3.connect(database)
    cursor = connection.cursor()

    try:
        for col_name, col_type in col_type_dict.items():
            cursor.execute(f"ALTER TABLE '{table}' ADD COLUMN '{col_name}' {col_type}")
        cursor.close()
    except Exception as e:
        cursor.close()
        raise e
        

def init_db(
    database:str,
    db_structure:dict[str, dict[str, str]], # {table : {col : type}}
    id_col:str,  
):
    """
    If DB not present, it is created.
    If table is not present, it is added.
    If column not present, it is added.
    If column type does not match db_structure - error is thrown.
    """

    # new DB altogether so its straight-forward creation
    if not os.path.exists(database):
        create_db(database=database, db_structure=db_structure, id_col=id_col)
        return

    # DB already exists - so now we check if it needs to be altered.
    # 1. check is all the tables exist - add if not
    tables_in_db = get_tables_in_db(database=database, use_WAL=True)
    completely_new_tables = [
        table for table in db_structure.keys() 
        if table not in tables_in_db
    ]
    new_tables_db_structure = {
        table : col_type_dict 
        for table, col_type_dict in db_structure.items() 
        if table in completely_new_tables
    }
    create_db(database=database, db_structure=new_tables_db_structure, id_col=id_col)

    # 2. check if all the columns exist - add if not
    preexisting_tables_db_struct = {
        table: get_table_col_type_dict(database=database, table=table)
        for table in db_structure.keys()
        if table not in completely_new_tables
    }
    mismatching_tables = [
        table for table, col_type_dict in preexisting_tables_db_struct.items()
        if col_type_dict != db_structure[table]
    ]

    for table in mismatching_tables:
        existing_col_type_dict = preexisting_tables_db_struct[table]
        desired_col_type_dict = db_structure[table]
        absent_cols = set(desired_col_type_dict.keys()) - set(existing_col_type_dict.keys())
        # check if all the column types match - raise error if not
        for col, desired_col_type in desired_col_type_dict.items():
            if col in absent_cols:
                continue
            existing_col_type = existing_col_type_dict[col]
            if desired_col_type != existing_col_type:
                raise ValueError(
                    f"In table='{table}'', col='{col}' already exists with type={existing_col_type}. "
                    f"It is non-trivial to change the type to {desired_col_type}. "
                    "Reconsider."
                )
        add_columns_to_table(
            database = database,
            table = table,
            col_type_dict = {
                col: desired_col_type_dict[col] for col in absent_cols
            }
        )


def ts_to_str(ts:pd.Timestamp)->str:
    return ts.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + 'Z'


def str_to_ts(ts_str:str)->pd.Timestamp:
    return pd.to_datetime(ts_str)


def gpd_to_sql_row(row_dict:dict)->dict:
    reformatted_dict = {}
    for k, v in row_dict.items():
        if isinstance(v, datetime.datetime):
            v = ts_to_str(v)
        elif isinstance(v, shapely.Geometry):
            v = shapely.to_wkt(v)
        reformatted_dict[k] = v
    return reformatted_dict


def sql_to_gpd_row(
    row_dict:dict, 
    timestamp_cols:list[str] = TIMESTAMP_COLS, 
    geometry_cols:list[str] = GEOMETRY_COLS,
)->dict:
    reformatted_dict = {}
    for k, v in row_dict.items():
        if k in timestamp_cols:
            v = str_to_ts(v)
        elif k in geometry_cols:
            v = shapely.from_wkt(v)
        reformatted_dict[k] = v
    return reformatted_dict


def insert_rows_to_db(
    database:str, 
    table:str, 
    data_dicts:list[dict], 
    use_WAL:bool = True,
):
    cols = []
    datas = []

    cols = set()

    for data_dict in data_dicts:
        cols |= set(data_dict.keys())

    cols = list(cols)

    for data_dict in data_dicts:
        data = []
        for col in cols:
            if col not in data_dict.keys():
                val = None
            else:
                val = data_dict[col]
            data.append(val)
        datas.append(tuple(data))
        del data

    connection = sqlite3.connect(database)
    cursor = connection.cursor()
    if use_WAL:
        connection.execute('pragma journal_mode=wal')

    try:
        cursor.executemany(f"INSERT INTO '{table}' ({', '.join(cols)}) VALUES ({', '.join(['?' for i in range(len(cols))])})", datas)
        connection.commit()
        connection.close()
    except Exception as e:
        connection.close()
        raise e


def fetch_value_in_db(
    database:str,
    table:str,
    id:str,
    id_col:str,
    col:str,
    use_WAL:bool = True,
    timestamp_cols:list[str] = TIMESTAMP_COLS, 
    geometry_cols:list[str] = GEOMETRY_COLS,
):
    return fetch_rows_from_db(
        database = database,
        table = table,
        use_WAL = use_WAL,
        columns = [col],
        ids = [id],
        id_col = id_col,
        timestamp_cols = timestamp_cols,
        geometry_cols = geometry_cols,
    )[col][0]


def get_table_col_type_dict(
    database:str,
    table:str,
    use_WAL:bool = True,
):
    table_info_df = fetch_rows_from_db(
        database = database,
        table = table,
        query = f"PRAGMA table_info('{table}')",
        use_WAL = use_WAL,
    )

    col_type_dict = {}
    for index, row in table_info_df.iterrows():
        col_type_dict[row['name']] = row['type']
        if row['pk'] == 1:
            col_type_dict[row['name']] += ' UNIQUE'

    return col_type_dict


def get_tables_in_db(
    database:str, 
    use_WAL:bool = True,
):
    sqlite_master_df = fetch_rows_from_db(
        database = database,
        table = None,
        query = "SELECT * FROM sqlite_master",
        use_WAL = use_WAL,
    )
    return sqlite_master_df[
        sqlite_master_df['type']=='table'
    ]['tbl_name'].to_list()


def update_value_in_db(
    database:str,
    table:str,
    id:str,
    id_col:str,
    col:str,
    update_value,
    use_WAL:bool = True,
    logger:logging.Logger = None,
):
    updation_success = True

    connection = sqlite3.connect(database)
    cursor = connection.cursor()
    if use_WAL:
        connection.execute('pragma journal_mode=wal')
    
    try:
        cursor.execute(f"UPDATE {table} SET {col} = ? WHERE {id_col} = ?", (update_value, id))
    except Exception as e:
        updation_success = False
        msg = f'Updation to DB failed -- {e}'
        if logger is not None:
            logger.error(msg)
        else:
            print(msg)

    connection.commit()
    connection.close()

    return updation_success


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
    timestamp_cols:list[str] = TIMESTAMP_COLS, 
    geometry_cols:list[str] = GEOMETRY_COLS,
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
    
    data = {col: [] for col in columns}

    for row in results:
        row_dict = dict(zip(columns, row))
        parsed_row_dict = sql_to_gpd_row(
            row_dict = row_dict,
            timestamp_cols = timestamp_cols,
            geometry_cols = geometry_cols,
        )
        for k, v in parsed_row_dict.items():
            data[k].append(v)
    
    if 'geometry' in columns:
        fetched_gdf = gpd.GeoDataFrame(data=data, crs='epsg:4326')
    else:
        fetched_gdf = gpd.GeoDataFrame(data=data)
    
    return fetched_gdf


def delete_rows_from_db():
    raise NotImplementedError()
    try:
        sqliteConnection = sqlite3.connect('SQLite_Python.db')
        cursor = sqliteConnection.cursor()
        print("Connected to SQLite")

        # Deleting single record now
        sql_delete_query = """DELETE from SqliteDb_developers where id = 6"""
        cursor.execute(sql_delete_query)
        sqliteConnection.commit()
        print("Record deleted successfully ")
        cursor.close()

    except sqlite3.Error as error:
        print("Failed to delete record from sqlite table", error)
    finally:
        if sqliteConnection:
            sqliteConnection.close()
            print("the sqlite connection is closed")
