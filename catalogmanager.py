import geopandas as gpd
import os
import shapely
import pandas as pd
import datetime

import exceptions


DTYPE_STR = 'str'
DTYPE_TIMESTAMP = 'timestamp'
DTYPE_FLOAT = 'float'
DTYPE_INT = 'int'
DTYPE_MULTIPOLYGON = 'multipolygon'
DTYPE_LIST_STR = 'list[str]'

COL_ID = 'id'
COL_LOCAL_FOLDERPATH = 'local_folderpath'
COL_FILES = 'files'
COL_LAST_UPDATE = 'last_update'

EPSG_4326 = 'epsg:4326'

DTYPE_PLACEHOLDER = {
    DTYPE_STR: 'blah',
    DTYPE_TIMESTAMP: pd.Timestamp('2021-01-28 07:57:53.635000+0000', tz='UTC'),
    DTYPE_INT: 0,
    DTYPE_FLOAT: 0.0,
    DTYPE_MULTIPOLYGON: shapely.MultiPolygon([
        shapely.Polygon([
            (38.09192035575927, 8.141009568663828), 
            (39.08858871535991, 8.142015435187918), 
            (39.088384163693355, 7.148816474425373), 
            (38.09401676842014, 7.147934625992314), 
            (38.09192035575927, 8.141009568663828),
        ])
    ]),
    DTYPE_LIST_STR: 'blah'
}


def dt2ts(
    dt:datetime.datetime, 
    tz='UTC',
):
    return pd.Timestamp(dt, tz=tz)


def merge_multiple_catalogs(
    filepaths:list[str],
    out_filepath:str,
):
    catalog_gdfs = [gpd.read_file(filepath) for filepath in filepaths]
    merged_catalog_gdf = gpd.GeoDataFrame(
        pd.concat(catalog_gdfs).set_index(COL_ID), 
        crs = EPSG_4326,
    )
    merged_catalog_gdf.to_file(out_filepath)


class CatalogManager(object):
    def __init__(self, catalog_filepath:str, cols_dtype_dict:dict):
        if COL_ID not in cols_dtype_dict.keys():
            cols_dtype_dict[COL_ID] = DTYPE_STR
        elif cols_dtype_dict[COL_ID] != DTYPE_STR:
            raise exceptions.CatalogManagerException(
                f'dtype of {COL_ID} column must be {DTYPE_STR}'
            )
        
        if COL_LAST_UPDATE not in cols_dtype_dict.keys():
            cols_dtype_dict[COL_LAST_UPDATE] = DTYPE_TIMESTAMP
        elif cols_dtype_dict[COL_LAST_UPDATE] != DTYPE_TIMESTAMP:
            raise exceptions.CatalogManagerException(
                f'dtype of {COL_LAST_UPDATE} column must be {DTYPE_TIMESTAMP}'
            )
        
        self.catalog_filepath = catalog_filepath
        self.cols_dtype_dict = cols_dtype_dict
    
        folderpath = os.path.split(catalog_filepath)[0]
        os.makedirs(folderpath, exist_ok=True)
        if not os.access(folderpath, os.W_OK):
            raise exceptions.CatalogManagerException(
                f'User does not have the permission to write catalog '
                f'in folderpath={folderpath}. Set a new catalog_filepath.'
            )

        if os.path.exists(self.catalog_filepath):
            self.catalog_gdf:gpd.GeoDataFrame = gpd.read_file(self.catalog_filepath)
            self.catalog_gdf = self.catalog_gdf.set_index(COL_ID)
        else:
            # it was difficult to figure out how to set the dtypes for each columns correctly
            # the work around implemented is to initialise with placeholders and let geopandas
            # figure out the dtype by itself.
            self.catalog_gdf = gpd.GeoDataFrame(
                data = {
                    col: [DTYPE_PLACEHOLDER[dtype]]
                    for col, dtype in self.cols_dtype_dict.items()
                },
                crs = EPSG_4326
            ).set_index(COL_ID).drop(index=DTYPE_PLACEHOLDER[DTYPE_STR])

    
    @staticmethod
    def get_current_timestamp():
        return pd.Timestamp(datetime.datetime.now(), tz='UTC')
    

    def add(
        self,
        entry:dict,
        force_update:bool = False,
    ):
        performed_update = False

        entry_modified_dict = {}

        _id = entry[COL_ID]
        del entry[COL_ID]

        first_entry = _id not in self.catalog_gdf.index

        for col, value in entry.items():
            col_dtype = self.cols_dtype_dict[col]
            entry_modified_dict[col] = False
            if value is None:
                if first_entry:
                    raise exceptions.CatalogManagerException(
                        f'First entry for col={col} can not be None for id={_id}'
                    )
                else:
                    continue
            else:
                """
                The DTYPE_LIST_STR is handelled in a specific manner. It was designed to
                store the list of files that are stores in a folder - for example the different
                band files for a satellite tile, and for datacube generation the datacube.npy
                and the metadata.pickle.npy.

                For this type, for a given id, if there is an entry present - say: ['A', 'B', 'C']
                if add function is used for the same id but the new entry being: ['C', 'D', 'E'],
                the new items would be added making the updated entry: ['A', 'B', 'C', 'D', 'E'].

                Deleting items from this type is non trivial thus a separate function is written
                below - modify_list_str_col.

                In all other types, the value is directly replaced.
                """
                if col_dtype == DTYPE_LIST_STR:
                    updated_list = set(value)
                    if not first_entry:
                        current_list = self.catalog_gdf.loc[_id, col].split(',')
                        if current_list != value:
                            entry_modified_dict[col] = True
                        updated_list = set(current_list) | set(value)
                    else:
                        entry_modified_dict[col] = True
                    self.catalog_gdf.loc[_id, col] = ','.join(updated_list)
                else:
                    if not first_entry:
                        if self.catalog_gdf.loc[_id, col] != value:
                            entry_modified_dict[col] = True
                    else:
                        entry_modified_dict[col] = True
                    self.catalog_gdf.loc[_id, col] = value
                        
        
        performed_update = any(entry_modified_dict.values())

        if performed_update or force_update:
            self.catalog_gdf.loc[_id, COL_LAST_UPDATE] = CatalogManager.get_current_timestamp()
        
        return performed_update
    

    def modify_list_str_col(self, _id:str, col:str, delete_items:list[str]=None, add_items:list[str]=None):
        col_dtype = self.cols_dtype_dict[col]
        if col_dtype!= DTYPE_LIST_STR:
            raise exceptions.CatalogManagerException(
                f'col={col} is of dtype {col_dtype}. dtype must be {DTYPE_LIST_STR} to use modify_list_str_col function.'
            )
        if pd.isna(self.catalog_gdf.loc[_id, col]):
            raise exceptions.CatalogManagerException(
                f'id={_id} not present in the catalog. Can not perform modify_list_str_col.'
            )
    
        if delete_items is None:
            delete_items = set()
        else:
            delete_items = set(delete_items)

        if add_items is None:
            add_items = set()
        else:
            add_items = set(add_items)
        
        current_items = set(self.catalog_gdf.loc[_id, col].split(','))
        
        conflicting_items = delete_items & add_items
        absent_items_to_delete = delete_items - current_items
        present_items_to_add = current_items & add_items

        if len(conflicting_items) > 0:
            raise exceptions.CatalogManagerException(
                f'Conflicting requests. Following items being added and deleted: {list(conflicting_items)}'
            )

        if len(absent_items_to_delete) > 0:
            raise exceptions.CatalogManagerException(
                f'Non-existent items requested to be deleted: {list(absent_items_to_delete)}'
            )
        
        if len(present_items_to_add) > 0:
            raise exceptions.CatalogManagerException(
                f'Pre-existent items requested to be added: {list(present_items_to_add)}'
            )

        updated_items = (current_items | add_items) - delete_items

        if updated_items != current_items:
            self.catalog_gdf.loc[_id, COL_LAST_UPDATE] = CatalogManager.get_current_timestamp()

        self.catalog_gdf.loc[_id, col] = ','.join(updated_items)
        
        return len(updated_items)

    
    def delete_entry(self, _id:str):
        if _id not in self.catalog_gdf.index:
            raise exceptions.CatalogManagerException(
                f'id={_id} not present in the catalog. Can not perform delete_entry.'
            )
        self.catalog_gdf = self.catalog_gdf.drop(index=_id)
    

    def save(self):
        self.catalog_gdf.to_file(self.catalog_filepath)