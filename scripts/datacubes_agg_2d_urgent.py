"""
TO DO:
-----

- check if metadatas match

"""


import argparse
import geopandas as gpd
import numpy as np
import os
import tqdm

import sys
sys.path.append('..')

import config
import datacube_ops
import sqlite_db_utils


def read_datacube_convert_to_2d(
    datacube_folderpath:str,
):
    """
    Basically do area median
    """
    datacube_filepath = os.path.join(datacube_folderpath, 'datacube.npy')
    metadata_filepath = os.path.join(datacube_folderpath, 'metadata.pickle.npy')
    datacube = np.load(datacube_filepath)
    metadata = np.load(metadata_filepath, allow_pickle=True)[()]
    new_datacube, new_metadata = datacube_ops.area_median(datacube=datacube, metadata=metadata)
    del datacube, metadata
    return new_datacube


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog = 'python datacubes_agg_2d_urgent.py',
        description = (
            'Script to aggregate datacubes into a single numpy array. '
            'The purpose of this is to convert datacubes of farm level polygons '
            'into data that could be use for training a model. The user passes '
            'a file with the datacube ids they want to aggregate, a list of '
            'operations to be run in sequence, and the export folderpath where a '
            'data.npy and a metadata.pickle.npy would be saved.'
        ),
        epilog = f"--- Send your complaints to {','.join(config.MAINTAINERS)} ---",
    )
    parser.add_argument('datacube_ids_filepath', action='store', help='Filepath of the file containing datacube ids.')
    parser.add_argument('export_folderpath', action='store', help='Folderpath to save the outputs to.')
    parser.add_argument('-j', '--njobs', default=1, action='store', required=False, help='[default = 1] Number of parallel processes to run.')

    args = parser.parse_args()

    datacube_ids_filepath = str(args.datacube_ids_filepath)
    export_folderpath = str(args.export_folderpath)
    njobs = int(args.njobs)

    with open(datacube_ids_filepath) as f:
        datacube_ids = f.readlines()
    datacube_ids = [x[:-1] for x in datacube_ids]
    
    selected_datacube_catalog_gdf = \
    sqlite_db_utils.fetch_rows_from_db(
        database = config.FILEPATH_DATACUBE_CATALOG_DB,
        table = config.S2L2A_TABLE,
        ids = datacube_ids,
        id_col = 'id',
    )

    if selected_datacube_catalog_gdf.shape[0] == 0:
        raise ValueError(f'None of the datacube ids present in the catalog.')
    
    missing_datacube_ids = len(datacube_ids) - selected_datacube_catalog_gdf.shape[0]

    if missing_datacube_ids > 0:
        raise ValueError(f'{missing_datacube_ids} datacube ids missing in the catalog.')
    
    selected_datacube_folderpaths = selected_datacube_catalog_gdf['local_folderpath']
    selected_datacube_ids = selected_datacube_catalog_gdf['id']

    datacubes_2d_list = []

    last_shape = None

    for folderpath in tqdm.tqdm(selected_datacube_folderpaths):
        datacube_2d = read_datacube_convert_to_2d(datacube_folderpath=folderpath)
        if last_shape is None:
            last_shape = datacube_2d.shape
        elif last_shape != datacube_2d.shape:
            raise ValueError(f'shape mismatch -- last_shape={last_shape}, cur_shape={datacube_2d.shape}')
        
        datacubes_2d_list.append(datacube_2d)
        del datacube_2d
    
    datacubes_2d = np.stack(datacubes_2d_list, axis=0)
    datacube_ids = np.array(selected_datacube_ids)

    os.makedirs(export_folderpath, exist_ok=True)

    np.save(os.path.join(export_folderpath, 'data.npy'), datacubes_2d)
    np.save(os.path.join(export_folderpath, 'ids.npy'), datacube_ids)
