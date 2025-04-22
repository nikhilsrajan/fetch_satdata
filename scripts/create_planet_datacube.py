import os

CURRENT_DIRECTORY = os.path.dirname(__file__)

import argparse
import geopandas as gpd
import pandas as pd
import numpy as np
import gc

import config

import sys
sys.path.append(os.path.join(CURRENT_DIRECTORY, '..'))

import rsutils.utils


def create_a_blank_file(filepath):
    with open(filepath, 'w') as h:
        h.write(pd.Timestamp.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + 'Z')


def get_filtered_catalog(
    catalog_gdf:gpd.GeoDataFrame,
    timestamp_col:str,
    startdate:pd.Timestamp,
    enddate:pd.Timestamp,
    shape_gdf:gpd.GeoDataFrame,
):
    daterange_catalog_gdf = catalog_gdf[
        (catalog_gdf[timestamp_col] >= startdate) &
        (catalog_gdf[timestamp_col] <= enddate)
    ]
    filtered_catalog_gdf = daterange_catalog_gdf[
        daterange_catalog_gdf.intersects(
            shape_gdf.to_crs(catalog_gdf.crs)['geometry'][0]
        )
    ]
    return filtered_catalog_gdf


def create_datacube(
    filtered_catalog_gdf:gpd.GeoDataFrame,
    timestamp_col:str,
    filepath_col:str,
    shape_gdf:gpd.GeoDataFrame,
):
    filtered_catalog_gdf = filtered_catalog_gdf.sort_values(by=timestamp_col, ascending=True)
    
    bands_stack = []
    for index, row in filtered_catalog_gdf.iterrows():
        filepath = row[filepath_col]
        cropped_imarray, _ = rsutils.utils.crop_tif(
            src_filepath = filepath,
            shapes_gdf = shape_gdf,
            nodata = 0,
            all_touched = True,
        )
        bands_stack.append(cropped_imarray.copy())
        del cropped_imarray
    
    bands_stack = np.stack(bands_stack, axis=-1)
    return bands_stack


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog = 'python create_planetdatacube.py',
        description = (
            "Script to create planet datacube. This script would be run with snakemake for "
            "parallelisation. Thus this script is to create a single plant datacube. \n"
            "This script makes the following assumptions: \n"
            "   1. Each polygon fell exactly into one planetdata tile. \n"
            "   2. The band order and the crs across all the images are the same. \n"
            "These assumption was acertained when the input csv was created "
            "using setup_planetdatacube_run.py where we found that for Malawi, every polygon "
            "fell into exactly one tile. This assumption makes us uncomplicate the script and "
            "avoid many operations like resampling and merging. In the situation this assumption "
            "becomes invalid, changes would be made. Until then YAGNI FTW."
        ),
        epilog = f"--- Send your complaints to {','.join(config.MAINTAINERS)} ---",
    )

    parser.add_argument('shapefilepath', help="shapefile containing polygon of interest.")
    parser.add_argument('catalog_filepath', help="catalog filepath contains path to the satellite files relevant to createing the datacube.")
    parser.add_argument('timestamp_col', help="Timestamp column in the catalog")
    parser.add_argument('filepath_col', help="Filepath column in the catalog")
    parser.add_argument('startdate', help='style: 2024-08-20 or 2024-08-20T13:54:40.022Z')
    parser.add_argument('enddate', help='style: 2024-08-20 or 2024-08-20T13:54:40.022Z')
    parser.add_argument('export_folderpath', help='Folderpath where the datacube should be saved.')
    
    args = parser.parse_args()

    catalog_gdf = gpd.read_file(args.catalog_filepath)
    shape_gdf = gpd.read_file(args.shapefilepath)
    timestamp_col = str(args.timestamp_col)
    filepath_col = str(args.filepath_col)
    startdate = pd.to_datetime(str(args.startdate))
    enddate = pd.to_datetime(str(args.enddate))
    export_folderpath = str(args.export_folderpath)

    filtered_catalog_gdf = get_filtered_catalog(
        catalog_gdf = catalog_gdf,
        timestamp_col = timestamp_col,
        startdate = startdate,
        enddate = enddate,
        shape_gdf = shape_gdf,
    )

    if filtered_catalog_gdf.shape[0] != catalog_gdf.shape[0]:
        raise ValueError(
            'catalog after filtering is smaller than original catalog. '
            'This is unexpected as subset catalog is what is created for '
            'planet datacube creation. '
        )

    bands_stack = create_datacube(
        filtered_catalog_gdf = filtered_catalog_gdf,
        timestamp_col = timestamp_col,
        filepath_col = filepath_col,
        shape_gdf = shape_gdf,
    )

    datacube_filepath = os.path.join(export_folderpath, 'datacube.npy')
    np.save(datacube_filepath, bands_stack)
    
    del bands_stack
    gc.collect()

    done_text_filepath = os.path.join(export_folderpath, 'done.npy')
    create_a_blank_file(done_text_filepath)
