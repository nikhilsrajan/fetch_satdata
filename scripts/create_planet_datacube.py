import os

CURRENT_DIRECTORY = os.path.dirname(__file__)

import argparse
import geopandas as gpd
import pandas as pd
import numpy as np
import gc
import rasterio.io
import rasterio.merge

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


def get_descriptions(filepath):
    with rasterio.open(filepath) as src:
        desc = src.descriptions
    return desc


def create_datacube(
    filtered_catalog_gdf:gpd.GeoDataFrame,
    timestamp_col:str,
    filepath_col:str,
    shape_gdf:gpd.GeoDataFrame,
):
    timestamp_filepaths_list = filtered_catalog_gdf.groupby(
        timestamp_col
    )[filepath_col].apply(list).to_dict()
    
    timestamps = filtered_catalog_gdf[timestamp_col].unique().tolist()
    timestamps.sort()
    
    bands_stack = []

    prev_desc = None
    meta = None

    for ts in timestamps:
        filepaths = timestamp_filepaths_list[ts]
        if len(filepaths) == 1:
            cur_desc = get_descriptions(filepath=filepaths[0])
            if prev_desc is None:
                prev_desc = cur_desc
            elif prev_desc != cur_desc:
                raise ValueError(f'Description mismatch. prev_desc={prev_desc}, cur_desc={cur_desc}')

            cropped_imarray, cropped_meta = rsutils.utils.crop_tif(
                src_filepath = filepaths[0],
                shapes_gdf = shape_gdf,
                nodata = 0,
                all_touched = True,
            )
            bands_stack.append(cropped_imarray.copy())
            del cropped_imarray

            if meta is None:
                meta = cropped_meta
        else:
            memfiles = []
            for filepath in filepaths:
                cur_desc = get_descriptions(filepath=filepath)
                if prev_desc is None:
                    prev_desc = cur_desc
                elif prev_desc != cur_desc:
                    raise ValueError(f'Description mismatch. prev_desc={prev_desc}, cur_desc={cur_desc}')
                
                cropped_imarray, cropped_meta = rsutils.utils.crop_tif(
                    src_filepath = filepath,
                    shapes_gdf = shape_gdf,
                    nodata = 0,
                    all_touched = True,
                )
                memfile = rasterio.io.MemoryFile()
                with memfile.open(**cropped_meta) as dataset:
                    dataset.write(cropped_imarray)
                memfiles.append(memfile)

            merged_ndarray, merged_transform = rasterio.merge.merge(
                [memfile.open() for memfile in memfiles], nodata = 0,
            )

            if meta is None:
                meta = cropped_meta
                _, height, width = merged_ndarray.shape
                meta['height'] = height
                meta['width'] = width
                meta['transform'] = merged_transform

            del memfiles
            bands_stack.append(merged_ndarray.copy())
            del merged_ndarray

    bands_stack = np.stack(bands_stack, axis=-1)
    bands_stack = bands_stack.swapaxes(0, -1) # axis: timestamps, height, width, bands
    return bands_stack, timestamps, meta, prev_desc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog = 'python create_planetdatacube.py',
        description = (
            "Script to create planet datacube. This script would be run with snakemake for "
            "parallelisation. Thus this script is to create a single plant datacube. \n"
            "This script makes the following assumptions: \n"
            "   1. Each polygon fell exactly into one planetdata tile. (this assumption didn't hold true).\n"
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
    startdate = pd.Timestamp(str(args.startdate))
    enddate = pd.Timestamp(str(args.enddate))
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

    bands_stack, timestamps, meta, prev_desc = create_datacube(
        filtered_catalog_gdf = filtered_catalog_gdf,
        timestamp_col = timestamp_col,
        filepath_col = filepath_col,
        shape_gdf = shape_gdf,
    )

    metadata = {
        'geotiff_metadata': meta,
        'timestamps': timestamps,
        'bands': prev_desc,
        'data_shape_desc': ('timestamps', 'height', 'width', 'bands'),
        'geometry': {
            'shape': shape_gdf['geometry'][0],
            'crs': shape_gdf.crs,
        }
    }

    datacube_filepath = os.path.join(export_folderpath, 'datacube.npy')
    metadata_filepath = os.path.join(export_folderpath, 'metadata.pickle.npy')
    np.save(datacube_filepath, bands_stack)
    np.save(metadata_filepath, metadata, allow_pickle=True)
    
    del bands_stack
    gc.collect()

    done_text_filepath = os.path.join(export_folderpath, 'done.txt')
    create_a_blank_file(done_text_filepath)
