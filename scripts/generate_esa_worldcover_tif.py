import os

CUR_DIR = os.path.dirname(__file__)

import argparse
import geopandas as gpd
import shapely
import rasterio.merge
import time
import shutil

import config

import sys

import rsutils.utils
import rsutils.esa_download
import rsutils.modify_images


FOLDERPATH_ESA = os.path.join(CUR_DIR, '../data/esa')


if __name__ == '__main__':
    start_time = time.time()

    parser = argparse.ArgumentParser(
        prog = 'python generate_esa_worldcover_tif.py',
        description = (
            'Script to create ESA WorldCover tif for a given shapefile and year.'
        ),
        epilog = f"--- Send your complaints to {','.join(config.MAINTAINERS)} ---",
    )

    parser.add_argument('shapefile',  action='store', help='path/to/shapefile')
    parser.add_argument('year',  action='store', help='path/to/shapefile')
    parser.add_argument('export_filepath',  action='store', help='path/to/worldcereal')
    parser.add_argument('--njobs', '-j',  action='store', required=False, default=1, help='[default = 1] Number of parallel jobs')
    parser.add_argument('-e', '--envelop', metavar='BUFFER', action='store', required=False, default=None, help='[default = None, recommended = 0.01] Create an envelop of the shapefile passed and add buffer in crs EPSG:4326')

    args = parser.parse_args()

    shapes_gdf = gpd.read_file(args.shapefile).to_crs('epsg:4326')
    year = int(args.year)
    export_filepath = str(args.export_filepath)
    njobs = int(args.njobs)

    geometry = shapely.unary_union(shapes_gdf['geometry'])
    if args.envelop is not None:
        buffer = float(args.envelop)
        geometry = geometry.buffer(buffer)

    single_shape_gdf = gpd.GeoDataFrame(
        data = {
            'id': [0],
            'geometry': [geometry],
        }, 
        crs='epsg:4326',
    )

    _, filepaths_df = rsutils.esa_download.download_intersecting_tiles(
        shapes_gdf = single_shape_gdf,
        years = [year],
        working_dir = FOLDERPATH_ESA,
        id_col = 'id',
        njobs = njobs,
    )

    work_id = rsutils.utils.get_epochs_str()

    working_dir = os.path.join(
        os.path.split(export_filepath)[0], 
        f'temp_esa_{work_id}',
    )

    src_filepaths = filepaths_df[rsutils.esa_download.COL_FILEPATH]
    dst_filepaths = [
        rsutils.utils.modify_filepath(
            filepath = fp,
            suffix = f'_{work_id}',
            new_folderpath = working_dir,
        ) for fp in src_filepaths
    ]

    rsutils.modify_images.modify_images(
        src_filepaths = src_filepaths,
        dst_filepaths = dst_filepaths,
        njobs = njobs,
        sequence = [
            (rsutils.modify_images.crop, dict(shapes_gdf = single_shape_gdf, 
                                              nodata = 0, 
                                              all_touched = True))
        ]
    )

    merged_ndarray, merged_transform = rasterio.merge.merge(
        dst_filepaths, nodata = 0,
    )

    with rasterio.open(dst_filepaths[0]) as src:
        merged_meta = src.meta.copy()

    merged_meta.update({
        'count': merged_ndarray.shape[0],
        'height': merged_ndarray.shape[1],
        'width': merged_ndarray.shape[2],
        'transform': merged_transform,
        'compress': 'lzw',
    })

    with rasterio.open(export_filepath, 'w', **merged_meta) as dst:
        dst.write(merged_ndarray)

    shutil.rmtree(working_dir)
    
    end_time = time.time()

    t_elapsed = end_time - start_time

    print(f't_elapsed: {round(t_elapsed, 2)} s')
