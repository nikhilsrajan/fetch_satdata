import os

CUR_DIR = os.path.dirname(__file__)

import argparse
import time
import geopandas as gpd
import pandas as pd
import shapely
import rasterio
import rasterio.io
import rasterio.merge
import tqdm

import sys

sys.path.append(os.path.join(CUR_DIR, '..'))

import rsutils.utils


def crop_and_merge(
    filepaths:list[str],
    shape_gdf:gpd.GeoDataFrame,
):
    memfiles = []
    print('cropping files and creating memfiles')
    for filepath in tqdm.tqdm(filepaths):
        cropped_imarray, cropped_meta = \
        rsutils.utils.crop_tif(
            src_filepath = filepath,
            shapes_gdf = shape_gdf,
            all_touched = True,
        )

        cropped_imarray[cropped_imarray != 100] = 0
        cropped_imarray[cropped_imarray == 100] = 1
        cropped_meta['nodata'] = 0
        cropped_meta['dtype'] = 'uint8'

        memfile = rasterio.io.MemoryFile()
        with memfile.open(**cropped_meta) as dataset:
            dataset.write(cropped_imarray)
        memfiles.append(memfile)

    print('merging cropped memfiles')
    merged_imarray, merged_transform = rasterio.merge.merge(
        [memfile.open() for memfile in memfiles], nodata = 0,
    )

    _, height, width = merged_imarray.shape
    cropped_meta['height'] = height
    cropped_meta['width'] = width
    cropped_meta['transform'] = merged_transform
    cropped_meta['compress'] = 'lzw'

    return merged_imarray, cropped_meta


def create_worldcereal_catalog(
    folderpath:str,
):
    filepaths = rsutils.utils.get_all_files_in_folder(folderpath=folderpath, keep_extensions=['.tif'])
    
    catalog_data = {
        'filepath': [],
        'aez': [],
        'product': [], # season_product
    }

    for filepath in filepaths:
        filename = os.path.split(filepath)[1]
        aez, season, product, _, _, _ = filename.removesuffix('.tif').split('_')
        product = season + '_' + product
        catalog_data['filepath'].append(filepath)
        catalog_data['aez'].append(int(aez))
        catalog_data['product'].append(product)
    
    catalog_df = pd.DataFrame(data=catalog_data)

    return catalog_df


if __name__ == '__main__':
    start_time = time.time()

    parser = argparse.ArgumentParser(
        prog = 'python generate_esa_worldcover_tif.py',
        description = (
            'Script to create ESA WorldCover tif for a given shapefile and year.'
        ),
        epilog = f"--- Send your complaints to nsasiraj@umd.edu ---",
    )

    parser.add_argument('shapefile',  action='store', help='path/to/shapefile')
    parser.add_argument('worldcereal_tifs_folderpath',  action='store', help='path/to/worldcereal_tifs')
    parser.add_argument('worldcereal_aez_shapefile',  action='store', help='path/to/WorldCereal_AEZ.geojson')
    parser.add_argument('products_to_merge',  action='store', help="comma separate list of product names. Example: \"tc-maize-main_maize_classification,tc-maize-second_maize_classification\"")
    parser.add_argument('export_filepath',  action='store', help='Filepath where the merged tif needs to be saved.')
    # parser.add_argument('--njobs', '-j',  action='store', required=False, default=1, help='[default = 1] Number of parallel jobs')
    parser.add_argument('-e', '--envelop', metavar='BUFFER', action='store', required=False, default=None, help='[default = None, recommended = 0.01] Create an envelop of the shapefile passed and add buffer in crs EPSG:4326')


    args = parser.parse_args()

    shapes_gdf = gpd.read_file(args.shapefile).to_crs('epsg:4326')
    worldcereal_tifs_folderpath = str(args.worldcereal_tifs_folderpath)
    worldcereal_aez_gdf = gpd.read_file(args.worldcereal_aez_shapefile)
    products_to_merge = str(args.products_to_merge).split(',')
    export_filepath = str(args.export_filepath)

    catalog_df = create_worldcereal_catalog(folderpath=worldcereal_tifs_folderpath)

    geometry = shapely.unary_union(shapes_gdf['geometry'])
    if args.envelop is not None:
        buffer = float(args.envelop)
        print(f'creating evelope with buffer = {buffer}')
        geometry = geometry.buffer(buffer)
    
    shape_gdf = gpd.GeoDataFrame(data = {'geometry': [geometry]}, crs='epsg:4326')

    intersecting_aezs = worldcereal_aez_gdf[
        worldcereal_aez_gdf.intersects(
            shapely.unary_union(
                shape_gdf.to_crs(worldcereal_aez_gdf.crs)['geometry']
            )
        )
    ]['aez_id'].tolist()

    print('intersecting aezs:', intersecting_aezs)

    required_catalog_df = catalog_df[
        catalog_df['aez'].isin(intersecting_aezs) &
        catalog_df['product'].isin(products_to_merge)
    ]
    
    print(required_catalog_df.shape)

    merged_imarray, merged_meta = crop_and_merge(
        filepaths = required_catalog_df['filepath'],
        shape_gdf = shape_gdf,
    )

    os.makedirs(os.path.dirname(export_filepath), exist_ok=True)

    print('writing merged file')
    with rasterio.open(export_filepath, 'w', **merged_meta) as dst:
        dst.write(merged_imarray)

    end_time = time.time()

    t_elapsed = end_time - start_time

    print(f't_elapsed: {round(t_elapsed, 2)} s')
