from datetime import datetime
import os
import time
import argparse
import shapely
import shutil
import geopandas as gpd
import pandas as pd

import planetary_computer

from stac_downloader.raster_processing import ResamplingMethod
from stac_downloader.stac_downloader import STACDownloader
from stac_downloader.utils import get_logger, prepare_geometry


logger = get_logger()

# Define MPC-Specific Params
STACK_CATALOG_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
STAC_COLLECTION_NAME = "sentinel-2-l2a"


def get_filepaths(
    item,
    assets,
    output_folder,
):
    asset_filepath_dict = {}
    for asset in assets:
        filepath = STACDownloader._get_file_output_path(
            item = item,
            asset_name = asset,
            resolution = None,
            output_folder = output_folder,
            extension = 'tif',
        )
        asset_filepath_dict[asset] = filepath
    return asset_filepath_dict


def get_catalog_gdf(
    items,
    assets,
    output_folder,
    extension = 'tif',
):
    data = {
        'id': [],
        'timestamp': [],
        'local_folderpath': [],
        'files': [],
        'geometry': [],
    }
    for item in items:
        data['id'].append(item.id)
        data['timestamp'].append(item.get_datetime())
        data['local_folderpath'].append(os.path.join(output_folder, item.id))
        data['files'].append(','.join([f'{asset}.{extension}' for asset in assets]))
        data['geometry'].append(shapely.geometry.shape(item.geometry))

    catalog_gdf = gpd.GeoDataFrame(data=data, crs='epsg:4326')
    return catalog_gdf


def get_new_filepaths(
    item_id,
    assets,
    output_folder,
):  
    filepaths = dict()
    extension = 'tif'
    for asset in assets:
        filepaths[asset] = os.path.join(output_folder, item_id, f'{asset}.{extension}')
    return filepaths

def get_asset_filepaths_df(
    items,
    assets,
    output_folder,
):
    data = {
        'id': [],
        'asset': [],
        'current filepath': [],
        'new filepath': [],
    }
    for item in items:
        raster_filepaths_cur = get_filepaths(
            item = item,
            assets = assets,
            output_folder = output_folder,
        )
        raster_filepaths_new = get_new_filepaths(
            item_id = item.id,
            assets = assets,
            output_folder = output_folder,
        )
        for asset in assets:
            data['id'].append(item.id)
            data['asset'].append(asset)
            data['current filepath'].append(raster_filepaths_cur[asset])
            data['new filepath'].append(raster_filepaths_new[asset])

    raster_asset_filepaths_df = pd.DataFrame(data=data)
    return raster_asset_filepaths_df


def move_assets(
    cur_filepaths:list,
    new_filepaths:list,
    ignore_missing = False,
):
    if len(cur_filepaths) != len(new_filepaths):
        raise ValueError(f'Length of cur_filepaths [{len(cur_filepaths)}] and new_filepaths [{len(new_filepaths)}] do not match.')
    for cur_filepath, new_filepath in zip(
        cur_filepaths, new_filepaths,
    ):
        if not os.path.exists(cur_filepath) and ignore_missing:
            continue

        os.makedirs(os.path.dirname(new_filepath), exist_ok=True)
        shutil.move(cur_filepath, new_filepath)


def main():
    parser = argparse.ArgumentParser(
        prog = 'python dl_sentinel2_l2a_stacdl_mpc.py',
        description = (
            "Using STAC-Downloader to download Sentinel 2 L2A from "
            "Microsoft Planatary Computer."
        ),
    )

    parser.add_argument('shapefilepath', help="shapefile containing polygon of interest.")
    parser.add_argument('startdate', help='style: 2024-08-20 or 2024-08-20T13:54:40.022Z')
    parser.add_argument('enddate', help='style: 2024-08-20 or 2024-08-20T13:54:40.022Z')
    parser.add_argument('export_folderpath', help='Folderpath where the datacube should be saved.')
    parser.add_argument('--assets', help="Assets to be downloaded. To be passed as a comma separated integers. Example: 'B04,B08,SCL'")
    parser.add_argument('--njobs', help="Number of parallel operations to execute for parallelised steps.", type=int)
    parser.add_argument('--max-cloud-cover', required=False, default=100, type=float, help='[default = 100] Filter STAC query by cloud cover.')
    parser.add_argument('--overwrite', action='store_true')

    args = parser.parse_args()
    
    RASTER_ASSETS = args.assets.split(',')
    GEOMETRY_PATH = args.shapefilepath
    START_DATE = args.startdate
    END_DATE = args.enddate
    MAX_CLOUD_COVER = args.max_cloud_cover
    OUTPUT_FOLDER = os.path.abspath(args.export_folderpath)

    OVERWRITE = args.overwrite  # Set to True to overwrite existing files

    NUM_WORKERS = args.njobs  # Number of parallel workers for downloading

    MODIFIER = planetary_computer.sign # Required from MPC

    gdf = gpd.read_file(GEOMETRY_PATH)

    stac_downloader = STACDownloader(
        catalog_url = STACK_CATALOG_URL, 
        logger = logger, 
        stac_item_modifier = MODIFIER,
    )

    try:
        geometry = prepare_geometry(gdf)
        items = stac_downloader.query_catalog(
            collection_name=STAC_COLLECTION_NAME,
            start_date=START_DATE,
            end_date=END_DATE,
            geometry=geometry,
            query={"eo:cloud_cover": {"lt": MAX_CLOUD_COVER}},
        )
    except Exception as e:
        # Try using bounding boxes instead of exact polygon to reduce size.
        # This is a typical reason why the requests fails.
        geometry = prepare_geometry(gdf, enveloped=True)
        items = stac_downloader.query_catalog(
            collection_name=STAC_COLLECTION_NAME,
            start_date=START_DATE,
            end_date=END_DATE,
            geometry=geometry,
            query={"eo:cloud_cover": {"lt": MAX_CLOUD_COVER}},
        )

    asset_filepaths_df = get_asset_filepaths_df(
        items = items,
        assets = RASTER_ASSETS,
        output_folder = OUTPUT_FOLDER,
    )

    logger.info("Starting download of items...")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    _ = stac_downloader.download_items(
        items = items,
        raster_assets = [],
        file_assets = RASTER_ASSETS,
        mask_assets = [],
        output_folder = OUTPUT_FOLDER,
        overwrite = OVERWRITE,
        resolution = 10,
        resampling_spec = None,
        num_workers = NUM_WORKERS,
    )

    move_assets(
        cur_filepaths = asset_filepaths_df['current filepath'],
        new_filepaths = asset_filepaths_df['new filepath'],
        ignore_missing = True,
    )

    catalog_gdf = get_catalog_gdf(
        items = items,
        assets = RASTER_ASSETS,
        output_folder = OUTPUT_FOLDER,
        extension = 'tif',
    )

    catalog_filepath = os.path.join(OUTPUT_FOLDER, 'catalog.geojson')
    catalog_gdf.to_file(catalog_filepath)

    print(f'Catalog saved at: {catalog_filepath}')


if __name__ == '__main__':
    main()
