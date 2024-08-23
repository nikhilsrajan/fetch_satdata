import geopandas as gpd
import datetime
import os
import rasterio
import shutil

import sys
sys.path.append('..')

import create_stack
import modify_images


def get_image_crs(filepath:str):
    with rasterio.open(filepath) as src:
        crs = src.crs
    return crs


def change_parent_folderpath(
    filepath:str,
    parent_folderpath:str,
    new_parent_folderpath:str,
):
    filepath = os.path.abspath(filepath)
    parent_folderpath = os.path.abspath(parent_folderpath)
    return os.path.join(
        new_parent_folderpath,
        os.path.relpath(path=filepath,
                        start=parent_folderpath)
    )


def crop_and_zip(
    shapes_gdf:gpd.GeoDataFrame,
    catalog_filepath:str,
    startdate:datetime.datetime,
    enddate:datetime.datetime,
    bands:list[str],
    zip_filepath:str, # should not have extension, automatically added by shutil.make_archive
    satellite_folderpath:str, # for maintaining the same folder structre
    nodata = 0,
    working_dir:str = None,
    njobs:int = 8,
    dst_crs = None,
):
    band_filepaths_df = \
    create_stack.get_intersecting_band_filepaths(
        shapes_gdf = shapes_gdf,
        startdate = startdate,
        enddate = enddate,
        catalog_filepath = catalog_filepath,
        bands = bands,
    )

    if band_filepaths_df.shape[0] == 0:
        return None
    
    band_filepaths_df['crs'] = band_filepaths_df['filepath'].apply(lambda x: str(get_image_crs(filepath=x)))

    if dst_crs is None:
        group_by_crs_df = band_filepaths_df.groupby(by='crs')
        area_contribution_mean = group_by_crs_df['area_contribution'].mean()
        sorted_area_contribution_mean = area_contribution_mean.sort_values(ascending=False)
        max_area_contribution_crs = sorted_area_contribution_mean.index[0]
        dst_crs = max_area_contribution_crs

    out_folderpath = zip_filepath.removesuffix('.zip')
    
    os.makedirs(out_folderpath, exist_ok=True)

    sequence = [
        (modify_images.crop, dict(shapes_gdf=shapes_gdf, nodata=nodata)),
        (modify_images.reproject, dict(dst_crs=dst_crs)),
        (modify_images.crop, dict(shapes_gdf=shapes_gdf, nodata=nodata)),
    ]

    satellite_folderpath = os.path.abspath(satellite_folderpath)

    band_filepaths_df['out_filepath'] = band_filepaths_df['filepath'].apply(
        lambda filepath: change_parent_folderpath(
            filepath = filepath,
            parent_folderpath = satellite_folderpath,
            new_parent_folderpath = out_folderpath,
        )
    )

    successes = modify_images.modify_images(
        src_filepaths = band_filepaths_df['filepath'],
        dst_filepaths = band_filepaths_df['out_filepath'],
        sequence = sequence,
        working_dir = working_dir,
        njobs = njobs,
    )

    print(f'Successful: {sum(successes)} / {len(successes)}')

    band_filepaths_df.drop(columns=['crs', 'filepath'], inplace=True)
    band_filepaths_df.rename(columns={'out_filepath':'filepath'}, inplace=True)
    band_filepaths_df['filepath'] = band_filepaths_df['filepath'].apply(
        lambda filepath: os.path.relpath(
            path = filepath,
            start = out_folderpath,
        )
    )
    band_filepaths_df.to_csv(os.path.join(out_folderpath, 'catalog.csv'), index=False)

    final_zip_filepath = shutil.make_archive(
        zip_filepath,
        'zip',
        out_folderpath,
    )

    shutil.rmtree(out_folderpath)

    return final_zip_filepath
