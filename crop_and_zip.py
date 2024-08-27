import geopandas as gpd
import datetime
import os
import rasterio
import shutil

import sys
sys.path.append('..')

import create_stack
import modify_images


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
    out_folderpath = zip_filepath.removesuffix('.zip')

    band_filepaths_df = create_stack.crop_and_reproject(
        shapes_gdf = shapes_gdf,
        catalog_filepath = catalog_filepath,
        startdate = startdate,
        enddate = enddate,
        bands = bands,
        out_folderpath = out_folderpath,
        satellite_folderpath = satellite_folderpath,
        nodata = nodata,
        working_dir = working_dir,
        njobs = njobs,
        dst_crs = dst_crs,
    )

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
