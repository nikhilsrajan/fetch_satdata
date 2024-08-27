import geopandas as gpd
import datetime
import shapely.ops
import pandas as pd
import rasterio
import rasterio.warp
import os
import functools
import multiprocessing as mp
import tqdm

import modify_images


def filter_catalog(
    catalog_filepath:str,
    shapes_gdf:gpd.GeoDataFrame,
    startdate:datetime,
    enddate:datetime,
):
    """
    Assumes catalog has following columns:
    - timestamp
    - geometry
    """
    catalog_gdf = gpd.read_file(catalog_filepath)

    union_shape = shapely.ops.unary_union(
        shapes_gdf.to_crs(catalog_gdf.crs)['geometry']
    )
    union_shape_gdf = gpd.GeoDataFrame(
        data = {'geometry': [union_shape]},
        crs = catalog_gdf.crs,
    )

    dt_filtered_catalog_gdf = catalog_gdf[
        (catalog_gdf['timestamp'] >= pd.Timestamp(startdate, tz='UTC'))
        & (catalog_gdf['timestamp'] <= pd.Timestamp(enddate, tz='UTC'))
    ].reset_index(drop=True)

    filtered_catalog_gdf = \
    gpd.overlay(
        dt_filtered_catalog_gdf,
        union_shape_gdf,
    )

    filtered_catalog_gdf['area_contribution'] = filtered_catalog_gdf['geometry'].apply(
        lambda x: x.area / union_shape.area * 100 # area contribution in terms of %
    )

    return filtered_catalog_gdf


def get_intersecting_band_filepaths(
    shapes_gdf:gpd.GeoDataFrame,
    catalog_filepath:str,
    startdate:datetime,
    enddate:datetime,
    bands:list[str],
    ext:str = '.jp2',
):
    """
    Assumes the catalog has following columns:
    - id
    - timestamp
    - local_folderpath
    - files
    """
    catalog_gdf = filter_catalog(
        catalog_filepath = catalog_filepath,
        shapes_gdf = shapes_gdf,
        startdate = startdate,
        enddate = enddate,
    )

    data = {
        'id': [],
        'timestamp': [],
        'band': [],
        'filepath': [],
        'area_contribution': [],
    }

    for _id, timestamp, local_folderpath, \
        files, area_contribution in zip(
        catalog_gdf['id'],
        catalog_gdf['timestamp'],
        catalog_gdf['local_folderpath'],
        catalog_gdf['files'],
        catalog_gdf['area_contribution'],
    ):
        filenames_of_interest = {
            xii + ext 
            for xii in {
                xi.removesuffix(ext) 
                for xi in files.split(',')
            } & set(bands)
        }
        
        # if no files of interest present, skip
        if len(filenames_of_interest) == 0:
            continue

        _filepaths = [
            local_folderpath + '/' + filename_of_interest
            for filename_of_interest in filenames_of_interest
        ]

        data['id'] += [_id for _ in range(len(_filepaths))]
        data['timestamp'] += [timestamp for _ in range(len(_filepaths))]
        data['area_contribution'] += [area_contribution for _ in range(len(_filepaths))]
        data['band'] += [
            filename_of_interest.removesuffix(ext)
            for filename_of_interest in filenames_of_interest
        ]
        data['filepath'] += _filepaths

        del _filepaths

    band_filepaths_df = pd.DataFrame(data=data).sort_values(
        by = ['timestamp', 'band'],
        ascending = [True, True],
    ).reset_index(drop=True)

    return band_filepaths_df


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


def crop_and_reproject(
    shapes_gdf:gpd.GeoDataFrame,
    catalog_filepath:str,
    startdate:datetime.datetime,
    enddate:datetime.datetime,
    bands:list[str],
    out_folderpath:str,
    satellite_folderpath:str, # for maintaining the same folder structure
    nodata = 0,
    working_dir:str = None,
    njobs:int = 8,
    dst_crs = None,
):
    band_filepaths_df = \
    get_intersecting_band_filepaths(
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

    
    os.makedirs(out_folderpath, exist_ok=True)

    sequence = [
        (modify_images.crop, dict(shapes_gdf=shapes_gdf, nodata=nodata, all_touched=True)),
        (modify_images.reproject, dict(dst_crs=dst_crs)),
        (modify_images.crop, dict(shapes_gdf=shapes_gdf, nodata=nodata, all_touched=True)),
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

    band_filepaths_df.drop(columns=['crs', 'filepath'], inplace=True)
    band_filepaths_df.rename(columns={'out_filepath': 'filepath'}, inplace=True)

    return band_filepaths_df


def resample_to_selected_band_inplace(
    band_filepath_dict:dict[str,str],
    selected_band:str,
    shapes_gdf:gpd.GeoDataFrame,
    resampling = rasterio.warp.Resampling.nearest,
    nodata = 0,
    working_dir:str = None,
):
    ref_filepath = band_filepath_dict[selected_band]
    sequence = [
        (modify_images.resample_by_ref, dict(ref_filepath = ref_filepath,
                                             resampling = resampling)),
        (modify_images.crop, dict(shapes_gdf=shapes_gdf, nodata=nodata, all_touched=True))
    ]
    filepaths = []
    for band, band_filepath in band_filepath_dict.items():
        if band != selected_band:
            filepaths.append(band_filepath)
    filepaths.append(ref_filepath)

    for filepath in filepaths:
        modify_images.modify_image(
            src_filepath = filepath,
            dst_filepath = filepath,
            sequence = sequence,
            working_dir = working_dir,
        )


def resample_to_selected_band_inplace_by_df(
    band_filepaths_df:pd.DataFrame,
    shapes_gdf:gpd.GeoDataFrame,
    nodata = 0,
    resampling_ref_band:str = 'B08',
    resampling = rasterio.warp.Resampling.nearest,
    working_dir:str = None,
    njobs:int = 8,
):
    list_of_band_filepath_dicts = band_filepaths_df.groupby('id')[
        ['band', 'filepath']
    ].apply(
        lambda g: dict(map(tuple, g.values.tolist()))
    ).to_list()

    resample_to_selected_band_inplace_partial = \
    functools.partial(
        resample_to_selected_band_inplace,
        selected_band = resampling_ref_band,
        shapes_gdf = shapes_gdf,
        resampling = resampling,
        nodata = nodata,
        working_dir = working_dir,
    )

    with mp.Pool(njobs) as p:
        list(tqdm.tqdm(
            p.imap(resample_to_selected_band_inplace_partial, list_of_band_filepath_dicts), 
            total=len(list_of_band_filepath_dicts)
        ))


def create_stack(
    shapes_gdf:gpd.GeoDataFrame,
    catalog_filepath:str,
    startdate:datetime.datetime,
    enddate:datetime.datetime,
    bands:list[str],
    out_folderpath:str,
    satellite_folderpath:str, # for maintaining the same folder structure
    nodata = 0,
    working_dir:str = None,
    njobs:int = 8,
    dst_crs = None,
    resampling = rasterio.warp.Resampling.nearest,
    resampling_ref_band:str = 'B08',
):
    band_filepaths_df = crop_and_reproject(
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

    resample_to_selected_band_inplace_by_df(
        band_filepaths_df = band_filepaths_df,
        shapes_gdf = shapes_gdf,
        nodata = nodata,
        resampling_ref_band = resampling_ref_band,
        resampling = resampling,
        working_dir = working_dir,
        njobs = njobs,
    )

    return band_filepaths_df
