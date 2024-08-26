import os
import rasterio
import rasterio.warp
import geopandas as gpd
import multiprocessing as mp
import functools
import tqdm

import rsutils.utils


"""
Logic here is that every function mandatorily takes in two inputs in the
order -- src_filepath and dst_filepath.
"""


def modify_image(
    src_filepath:str,
    dst_filepath:str,
    sequence:list,
    working_dir:str = None,
    delete_temp_files:bool = True,
):
    temp_src_filepath = src_filepath

    for func, kwargs in sequence:
        process_name = func.__qualname__
        temp_dst_filepath = rsutils.utils.add_epochs_prefix(
            filepath = temp_src_filepath,
            prefix = process_name,
            new_folderpath = working_dir,
        )

        func(
            src_filepath = temp_src_filepath, 
            dst_filepath = temp_dst_filepath, 
            **kwargs,
        )

        if delete_temp_files and src_filepath != temp_src_filepath:
            os.remove(temp_src_filepath)

        temp_src_filepath = temp_dst_filepath

    dst_folderpath = os.path.split(dst_filepath)[0]
    os.makedirs(dst_folderpath, exist_ok=True)
    os.rename(temp_dst_filepath, dst_filepath)

    return os.path.exists(dst_filepath)


def _modify_image_by_tuple(
    src_filepath_dst_filepath_tuple:tuple[str,str],
    sequence:list,
    working_dir:str = None,
):
    src_filepath, dst_filepath = src_filepath_dst_filepath_tuple
    return modify_image(
        src_filepath = src_filepath,
        dst_filepath = dst_filepath,
        sequence = sequence,
        working_dir = working_dir,
    )


def modify_images(
    src_filepaths:list[str],
    dst_filepaths:list[str],
    sequence:list,
    working_dir:str = None,
    njobs:int = mp.cpu_count() - 2,
):
    if len(src_filepaths) != len(dst_filepaths):
        raise ValueError('Size of src_filepaths and dst_filepaths do not match.')
    
    _modify_image_by_tuple_partial = functools.partial(
        _modify_image_by_tuple,
        sequence = sequence,
        working_dir = working_dir,
    )

    src_filepath_dst_filepath_tuples = list(zip(src_filepaths, dst_filepaths))

    with mp.Pool(njobs) as p:
        successes = list(tqdm.tqdm(
            p.imap(_modify_image_by_tuple_partial, src_filepath_dst_filepath_tuples), 
            total=len(src_filepath_dst_filepath_tuples)
        ))
    
    return successes


def crop(
    src_filepath:str,
    dst_filepath:str,
    shapes_gdf:gpd.GeoDataFrame,
    nodata = None,
    all_touched:bool = False,
):
    out_ndarray, out_meta = rsutils.utils.crop_tif(
        src_filepath = src_filepath,
        shapes_gdf = shapes_gdf,
        nodata = nodata,
        all_touched = all_touched,
    )

    with rasterio.open(dst_filepath, 'w', **out_meta) as dst:
        dst.write(out_ndarray)


def reproject(
    src_filepath:str,
    dst_filepath:str,
    dst_crs,
    resampling = rasterio.warp.Resampling.nearest,
):
    # https://rasterio.readthedocs.io/en/stable/topics/reproject.html
    with rasterio.open(src_filepath) as src:
        transform, width, height = rasterio.warp.calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        kwargs = rsutils.utils.driver_specific_meta_updates(meta=kwargs)

        with rasterio.open(dst_filepath, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                rasterio.warp.reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=resampling)


def resample_by_ref(
    src_filepath:str,
    dst_filepath:str,
    ref_filepath:str,
    resampling = rasterio.warp.Resampling.nearest, 
):
    rsutils.utils.resample_tif(
        ref_filepath = ref_filepath,
        src_filepath = src_filepath,
        dst_filepath = dst_filepath,
        resampling = resampling,
    )
