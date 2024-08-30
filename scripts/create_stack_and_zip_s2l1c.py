import os
import geopandas as gpd
import datetime
import shutil
import s2cloudless
import numpy as np

import sys
sys.path.append('..')

import config
import create_stack
import extract_metadata
import stack_ops


def add_s2cloudless_band_and_save(
    folderpath:str,
    chunksize:int,
    njobs:int,
):
    bands, metadata = create_stack.load_stack(
        folderpath = folderpath
    )
    bands, metadata = stack_ops.run_s2cloudless(
        bands = bands,
        metadata = metadata,
        chunksize = chunksize,
        njobs = njobs,
    )
    create_stack.save_stack(
        bands = bands,
        metadata = metadata,
        folderpath = folderpath,
    )


def cloud_masked_median_mosaicing(
    folderpath:str,
    cloud_threshold:float,
    startdate:datetime.datetime,
    enddate:datetime.datetime,
    mosaic_days:int,
):
    if cloud_threshold < 0 or cloud_threshold > 1:
        raise ValueError('cloud_threshold must be from 0-1')
    if mosaic_days < 0:
        raise ValueError('mosaic_days can not be negative.')

    bands, metadata = create_stack.load_stack(
        folderpath = folderpath
    )
    band_indices = {band:index for index,band in enumerate(metadata['bands'])}
    if 'CMK' not in band_indices.keys():
        raise ValueError(f'CMK band not present in bands in folderpath: {folderpath}')
    
    cmk_index = band_indices['CMK']
    non_cmk_indices = [band_indices[band] for band in metadata['bands'] if band!='CMK']

    cmk = bands[:,:,:,cmk_index]
    bands_wo_cmk = bands[:,:,:,non_cmk_indices]

    bands_wo_cmk[np.where(cmk >= cloud_threshold * 10000)] = 0

    metadata['bands'].remove('CMK')

    mosaiced_bands, mosaiced_metadata \
    = stack_ops.median_mosaic(
        bands = bands_wo_cmk, 
        metadata = metadata,
        startdate = startdate,
        enddate = enddate,
        mosaic_days = mosaic_days,
    )

    create_stack.save_stack(
        bands = mosaiced_bands,
        metadata = mosaiced_metadata,
        folderpath = folderpath,
    )


if __name__ == '__main__':
    """
    Arguments:
    - ROI shape filepath
    - startdate (YYYY-MM-DD)
    - enddate (YYYY-MM-DD)
    - bands separated by comma, eg B02,B03,B04,B08
    - zip filepath without extension
    - njobs
    - s2cloudless chunksize (recommended: 10)
    - cloud_threshold (float, 0-1)
    - mosaic_days (int)
    """
    NODATA = 0 # since the script is hardcoded for sentinel-2-l1c

    shapes_gdf = gpd.read_file(sys.argv[1])
    startdate = datetime.datetime.strptime(sys.argv[2], '%Y-%m-%d')
    enddate = datetime.datetime.strptime(sys.argv[3], '%Y-%m-%d')
    bands = sys.argv[4].split(',')
    zip_filepath = sys.argv[5]
    njobs = int(sys.argv[6])
    s2cloudless_chunksize = int(sys.argv[7])
    cloud_threshold = float(sys.argv[8])
    mosaic_days = int(sys.argv[9])

    if s2cloudless_chunksize < 0:
        raise ValueError('s2cloudless_chunksize can not be negative.')
    if cloud_threshold < 0 or cloud_threshold > 1:
        raise ValueError('cloud_threshold must be from 0-1')
    if mosaic_days < 0:
        raise ValueError('mosaic_days can not be negative.')

    create_stack.create_stack(
        shapes_gdf = shapes_gdf,
        catalog_filepath = config.FILEPATH_SENTINEL2_LOCAL_CATALOG,
        startdate = startdate,
        enddate = enddate,
        bands = bands,
        out_folderpath = zip_filepath,
        working_dir = zip_filepath,
        nodata = NODATA,
        njobs = njobs,
        resampling_ref_band = 'B08',
        delete_working_dir = True,
        satellite_folderpath = None,
    )

    print('Extracting mean_sun_angle:')
    mean_sun_angle_df = \
    extract_metadata.extract_s2l1c_mean_sun_angle(
        shapes_gdf = shapes_gdf,
        catalog_filepath = config.FILEPATH_SENTINEL2_LOCAL_CATALOG,
        startdate = startdate,
        enddate = enddate,
    )
    mean_sun_angle_df.to_csv(os.path.join(zip_filepath, 'mean_sun_angle.csv'), index=False)

    print('Running s2cloudless...')
    add_s2cloudless_band_and_save(
        folderpath = zip_filepath, 
        chunksize = s2cloudless_chunksize,
        njobs = njobs,
    )

    print(f'Performing cloud masked median mosaicing - cloud_threshold={cloud_threshold}, mosaic_days={mosaic_days}')
    cloud_masked_median_mosaicing(
        folderpath = zip_filepath,
        cloud_threshold = cloud_threshold,
        startdate = startdate,
        enddate = enddate,
        mosaic_days = mosaic_days,
    )

    print('Zipping files...')
    final_zip_filepath = shutil.make_archive(
        zip_filepath,
        'zip',
        zip_filepath,
    )

    shutil.rmtree(zip_filepath)

    print(f'Outputs zipped and saved at: {os.path.abspath(final_zip_filepath)}')
