import os
import geopandas as gpd
import datetime
import shutil
import numpy as np
import time

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


def parse_args():
    roi_filepath = sys.argv[1]
    startdate_str = sys.argv[2]
    enddate_str = sys.argv[3]
    bands_arg = sys.argv[4].lower()
    zip_filepath = sys.argv[5]
    njobs = int(sys.argv[6])
    s2cloudless_chunksize = int(sys.argv[7])
    cloud_threshold = float(sys.argv[8])
    mosaic_days = int(sys.argv[9])

    shapes_gdf = gpd.read_file(roi_filepath)
    startdate = datetime.datetime.strptime(startdate_str, '%Y-%m-%d')
    enddate = datetime.datetime.strptime(enddate_str, '%Y-%m-%d')
    if bands_arg == 'all':
        bands = [
            'B01', 'B02', 'B03', 'B04',
            'B05', 'B06', 'B07', 'B08',
            'B8A', 'B09', 'B10', 'B11',
            'B12',
        ]
    else:
        bands = bands_arg.upper().split(',')

    if s2cloudless_chunksize < 0:
        raise ValueError('s2cloudless_chunksize can not be negative.')
    if cloud_threshold < 0 or cloud_threshold > 1:
        raise ValueError('cloud_threshold must be from 0-1')
    if mosaic_days < 0:
        raise ValueError('mosaic_days can not be negative.')

    print(f'roi_filepath = {roi_filepath}')
    print(f'startdate = {startdate_str}')
    print(f'enddate = {enddate_str}')
    print(f'bands = {bands}')
    print(f'zip_filepath = {zip_filepath}')
    print(f'njobs = {njobs}')
    print(f's2cloudless_chunksize = {s2cloudless_chunksize}')
    print(f'cloud_threshold = {cloud_threshold}')
    print(f'mosaic_days = {mosaic_days}')
    
    return shapes_gdf, startdate, enddate, bands, \
        zip_filepath, njobs, s2cloudless_chunksize, \
        cloud_threshold, mosaic_days


if __name__ == '__main__':
    start_time = time.time()

    if len(sys.argv) == 1:
        print((
            "Arguments:\n"
            "- ROI shape filepath\n"
            "- startdate (YYYY-MM-DD)\n"
            "- enddate (YYYY-MM-DD)\n"
            "- 'all' or bands separated by comma, eg 'B02,B03,B04,B08'\n"
            "- zip filepath without extension\n"
            "- njobs (recommended: 16)\n"
            "- s2cloudless chunksize (recommended: 1)\n"
            "- cloud_threshold (float, 0-1)\n"
            "- mosaic_days (int)\n"
        ))
        exit()

    NODATA = 0 # since the script is hardcoded for sentinel-2-l1c

    shapes_gdf, startdate, enddate, bands, \
    zip_filepath, njobs, s2cloudless_chunksize, \
    cloud_threshold, mosaic_days = parse_args()

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

    end_time = time.time()
    print(f'--- t_elapsed: {round(end_time - start_time, 2)} secs ---')
