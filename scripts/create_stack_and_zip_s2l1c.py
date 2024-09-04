import os
import geopandas as gpd
import datetime
import shutil
import numpy as np
import time
import argparse

import sys
sys.path.append('..')

import config
import create_stack
import extract_metadata
import stack_ops
import rsutils.s2_grid_utils


def add_s2cloudless_band_and_save(
    folderpath:str,
    chunksize:int,
    njobs:int,
    print_messages:bool = True,
):
    bands, metadata = create_stack.load_stack(
        folderpath = folderpath
    )
    bands, metadata = stack_ops.run_s2cloudless(
        bands = bands,
        metadata = metadata,
        chunksize = chunksize,
        njobs = njobs,
        print_messages = print_messages,
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


REF_BAND_ORDER = [
    'B08', 'B04', 'B03', 'B02', # 10m
    'B8A', 'B11', 'B12', 'B05', 'B06', 'B07', # 20m
    'B01', 'B09', 'B10', # 60m
]


def main(
    shapes_gdf:gpd.GeoDataFrame, 
    startdate:datetime.datetime, 
    enddate:datetime.datetime, 
    bands:list[str],
    zip_filepath:str, 
    njobs:int, 
    resampling_ref_band:str,
    s2cloudless_chunksize:int = None,
    cloud_threshold:float = 1, 
    mosaic_days = None,
    print_messages:bool = True,
    if_missing_files:bool = 'raise_error',
):
    start_time = time.time()

    NODATA = 0 # since the script is hardcoded for sentinel-2-l1c
    EXT = '.jp2' # since the script is hardcoded for sentinel-2-l1c

    if print_messages:
        print('--- run ---')

    out_folderpath = zip_filepath.removesuffix('.zip')

    create_stack.create_stack(
        shapes_gdf = shapes_gdf,
        catalog_filepath = config.FILEPATH_SENTINEL2_LOCAL_CATALOG,
        startdate = startdate,
        enddate = enddate,
        bands = bands,
        out_folderpath = out_folderpath,
        working_dir = out_folderpath,
        nodata = NODATA,
        njobs = njobs,
        resampling_ref_band = resampling_ref_band,
        delete_working_dir = True,
        satellite_folderpath = None,
        print_messages = print_messages,
        ext = EXT,
        if_missing_files = if_missing_files,
    )

    if print_messages:
        print('Extracting mean_sun_angle:')
    mean_sun_angle_df = \
    extract_metadata.extract_s2l1c_mean_sun_angle(
        shapes_gdf = shapes_gdf,
        catalog_filepath = config.FILEPATH_SENTINEL2_LOCAL_CATALOG,
        startdate = startdate,
        enddate = enddate,
        print_messages = print_messages,
    )
    mean_sun_angle_df.to_csv(os.path.join(zip_filepath, 'mean_sun_angle.csv'), index=False)

    if s2cloudless_chunksize is not None:
        if print_messages:
            print('Running s2cloudless:')
        add_s2cloudless_band_and_save(
            folderpath = zip_filepath, 
            chunksize = s2cloudless_chunksize,
            njobs = njobs,
            print_messages = print_messages,
        )

    if cloud_threshold is not None and mosaic_days is not None:
        if print_messages:
            print(
                f'Performing cloud masked median mosaicing - '
                f'cloud_threshold={cloud_threshold}, mosaic_days={mosaic_days} ... ',
                end='',
            )
        cloud_masked_median_mosaicing(
            folderpath = zip_filepath,
            cloud_threshold = cloud_threshold,
            startdate = startdate,
            enddate = enddate,
            mosaic_days = mosaic_days,
        )
        if print_messages:
            print('Done.')

    if print_messages:
        print('Zipping files...')
    final_zip_filepath = shutil.make_archive(
        out_folderpath,
        'zip',
        out_folderpath,
    )

    shutil.rmtree(out_folderpath)

    if print_messages:
        print(f'Outputs zipped and saved at: {os.path.abspath(final_zip_filepath)}')

    end_time = time.time()

    if print_messages:
        print(f'--- t_elapsed: {round(end_time - start_time, 2)} secs ---')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog = 'python create_stack_and_zip_s2l1c.py',
        description = (
            'Script to create a datacube of S2L1C data for given shapefile '
            'or s2grid id, and daterange. The script also runs s2cloudless to '
            'compute cloud probability if requested, and performs median '
            'mosaicing. The datacube is then saved into the specified folderpath '
            'and zipped.'    
        ),
        epilog = '--- Send your complaints to nsasiraj@umd.edu ---',
    )
    parser.add_argument('roi', help='filepath=path/to/shapefile | s2gridid=S2GridID')
    parser.add_argument('startdate', help='YYYY-MM-DD')
    parser.add_argument('enddate', help='YYYY-MM-DD')
    parser.add_argument('bands', default='all', help='all, or bands comma separated B02,B03,B04,B08 (default = all)')
    parser.add_argument('out', help='/path/to/output')
    parser.add_argument('--njobs', default=4, help='Number of cores to use. (default = 4)')
    parser.add_argument('-s2c', '--s2cloudless', action='store', required=False, help='Whether to run s2cloudless and if so at what chunk size.')
    parser.add_argument('-m', '--mosaic', action='store', required=False, help='Whether to perform median mosaicing and if so in intervals of how many days (most used interval: 20.')
    parser.add_argument('-c', '--cloud-threshold', action='store', default=0, required=False, help='The probabiliy threshold at and above which the pixel is considered cloud. Must be from 0 to 1.')
    parser.add_argument('--silent', action='store_true', help='To run the script without any print statements.')
    parser.add_argument('--ignore-missing-files', action='store_true', help='If there are missing files for requested region and date range, this option ignores the error and proceeds, except when there are no files present.')
    parser.add_argument('--warn-missing-files', action='store_true', help='If there are missing files for requested region and date range, this option raises a warning and proceeds, except when there are no files present.')
    args = parser.parse_args()

    if args.roi.startswith('filepath='):
        roi_filepath = args.roi.removeprefix('filepath=')
        shapes_gdf = gpd.read_file(roi_filepath)
    elif args.roi.startswith('s2gridid='):
        s2gridid = args.roi.removeprefix('s2gridid=')
        s2grid_geom = \
        rsutils.s2_grid_utils.get_grid_geometry_from_id(
            grid_id=s2gridid, scale_fact=1.1,
        )
        shapes_gdf = gpd.GeoDataFrame(
            data={'geometry':[s2grid_geom]}, 
            crs='epsg:4326',
        )
    else:
        raise ValueError(f"Invalid ROI input='{args.roi}'. Must start with 'filepath=' or 's2gridid='.")

    startdate = datetime.datetime.strptime(args.startdate, '%Y-%m-%d')
    
    enddate = datetime.datetime.strptime(args.enddate, '%Y-%m-%d')

    if args.bands == 'all':
        bands = [
            'B01', 'B02', 'B03', 'B04',
            'B05', 'B06', 'B07', 'B08',
            'B8A', 'B09', 'B10', 'B11',
            'B12',
        ]
    else:
        bands = args.bands.upper().split(',')

    zip_filepath = args.out.removesuffix('.zip') + '.zip'

    njobs = int(args.njobs)
    
    s2cloudless_chunksize = args.s2cloudless
    if s2cloudless_chunksize is not None:
        s2cloudless_chunksize = int(s2cloudless_chunksize)
        if s2cloudless_chunksize < 0:
            raise ValueError('s2cloudless_chunksize can not be negative.')
        
    cloud_threshold = float(args.cloud_threshold)
    if cloud_threshold < 0 or cloud_threshold > 1:
        raise ValueError('cloud_threshold must be from 0-1')
    
    mosaic_days = args.mosaic
    if mosaic_days is not None:
        mosaic_days = int(mosaic_days)
        if mosaic_days < 0:
            raise ValueError('mosaic_days can not be negative.')
        
    print_messages = not args.silent

    if_missing_files = 'raise_error'
    if args.ignore_missing_files:
        if_missing_files = None
    if args.warn_missing_files:
        if_missing_files = 'warn'

    for ref_band_candidate in REF_BAND_ORDER:
        if ref_band_candidate in bands:
            resampling_ref_band = ref_band_candidate
            break

    if print_messages:
        print('--- inputs ---')
        print(f'roi: {args.roi}')
        print(f'startdate: {args.startdate}')
        print(f'enddate: {args.enddate}')
        print(f'bands: {bands}')
        print(f'out: {zip_filepath}')
        print(f'njobs: {njobs}')
        if s2cloudless_chunksize is not None:
            print(f's2cloudless_chunksize: {s2cloudless_chunksize}')
        if mosaic_days is not None:
            print(f'cloud_threshold: {cloud_threshold}')
            print(f'mosaic_days: {mosaic_days}')
        if if_missing_files is not None:
            print(f'if_missing_files: {if_missing_files}')
        print(f'resampling_ref_band: {resampling_ref_band}')

    main(
        shapes_gdf = shapes_gdf,
        startdate = startdate,
        enddate = enddate,
        bands = bands,
        zip_filepath = zip_filepath,
        njobs = njobs,
        s2cloudless_chunksize = s2cloudless_chunksize,
        cloud_threshold = cloud_threshold,
        mosaic_days = mosaic_days,
        print_messages = print_messages,
        if_missing_files = if_missing_files,
        resampling_ref_band = resampling_ref_band,
    )
