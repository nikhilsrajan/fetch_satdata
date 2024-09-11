"""
Child script for 'create_s2l1c_datacubes.py'. Since python doesn't allow multiprocessed
operations to have multiprocessed child operations within it to avoid zombies (detached 
processes), I intend to call this script from 'create_s2l1c_datacubes.py' through multi-
threading. This script as such is never intended to be used by itself, but I ofcourse
will for the purposes of testing (krkrkkk)

--- nikhilsasirajan@gmail.com ---

"""

import argparse
import geopandas as gpd
import datetime
import time

import sys
sys.path.append('..')

import config
import create_s2l1c_datacube
import rsutils.s2_grid_utils


def main(
    roi_name:str,
    shapes_gdf:gpd.GeoDataFrame, 
    startdate:datetime.datetime, 
    enddate:datetime.datetime, 
    bands:list[str],
    njobs:int, 
    datacube_catalog_filepath:str,
    s2cloudless_chunksize:int = None,
    cloud_threshold:float = 1, 
    mosaic_days = None,
    print_messages:bool = False,
    if_missing_files:str = 'raise_error', # options: ['raise_error', 'warn', None]
    overwrite:bool = False,
):
    if print_messages:
        print('--- run ---')
    
    create_s2l1c_datacube.create_s2l1c_datacube_and_update_catalog(
        satellite_catalog_filepath = config.FILEPATH_SENTINEL2_LOCAL_CATALOG,
        datacube_catalog_filepath = datacube_catalog_filepath,
        configs_filepath = config.FILEPATH_S2L1C_DATACUBE_CONFIG_TRACKER, # this needs to be updated in create_s2l1c_datacubes.py
        datacubes_folderpath = config.FOLDERPATH_DATACUBES_S2L1C,
        roi_name = roi_name,
        shapes_gdf = shapes_gdf,
        startdate = startdate,
        enddate = enddate,
        bands = bands,
        njobs = njobs,
        s2cloudless_chunksize = s2cloudless_chunksize,
        cloud_threshold = cloud_threshold,
        mosaic_days = mosaic_days,
        print_messages = print_messages,
        if_missing_files = if_missing_files,
        if_new_config = 'raise_error', # by default left as raise_error as create_s2l1c_datacubes.py should have created them already.
        overwrite = overwrite,
    )


if __name__ == '__main__':
    start_time = time.time()

    parser = argparse.ArgumentParser(
        prog = 'python _create_s2l1c_datacube.py',
        description = (
            'THIS SCRIPT IS NOT MEANT TO BE USED BY A USER. '
            'THIS IS A CHILD SCRIPT FOR create_s2l1c_datacubes.py '
            'PLEASE USE create_s2l1c_datacubes.py -- '
            'Script to create a datacube of S2L1C data for given shapefile '
            'or s2grid id, and daterange. The script also runs s2cloudless to '
            'compute cloud probability if requested, and performs median '
            'mosaicing. The datacube is then saved into the datacubes folderpath '
            'and the catalog is updated.'
        ),
        epilog = f"--- Send your complaints to {','.join(config.MAINTAINERS)} ---",
    )
    parser.add_argument('roi_name', help='Name to uniquely identify the region of the interest. This is logged in the catalog and will throw error if same shape has been given a different name earlier or if the name is already used.')
    parser.add_argument('roi', help='filepath=path/to/shapefile | s2gridid=S2GridID')
    parser.add_argument('startdate', help='YYYY-MM-DD')
    parser.add_argument('enddate', help='YYYY-MM-DD')
    parser.add_argument('bands', default='all', help='all, or bands comma separated B02,B03,B04,B08 (default = all)')
    parser.add_argument('--njobs', default=4, help='Number of cores to use. (default = 4)')
    parser.add_argument('-s2c', '--s2cloudless', action='store', required=False, help='Whether to run s2cloudless and if so at what chunk size.')
    parser.add_argument('-m', '--mosaic', action='store', required=False, help='Whether to perform median mosaicing and if so in intervals of how many days (most used interval: 20.')
    parser.add_argument('-c', '--cloud-threshold', action='store', default=0, required=False, help='The probabiliy threshold at and above which the pixel is considered cloud. Must be from 0 to 1.')
    parser.add_argument('--silent', action='store_true', help='To run the script without any print statements.')
    parser.add_argument('--ignore-missing-files', action='store_true', help='If there are missing files for requested region and date range, this option ignores the error and proceeds, except when there are no files present.')
    parser.add_argument('--warn-missing-files', action='store_true', help='If there are missing files for requested region and date range, this option raises a warning and proceeds, except when there are no files present.')
    parser.add_argument('--datacube-catalog', action='store', help='Datacube catalog filepath where the catalog is to be created. It is a variable in this script to avoid deadlocks/race-conditions when this script is parallelised by create_s2l1c_datacubes.py')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing datacube.')

    args = parser.parse_args()

    roi_name = args.roi_name

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

    overwrite = args.overwrite

    if print_messages:
        print('--- inputs ---')
        print(f'roi-name: {roi_name}')
        print(f'roi: {args.roi}')
        print(f'startdate: {args.startdate}')
        print(f'enddate: {args.enddate}')
        print(f'bands: {bands}')
        print(f'njobs: {njobs}')
        if s2cloudless_chunksize is not None:
            print(f's2cloudless_chunksize: {s2cloudless_chunksize}')
        if mosaic_days is not None:
            print(f'cloud_threshold: {cloud_threshold}')
            print(f'mosaic_days: {mosaic_days}')
        if if_missing_files is not None:
            print(f'if_missing_files: {if_missing_files}')
        if overwrite:
            print('OVERWRITE: True')

    main(
        roi_name = roi_name,
        shapes_gdf = shapes_gdf,
        startdate = startdate,
        enddate = enddate,
        bands = bands,
        njobs = njobs,
        datacube_catalog_filepath = args.datacube_catalog,
        s2cloudless_chunksize = s2cloudless_chunksize,
        cloud_threshold = cloud_threshold,
        mosaic_days = mosaic_days,
        print_messages = print_messages,
        if_missing_files = if_missing_files,
        overwrite = overwrite,
    )
    
    end_time = time.time()

    if print_messages:
        print(f'--- t_elapsed: {round(end_time - start_time, 2)} secs ---')

