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
import json
import os

import sys
sys.path.append('..')

import config
import create_s2l1c_datacube
import rsutils.s2_grid_utils
import exceptions


RET_SUCCESS = 0
RET_FAILED = 1
RET_CLI_FAILED = 2 # see argparse


def log(log_filepath:str, entry:dict):
    entry['timestamp'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    if os.path.exists(log_filepath):
        with open(log_filepath) as h:
            _logs = json.load(h)
    else:
        _logs = []
    
    _logs.append(entry)

    with open(log_filepath, 'w') as h:
        json.dump(_logs, h)


def main_single(
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
    if_new_config:str = 'raise_error', # by default left as raise_error as create_s2l1c_datacubes.py should have created them already.
    overwrite:bool = False,
    log_filepath:str = None,
    override_gap_days:int = None,
):
    ret_code = RET_FAILED

    entry = {
        'parameters': {
            'roi_name': roi_name,
            'startdate': startdate.strftime('%Y-%m-%d'),
            'enddate': enddate.strftime('%Y-%m-%d'),
            'bands': bands,
            's2cloudless_chunksize': s2cloudless_chunksize,
            'cloud_threshold': cloud_threshold,
            'mosaic_days': mosaic_days,
        }
    }
    
    error_type = None
    error_message = None

    try:
        ret = create_s2l1c_datacube.create_s2l1c_datacube_and_update_catalog(
            satellite_catalog_filepath = config.FILEPATH_SENTINEL2_LOCAL_CATALOG,
            datacube_catalog_filepath = datacube_catalog_filepath,
            configs_filepath = config.FILEPATH_S2L1C_DATACUBE_CONFIG_TRACKER, # this needs to be updated in create_s2l1c_datacubes.py
            datacubes_folderpath = config.FOLDERPATH_DATACUBES_S2L1C + '/urgent', # designed for urgent case
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
            if_new_config = if_new_config,
            overwrite = overwrite,
            override_gap_days = override_gap_days,
        )
        if ret == create_s2l1c_datacube.DATACUBE_CREATED:
            entry.update({'status': 'success', 'type':'new'})
        elif ret == create_s2l1c_datacube.DATACUBE_ALREADY_EXISTS:
            entry.update({'status': 'success', 'type':'already exists'})
        elif ret == create_s2l1c_datacube.DATACUBE_OVERWRITTEN:
            entry.update({'status': 'success', 'type':'overwritten'})

        ret_code = RET_SUCCESS

    except exceptions.CatalogManagerException as e:
        error_type = 'CatalogManagerException'
        error_message = str(e)
    
    except exceptions.DatacubeException as e:
        error_type = 'DatacubeException'
        error_message = str(e)

    except exceptions.MetadataException as e:
        error_type = 'MetadataException'
        error_message = str(e)


    if error_type is not None:
        entry.update({'status': 'failed',
                      'error_type': error_type,
                      'error_message': error_message})
        if print_messages:
            print(f'FAILED -- {error_type}: {error_message}')

    if log_filepath is not None:
        log(log_filepath=log_filepath, entry = entry)
    
    return ret_code


def main(
    roi_name_col:str,
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
    if_new_config:str = 'raise_error', # by default left as raise_error as create_s2l1c_datacubes.py should have created them already.
    overwrite:bool = False,
    log_filepath:str = None,
    override_gap_days:int = None,
):
    N = shapes_gdf.shape[0]
    i = 0
    ret_codes = []
    for index, row in shapes_gdf.iterrows():
        start_time = time.time()
        i += 1
        if print_messages:
            print(f'--- run --- [{i} / {N}]')
        roi_name = row[roi_name_col]
        geom = row['geometry']
        _shapes_gdf = gpd.GeoDataFrame({
            'geometry': [geom]
        }, crs=shapes_gdf.crs)
        ret_code = main_single(
            roi_name = roi_name,
            shapes_gdf = _shapes_gdf,
            startdate = startdate,
            enddate = enddate,
            bands = bands,
            njobs = njobs,
            datacube_catalog_filepath = datacube_catalog_filepath,
            s2cloudless_chunksize = s2cloudless_chunksize,
            cloud_threshold = cloud_threshold,
            mosaic_days = mosaic_days,
            print_messages = print_messages,
            if_missing_files = if_missing_files,
            if_new_config = if_new_config,
            overwrite = overwrite,
            log_filepath = log_filepath,
            override_gap_days = override_gap_days,
        )
        end_time = time.time()
        if print_messages:
            print(f'--- t_elapsed: {round(end_time - start_time, 2)} secs ---')
        ret_codes.append(ret_code)
    
    return ret_codes


if __name__ == '__main__':
    start_time = time.time()

    parser = argparse.ArgumentParser(
        prog = 'python _create_s2l1c_datacubes_urgent.py',
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
    # parser.add_argument('roi_name', help='Name to uniquely identify the region of the interest. This is logged in the catalog and will throw error if same shape has been given a different name earlier or if the name is already used.')
    # parser.add_argument('roi', help='filepath=path/to/shapefile | s2gridid=S2GridID')
    parser.add_argument('filepath')
    parser.add_argument('roi_name_col')
    parser.add_argument('startdate', help='YYYY-MM-DD')
    parser.add_argument('enddate', help='YYYY-MM-DD')
    parser.add_argument('bands', default='all', help='[default = all] all, or bands comma separated B02,B03,B04,B08')
    parser.add_argument('-j', '--njobs', default=4, help='[default = 4] Number of cores to use.')
    parser.add_argument('-s2c', '--s2cloudless', action='store', metavar='CHUNKSIZE', required=False, help='Whether to run s2cloudless and if so at what chunk size.')
    parser.add_argument('-m', '--mosaic', action='store', required=False, help='[suggested = 20] Whether to perform median mosaicing and if so in intervals of how many days.')
    parser.add_argument('-c', '--cloud-threshold', action='store', default=0, required=False, help='The probabiliy threshold at and above which the pixel is considered cloud. Must be from 0 to 1.')
    parser.add_argument('--silent', action='store_true', help='To run the script without any print statements.')
    parser.add_argument('--ignore-missing-files', action='store_true', help='If there are missing files for requested region and date range, this option ignores the error and proceeds, except when there are no files present.')
    parser.add_argument('--warn-missing-files', action='store_true', help='If there are missing files for requested region and date range, this option raises a warning and proceeds, except when there are no files present.')
    parser.add_argument('--datacube-catalog', action='store', required=True, help='Datacube catalog filepath where the catalog is to be created. It is a variable in this script to avoid deadlocks/race-conditions when this script is parallelised by create_s2l1c_datacubes.py')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing datacube.')
    parser.add_argument('--add-new-config', action='store_true', help='To allow addition of new config.')
    parser.add_argument('--log-file', action='store', required=False, help='Log file where the status of the different runs would be appended to.')
    parser.add_argument('--override-gap-days', action='store', required=False, help='Override the permitted time gap.')

    args = parser.parse_args()

    # roi_name = args.roi_name
    datacube_catalog_filepath = args.datacube_catalog
    log_filepath = args.log_file

    roi_filepath = args.filepath
    shapes_gdf = gpd.read_file(roi_filepath)

    roi_name_col = args.roi_name_col
    
    startdate = datetime.datetime.strptime(args.startdate, '%Y-%m-%d')
    
    enddate = datetime.datetime.strptime(args.enddate, '%Y-%m-%d')

    override_gap_days = args.override_gap_days

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

    if_new_config = 'raise_error'
    if args.add_new_config:
        if_new_config = None

    if print_messages:
        print('--- inputs ---')
        print(f'datacube_catalog_filepath: {datacube_catalog_filepath}')
        print(f'filepath: {roi_filepath}')
        print(f'roi_name_col: {roi_name_col}')
        # print(f'roi: {args.roi}')
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
        if if_new_config is not None:
            print(f'if_new_config: {if_new_config}')
        else:
            print(f'if_new_config: add')
        if log_filepath is not None:
            print(f'log-file: {log_filepath}')

    ret_codes = main(
        roi_name_col = roi_name_col,
        shapes_gdf = shapes_gdf,
        startdate = startdate,
        enddate = enddate,
        bands = bands,
        njobs = njobs,
        datacube_catalog_filepath = datacube_catalog_filepath,
        s2cloudless_chunksize = s2cloudless_chunksize,
        cloud_threshold = cloud_threshold,
        mosaic_days = mosaic_days,
        print_messages = print_messages,
        if_missing_files = if_missing_files,
        overwrite = overwrite,
        if_new_config = if_new_config,
        log_filepath = log_filepath,
        override_gap_days = override_gap_days,
    )
    
    end_time = time.time()

    if print_messages:
        print(f'--- t_elapsed: {round(end_time - start_time, 2)} secs ---')

    ret_code = RET_SUCCESS if sum(ret_codes) == 0 else RET_FAILED

    exit(ret_code)

