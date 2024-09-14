"""
Steps:
0. create a logging mechanism [DONE]
1. go through config tracker and all configurations passed and ask user if they
   wish to add new configs (have a flag to directly say yes)
2. update config tracker with new configurations to avoid deadlocks/race-conditions
3. check for roi_name and geometry conflicts and raise error if any present
4. create a copy of the datacube catalog with new name for each thread
5. create a list of cli inputs
6. execute cli inputs via multi-threading -- stagger the calls by random or fixed delays
7. merge the datacube catalogs in a non-conflicting manner
"""

import argparse
import multiprocessing as mp
import multiprocessing.pool as mppool
import pandas as pd
import geopandas as gpd
import datetime
import subprocess
import tqdm
import shutil
import os
import json

import sys
sys.path.append('..')

import config
import create_s2l1c_datacube
import _create_s2l1c_datacube
import rsutils.utils


COL_S2GRIDID = 's2gridid'
COL_ROI_NAME = 'roi_name'
COL_ROI = 'roi' # later added, need not exist in the input file
COL_GEOMETRY = 'geometry'
COL_STARTDATE = 'startdate'
COL_ENDDATE = 'enddate'
COL_BANDS = 'bands'
COL_IF_MISSING_FILES = 'if_missing_files'
COL_S2CLOUDLESS_CHUNKSIZE = 's2cloudless_chunksize' # 0 means no s2cloudless running
COL_CLOUD_THRESHOLD = 'cloud_threshold' # None means no thresholding
COL_MOSAIC_DAYS = 'mosaic_days' # 0 means no median mosaicing
COL_SUCCESSFUL = 'successful'

EPSG_4326 = 'epsg:4326'


def split(a, n):
    # https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))


def run_cli(
    cli:str,
    datacube_catalog_filepath:str,
    log_filepath:str,
):
    cli = ' '.join([
        cli, 
        f'--datacube-catalog {datacube_catalog_filepath}',
        f'--log-file {log_filepath}',
        '--silent'
    ])
    ret = subprocess.run(cli.split())
    return ret.returncode


def run_cli_list(
    cli_list:list[str],
    datacube_catalog_filepath:str,
    log_filepath:str,
    pbar:tqdm.tqdm = None,
):
    ret_codes = []
    for cli in cli_list:
        ret_code = run_cli(
            cli=cli,
            datacube_catalog_filepath=datacube_catalog_filepath,
            log_filepath=log_filepath,
        )
        if pbar is not None:
            pbar.update()
        ret_codes.append(ret_code)
    return ret_codes


def merge_separate_catalogs_and_save(
    datacube_catalog_filepaths:list[str],
    main_datacube_catalog_filepath:str
):
    catalogs_gdfs = [gpd.read_file(dc_catalog_filepath) 
                     for dc_catalog_filepath in datacube_catalog_filepaths
                     if os.path.exists(dc_catalog_filepath)]
    merged_catalog_gdf = gpd.GeoDataFrame(
        pd.concat(catalogs_gdfs).drop_duplicates(), 
        crs=catalogs_gdfs[0].crs,
    )
    merged_catalog_gdf.to_file(main_datacube_catalog_filepath)
    for dc_catalog_filepath in datacube_catalog_filepaths:
        os.remove(dc_catalog_filepath)


def merge_logs_and_save(
    log_filepaths:list[str],
    main_log_filepath:str,
):
    combined_log = []
    for log_filepath in log_filepaths:
        if not os.path.exists(log_filepath):
            continue
        with open(log_filepath) as h:
            log = json.load(h)
        combined_log += log
        os.remove(log_filepath)
    
    with open(main_log_filepath, 'w') as h:
        json.dump(combined_log, h)


def run_cli_list_parallel(
    cli_inputs:list[str],
    max_compute:int,
    njobs:int,
    log_folderpath:str,
    main_datacube_catalog_filepath:str,
):
    main_log_filepath = os.path.join(
        log_folderpath, f'log_{rsutils.utils.get_epochs_str(length=0)}.json'
    )

    pbar = tqdm.tqdm(total=len(cli_inputs))

    parallel_thread_count = max_compute // njobs
    
    parallel_cli_inputs = split(cli_inputs, parallel_thread_count)

    datacube_catalog_filepaths, log_filepaths = \
    create_datacube_catalog_copies_and_log_filepaths(
        n = parallel_thread_count,
        main_datacube_catalog_filepath = main_datacube_catalog_filepath,
        log_folderpath = log_folderpath,
    )

    with mppool.ThreadPool(processes=parallel_thread_count) as pool:
    # pool = mp.Pool(processes = parallel_thread_count)
        ret_code_lists = pool.starmap(run_cli_list, zip(parallel_cli_inputs,
                                                        datacube_catalog_filepaths,
                                                        log_filepaths,
                                                        [pbar for _ in range(parallel_thread_count)]))
        pool.close()
        pool.join()

    ret_codes = []
    for _ret_codes in ret_code_lists:
        ret_codes += _ret_codes

    merge_separate_catalogs_and_save(
        datacube_catalog_filepaths = datacube_catalog_filepaths,
        main_datacube_catalog_filepath = main_datacube_catalog_filepath,
    )

    merge_logs_and_save(
        log_filepaths = log_filepaths,
        main_log_filepath = main_log_filepath,
    )

    successes = [ret_code == _create_s2l1c_datacube.RET_SUCCESS
                 for ret_code in ret_codes]
    
    return successes


def generate_cli_input(
    roi_name:str,
    roi:str,
    startdate:str,
    enddate:str,
    bands:str,
    s2cloudless_chunksize:int,
    mosaic_days:int,
    cloud_threshold:float,
    njobs:int,
    if_missing_files:str, # options: ['raise_error', 'warn', None]
    overwrite:bool,
):
    """
    --datacube-catalog and --log-file gets passed right before calling
    """
    cli = [
        'python _create_s2l1c_datacube.py',
        roi_name,
        roi,
        startdate,
        enddate,
        bands,
        f'--njobs {njobs}',
    ]

    if s2cloudless_chunksize > 0:
        cli.append(f'--s2cloudless {s2cloudless_chunksize}')
    
    if mosaic_days > 0:
        cli.append(f'--mosaic {mosaic_days}')
    
    if cloud_threshold is not None:
        cli.append(f'--cloud-threshold {cloud_threshold}')
    
    if if_missing_files == 'warn':
        cli.append('--warn-missing-files')
    elif if_missing_files is None:
        cli.append('--ignore-missing-files')

    if overwrite:
        cli.append('--overwrite')

    return ' '.join(cli)
    

def generate_cli_inputs(
    parameters_df:pd.DataFrame,
    njobs:int,
    if_missing_files:str, # options: ['raise_error', 'warn', None]
    overwrite:bool,
):
    cli_list = []
    for index, row in parameters_df.iterrows():
        cli_list.append(
            generate_cli_input(
                roi_name = row[COL_ROI_NAME],
                roi = row[COL_ROI],
                startdate = row[COL_STARTDATE],
                enddate = row[COL_ENDDATE],
                bands = row[COL_BANDS],
                s2cloudless_chunksize = row[COL_S2CLOUDLESS_CHUNKSIZE],
                mosaic_days = row[COL_MOSAIC_DAYS],
                cloud_threshold = row[COL_CLOUD_THRESHOLD],
                njobs = njobs,
                if_missing_files = if_missing_files,
                overwrite = overwrite,
            )
        )
    return cli_list


def check_for_new_configs_and_log(
    parameters_df:pd.DataFrame,
    add_new_config:bool,
    configs_filepath:str,
):  
    if_new_config = None if add_new_config else 'raise_error'
    
    for index, row in parameters_df.iterrows():
        if row[COL_BANDS] == 'all':
            bands = [
                'B01', 'B02', 'B03', 'B04',
                'B05', 'B06', 'B07', 'B08',
                'B8A', 'B09', 'B10', 'B11',
                'B12',
            ]
        else:
            bands = row[COL_BANDS].upper().split(',')

        _config = create_s2l1c_datacube.Configuration(
            cloud_threshold = row[COL_CLOUD_THRESHOLD],
            mosaic_days = row[COL_MOSAIC_DAYS],
            bands = bands,
        )
        is_new_config = create_s2l1c_datacube.new_config_action(
            config = _config,
            configs_filepath = configs_filepath,
            if_new_config = if_new_config,
        )
        # we reach here when a user has decided not to raise_error
        config_id = create_s2l1c_datacube.log_and_get_config_id(
            config = _config,
            configs_filepath = configs_filepath,
        )


def create_datacube_catalog_copies_and_log_filepaths(
    n:int, 
    main_datacube_catalog_filepath:str,
    log_folderpath:str,
    catalog_folderpath:str = None,
):
    if catalog_folderpath is not None:
        os.makedirs(catalog_folderpath, exist_ok=True)
    
    os.makedirs(log_folderpath, exist_ok=True)
    
    datacube_catalog_copies_filepaths = []
    log_filepaths = []

    for _ in range(n):
        suffix = '_' + rsutils.utils.get_epochs_str()
        
        _dc_copy_filepath = rsutils.utils.modify_filepath(
            filepath = main_datacube_catalog_filepath,
            suffix = suffix,
            new_folderpath = catalog_folderpath,
        )
        _log_filepath = os.path.join(log_folderpath, f'logs{suffix}.json')

        datacube_catalog_copies_filepaths.append(_dc_copy_filepath)
        log_filepaths.append(_log_filepath)

        if os.path.exists(main_datacube_catalog_filepath):
            shutil.copyfile(main_datacube_catalog_filepath, _dc_copy_filepath)
        
    return datacube_catalog_copies_filepaths, log_filepaths


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog = 'python create_s2l1c_datacubes.py',
        description = (
            'Script to create a datacubes of S2L1C data for list of geometry '
            'or s2grid ids, dateranges, and cloud masked median mosaicing parameters. '
            'A catalog file is updated at the end of the script.'
        ),
        epilog = f"--- Send your complaints to {','.join(config.MAINTAINERS)} ---",
    )

    DEFAULT_MAX_COMPUTE = min(mp.cpu_count() - 2, 16)
    DEFAULT_NJOBS = max(1, min(4, DEFAULT_MAX_COMPUTE // 2))

    WILL_BE_OVERRIDDEN = 'WILL BE OVERRIDDEN BY THE PARAMETER IN THE INPUT FILE.'

    parser.add_argument('filepath', help='path to the file containing parameters to create datacube.')
    parser.add_argument('--geojson', metavar='GEOM_DUMP_FOLDERPATH', action='store', required=False, help='To inform that the filepath passed is to a geojson, and folderpath where the geometries are to be dumped.')
    parser.add_argument('--csv', action='store_true', help='To inform that the filepath passed is to a csv, thus containing only list of s2grid ids for ROIs.')
    
    parser.add_argument('--startdate', metavar='YYYY-MM-DD', action='store', required=False, help=f'startdate for datacube. {WILL_BE_OVERRIDDEN}')
    parser.add_argument('--enddate', metavar='YYYY-MM-DD', action='store', required=False, help=f'enddate for datacube. {WILL_BE_OVERRIDDEN}')
    parser.add_argument('--bands', action='store', required=False, default='all', help=f'all, or bands comma separated B02,B03,B04,B08. {WILL_BE_OVERRIDDEN}')

    parser.add_argument('--njobs', action='store', default=DEFAULT_NJOBS, required=False, help=f'[default = {DEFAULT_NJOBS}] Number of cores to be allocated per instance.')
    parser.add_argument('--max-compute', action='store', default=DEFAULT_MAX_COMPUTE, required=False, help=f'[default = {DEFAULT_MAX_COMPUTE}] Max number of cores to be used for the whole execution.')
    
    parser.add_argument('-s2c', '--s2cloudless', metavar='CHUNKSIZE', default=0, action='store', required=False, help=f'Whether to run s2cloudless and if so at what chunk size. {WILL_BE_OVERRIDDEN}')
    parser.add_argument('-m', '--mosaic', action='store', required=False, default=0, help=f'Whether to perform median mosaicing and if so in intervals of how many days (most used interval: 20). {WILL_BE_OVERRIDDEN}')
    parser.add_argument('-c', '--cloud-threshold', action='store', required=False, help=f'The probabiliy threshold at and above which the pixel is considered cloud. Must be from 0 to 1. {WILL_BE_OVERRIDDEN}')
    
    parser.add_argument('--ignore-missing-files', action='store_true', help=f'If there are missing files for requested region and date range, this option ignores the error and proceeds, except when there are no files present. {WILL_BE_OVERRIDDEN}')
    parser.add_argument('--warn-missing-files', action='store_true', help=f'If there are missing files for requested region and date range, this option raises a warning and proceeds, except when there are no files present. {WILL_BE_OVERRIDDEN}')
    parser.add_argument('--overwrite', action='store_true', help=f'Overwrite existing datacube. {WILL_BE_OVERRIDDEN}')
    parser.add_argument('--add-new-config', action='store_true', help=f'To allow addition of new config. {WILL_BE_OVERRIDDEN}')

    parser.add_argument('--log-folder', action='store', required=False, help='Folderpath where log files will be created.')
    parser.add_argument('--parameters-folder', action='store', required=False, help='Folderpath where parameter files will be created.')

    args = parser.parse_args()

    if args.geojson is not None and args.csv:
        raise ValueError('Both --geojson and --csv options cannot be used together. Use either one.')
    if args.geojson is None and not args.csv:
        raise ValueError('Either --geojson and --csv options should be used.')
    
    filetype = 'geojson' if args.geojson is not None else 'csv'

    if filetype == 'geojson':
        raise NotImplementedError(
            'At this current moment --geojson option has not been implemented. '
            'Please wait for future updates.'
        )
    
        parameters_table = gpd.read_file(args.filepath).to_crs(EPSG_4326)
        for col in [COL_ROI_NAME, COL_GEOMETRY]:
            if col not in parameters_table.columns:
                raise ValueError(f'when --geojson is used, geojson must have {col} column.')
            
    else: # csv
        parameters_table = pd.read_csv(args.filepath).drop_duplicates().reset_index(drop=True)
        if COL_S2GRIDID not in parameters_table.columns:
            raise ValueError(f'when --csv is used, csv must have {COL_S2GRIDID} column.')
    
    if COL_STARTDATE not in parameters_table.columns:
        if args.startdate is None:
            raise ValueError(f'use --startdate option if {COL_STARTDATE} column not present {filetype} file.')
        startdate = datetime.datetime.strptime(args.startdate, '%Y-%m-%d').strftime( '%Y-%m-%d')
        parameters_table[COL_STARTDATE] = startdate

    if COL_ENDDATE not in parameters_table.columns:
        if args.enddate is None:
            raise ValueError(f'use --enddate option if {COL_ENDDATE} column not present {filetype} file.')
        enddate = datetime.datetime.strptime(args.enddate, '%Y-%m-%d').strftime( '%Y-%m-%d')
        parameters_table[COL_ENDDATE] = enddate
    
    if COL_BANDS not in parameters_table.columns:
        parameters_table[COL_BANDS] = args.bands

    if COL_S2CLOUDLESS_CHUNKSIZE not in parameters_table.columns:
        parameters_table[COL_S2CLOUDLESS_CHUNKSIZE] = int(args.s2cloudless)

    if COL_MOSAIC_DAYS not in parameters_table.columns:
        parameters_table[COL_MOSAIC_DAYS] = int(args.mosaic)
    
    if COL_CLOUD_THRESHOLD not in parameters_table.columns:
        if args.cloud_threshold is not None:
            parameters_table[COL_CLOUD_THRESHOLD] = float(args.cloud_threshold)
        else:
            parameters_table[COL_CLOUD_THRESHOLD] = None

    log_folderpath = args.log_folder
    parameters_folderpath = args.parameters_folder
    
    if_missing_files = 'raise_error'
    if args.ignore_missing_files:
        if_missing_files = None
    if args.warn_missing_files:
        if_missing_files = 'warn'

    overwrite = args.overwrite
    add_new_config = args.add_new_config

    njobs = int(args.njobs)
    max_compute = int(args.max_compute)

    if njobs > max_compute:
        raise ValueError(f'njobs per instance cannot exceed max_compute.')
    if max_compute > mp.cpu_count():
        raise ValueError(f'max_compute cannot exceed cpu_count of the machine (cpu_count = {mp.cpu_count()}).')

    """
    Writing only the implementation for --csv option

    CSV currently has the following columns:
    - s2gridid
    - startdate
    - enddate
    - bands
    - s2cloudless_chunksize
    - mosaic_days
    - cloud_threshold

    And global settings:
    - njobs
    - max_compute
    - if_missing_files
    - overwrite
    - add_new_config

    Now to create list of CLI inputs ...
    
    Note: The CLI utilities to run in parallely the scripts via multithreading
    would become its own module used by more pipelines.
    """
    if filetype == 'geojson':
        """
        - write out geometries into the GEOM_DUMP_FOLDERPATH
        - remove geometry column and add roi column which contains entries
          like 'filepath=/path/to/shapefile'
        """
    else:
        parameters_df = parameters_table
        parameters_df[COL_ROI] = 's2gridid=' + parameters_df[COL_S2GRIDID]
        parameters_df = parameters_df.rename(columns={COL_S2GRIDID: COL_ROI_NAME})
    
    check_for_new_configs_and_log(
        parameters_df = parameters_df,
        add_new_config = add_new_config,
        configs_filepath = config.FILEPATH_S2L1C_DATACUBE_CONFIG_TRACKER,
    )

    cli_inputs = generate_cli_inputs(
        parameters_df = parameters_df,
        njobs = njobs,
        if_missing_files = if_missing_files,
        overwrite = overwrite,
    )

    successes = run_cli_list_parallel(
        cli_inputs = cli_inputs,
        max_compute = max_compute,
        njobs = njobs,
        log_folderpath = log_folderpath,
        main_datacube_catalog_filepath = config.FILEPATH_S2L1C_DATACUBE_CATALOG,
    )

    parameters_df[COL_SUCCESSFUL] = successes

    os.makedirs(parameters_folderpath, exist_ok=True)
    parameters_filepath = os.path.join(parameters_folderpath, 
                                       f'parameters_{rsutils.utils.get_epochs_str(length=0)}.csv')
    parameters_df.to_csv(parameters_filepath, index=False)

    print(f'success: {sum(successes)} / {len(successes)}')

