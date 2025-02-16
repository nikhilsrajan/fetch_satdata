import os
import argparse
import pandas as pd
import tqdm
import multiprocessing as mp
import functools

import create_s2l2a_datacube as cs2l2ad

import config

import sys
sys.path.append('..')

import cdseutils.constants
import create_datacube
import sqlite_db_utils


def get_roi_stats(
    roi:str,
    startdate:pd.Timestamp,
    enddate:pd.Timestamp,
):
    MAX_TIMEDELTA_DAYS = 5 # hard-coded for sentinel-2
    EXT = '.jp2'

    shapes_gdf = cs2l2ad.get_shapes_gdf(roi=roi)
    query_stats, missing_flags, msg = \
    create_datacube.check_if_there_are_files_missing(
        catalog_db_filepath = config.FILEPATH_SATELLITE_CATALOG_DB,
        table_name = config.S2L2A_TABLE,
        satellite = cdseutils.constants.Bands.S2L2A.NAME,
        shapes_gdf = shapes_gdf,
        startdate = startdate,
        enddate = enddate,
        files = [f'{band}{EXT}' for band in cdseutils.constants.Bands.S2L2A.ALL],
        max_timedelta_days = MAX_TIMEDELTA_DAYS,
    )

    actual_startdate = None
    actual_enddate = None
    if query_stats['timestamp_range'] is not None:
        actual_startdate = query_stats['timestamp_range'][0]
        actual_enddate = query_stats['timestamp_range'][1]

    return actual_startdate, actual_enddate, missing_flags, msg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog = 'python generate_s2l2a_datacube_input_csv.py',
        description = (
            'Script to create input csv for create_s2l2a_datacube.py snakemake run.'
        ),
        epilog = f"--- Send your complaints to {','.join(config.MAINTAINERS)} ---",
    )
    parser.add_argument('rois_filepath', help="Text file containing ROI names. If ROI is an s2grid then pass 's2grid=GRID_ID'. Otherwise have the shape registered using register_polygon.py and pass 'geom=GEOM_ID'.")
    parser.add_argument('startdate', help='style: 2024-08-20 or 2024-08-20T13:54:40.022Z')
    parser.add_argument('enddate', help='style: 2024-08-20 or 2024-08-20T13:54:40.022Z')
    parser.add_argument('config_id', help='Make sure to have configuration registered using the script register_configurations.py - the ID that is printed is what needs to be passed.')
    parser.add_argument('out_csv_filepath', help='CSV export filepath.')
    parser.add_argument('--overwrite', help='If CSV file already exists, overwrite to it, else append.')
    parser.add_argument('-j', '--njobs', required=False, action='store', default=1, help='[default = 1] Number of parallel jobs')

    args = parser.parse_args()

    with open(args.rois_filepath) as f:
        rois = f.readlines()
    rois = [x[:-1] for x in rois]

    startdate = pd.to_datetime(str(args.startdate))
    enddate = pd.to_datetime(str(args.enddate))
    config_id = int(args.config_id)
    out_csv_filepath = str(args.out_csv_filepath)
    njobs = int(args.njobs)

    data = {
        'roi': [],
        'startdate': [],
        'enddate': [],
        'config_id': [],
        'missing_all': [],
        'missing_area': [],
        'missing_time': [],
        'missing_files': [],
        'missing_comment': [],
    }

    get_roi_stats_partial = functools.partial(
        get_roi_stats,
        startdate = startdate,
        enddate = enddate,
    )

    with mp.Pool(njobs) as p:
        outputs = list(tqdm.tqdm(
            p.imap(get_roi_stats_partial, rois), total=len(rois)
        ))
        
    for i in range(len(rois)):
        roi = rois[i]
        actual_startdate, actual_enddate, missing_flags, msg = outputs[i]
        if actual_startdate is not None:
            actual_startdate = sqlite_db_utils.ts_to_str(actual_startdate)
        if actual_enddate is not None:
            actual_enddate = sqlite_db_utils.ts_to_str(actual_enddate)
        data['roi'].append(roi)
        data['startdate'].append(actual_startdate)
        data['enddate'].append(actual_enddate)
        data['config_id'].append(config_id)
        data['missing_all'].append(missing_flags['all'])
        data['missing_area'].append(missing_flags['area'])
        data['missing_time'].append(missing_flags['time'])
        data['missing_files'].append(missing_flags['files'])
        data['missing_comment'].append(msg)

    input_df = pd.DataFrame(data=data)
    input_df['added_on'] = sqlite_db_utils.ts_to_str(pd.Timestamp.now(tz='UTC'))

    if os.path.exists(out_csv_filepath):
        existing_df = pd.read_csv(out_csv_filepath)
        input_df = pd.concat([existing_df, input_df])
    
    input_df.to_csv(out_csv_filepath, index=False)
