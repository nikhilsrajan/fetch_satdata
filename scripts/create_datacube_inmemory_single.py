import geopandas as gpd
import datetime
import logging
import shapely
import numpy as np
import pandas as pd
import warnings
import rasterio.warp
import argparse

from init_logger import get_logger

import os
CUR_DIR = os.path.dirname(__file__)

import sys
sys.path.append(os.path.join(CUR_DIR, '..'))

import exceptions
import cdseutils.constants
import rsutils.modify_images
import datacube_ops

import config

import time


class Timer:
    def __init__(self, print_messages:bool=True, logger=None):
        self.record_start_time = dict()
        self.record_elapsed_time = dict()
        self.print_messages = print_messages

    
    def start(self, key:str):
        if key not in self.record_start_time.keys():
            if self.print_messages:
                logger.info(f'[START] {key}')
            self.record_start_time[key] = time.time()
    

    def stop(self, key:str):
        end_time = time.time()
 
        if key not in self.record_start_time.keys():
            raise KeyError(f'{key} was never recorded.')
        
        start_time = self.record_start_time[key]
        del self.record_start_time[key]

        elapsed_time = end_time - start_time

        if key not in self.record_elapsed_time.keys():
            self.record_elapsed_time[key] = 0

        self.record_elapsed_time[key] += elapsed_time

        if self.print_messages:
            logger.info(f'[STOP] {key} : t_elapsed = {round(elapsed_time, 2)} secconds.')
    

    def summary(self):
        logger.info('SUMMARY')
        logger.info('=======')
        for key, elapsed_time in self.record_elapsed_time.items():
            logger.info(f'{key} : {round(elapsed_time, 2)} secconds.')


logger = get_logger()
timer = Timer(print_messages=False)


def dt2ts(
    dt:datetime.datetime, 
    tz='UTC',
):
    if dt.tzinfo is None:
        return pd.Timestamp(dt, tz=tz)
    
    return pd.Timestamp(dt)


def compute_timestamp_stats(
    timestamps:list,
):
    timestamps = np.array(timestamps)
    timestamps.sort()
    timedeltas_round = [td.round('d').days for td in (timestamps[1:] - timestamps[:-1])]
    timedelta_days = dict(zip(*np.unique(timedeltas_round, return_counts=True)))
    timestamp_range = (timestamps.min(), timestamps.max())
    return timedelta_days, timestamp_range


def query_catalog_stats(
    catalog_gdf:gpd.GeoDataFrame,
    shapes_gdf:gpd.GeoDataFrame,
):
    stats = {
        k: None for k in [
            'tile_count',
            'area_coverage',
            'timedelta_days',
            'timestamp_range',
            'file_counts',
        ]
    }

    filtered_catalog_gdf = catalog_gdf

    shapes_gdf = shapes_gdf.to_crs(filtered_catalog_gdf.crs)

    if filtered_catalog_gdf.shape[0] == 0:
        stats['tile_count'] = 0
        return stats

    target_shape = shapely.unary_union(shapes_gdf['geometry'])
    queried_shape = shapely.unary_union(filtered_catalog_gdf['geometry'])

    stats['area_coverage'] = 1 - (target_shape - queried_shape).area / target_shape.area
    stats['area_coverage'] = round(stats['area_coverage'], 4)

    stats['timedelta_days'], stats['timestamp_range'] = compute_timestamp_stats(
        timestamps = filtered_catalog_gdf['timestamp'].tolist()
    )

    stats['band_counts'] = filtered_catalog_gdf['band'].value_counts().to_dict()

    stats['tile_count'] = max(stats['band_counts'].values())

    return stats


def check_if_there_are_files_missing(
    shapes_gdf:gpd.GeoDataFrame,
    catalog_gdf:gpd.GeoDataFrame,
    startdate:datetime.datetime,
    enddate:datetime.datetime,
    bands:list[str],
    max_timedelta_days:int = 5,
):
    stats = query_catalog_stats(
        catalog_gdf = catalog_gdf,
        shapes_gdf = shapes_gdf,
    )

    missing_flags = {
        'all': False,
        'area': False,
        'time': False,
        'bands': False,
    }

    msgs = []

    if stats['tile_count'] == 0:
        for k in missing_flags.keys():
            missing_flags[k] = True
        msgs.append('No tiles found.')
    else:
        if stats['area_coverage'] < 1:
            missing_flags['area'] = True
            msgs.append(f"Incompelete area coverage: {round(stats['area_coverage'] * 100, 2)}%")

        td_msg = None
        for td in stats['timedelta_days'].keys():
            if td > max_timedelta_days:
                if td_msg is None:
                    td_msg = 'Unusual time gaps found (days):'
                td_msg += f' {td},'
                missing_flags['time'] = True
        if td_msg is not None:
            td_msg = td_msg[:-1] # removing the last comma
            msgs.append(td_msg)

        first_image_gap_days = (stats['timestamp_range'][0] - dt2ts(startdate)).days
        last_image_gap_days = (dt2ts(enddate) - stats['timestamp_range'][1]).days
        if first_image_gap_days > max_timedelta_days:
            missing_flags['time'] = True
            msgs.append(f'First available image is {first_image_gap_days} days from startdate')
        if last_image_gap_days > max_timedelta_days:
            missing_flags['time'] = True
            msgs.append(f'Last available image is {last_image_gap_days} days from enddate')
        
        completely_missing_bands = []
        partially_missing_bands = []
        for band in bands:
            if band not in stats['band_counts'].keys():
                missing_flags['bands'] = True
                completely_missing_bands.append(band)
            elif stats['band_counts'][band] < stats['tile_count']:
                missing_flags['bands'] = True
                partially_missing_bands.append(band)
        if len(completely_missing_bands) > 0:
            msgs.append(f'Completely missing bands: {completely_missing_bands}')
        if len(partially_missing_bands) > 0:
            msgs.append(f'Partially missing bands: {partially_missing_bands}')

    msg = '; '.join(msgs)

    return stats, missing_flags, msg


def missing_files_action(
    catalog_gdf:gpd.GeoDataFrame,
    shapes_gdf:gpd.GeoDataFrame,
    startdate:datetime.datetime,
    enddate:datetime.datetime,
    bands:list[str],
    if_missing_files = 'raise_error',
    max_timedelta_days:int = 5,
):
    VALID_IF_MISSING_FILES_OPTIONS = ['raise_error', 'warn', None]
    if not any([if_missing_files is x for x in VALID_IF_MISSING_FILES_OPTIONS]):
        raise exceptions.DatacubeException(
            f'Invalid if_missing_files={if_missing_files}. '
            f'if_missing_files must be from {VALID_IF_MISSING_FILES_OPTIONS}'
        )

    query_stats, missing_flags, msg = \
    check_if_there_are_files_missing(
        catalog_gdf = catalog_gdf,
        shapes_gdf = shapes_gdf,
        startdate = startdate,
        enddate = enddate,
        bands = bands,
        max_timedelta_days = max_timedelta_days,
    )
    # If there are no files present raise error no matter.
    if missing_flags['all']:
        raise exceptions.DatacubeException('Missing files error -- ' + msg)
    if any(missing_flags.values()):
        if if_missing_files == 'raise_error':
            raise exceptions.DatacubeException('Missing files error -- ' + msg)
        elif if_missing_files == 'warn':
            warnings.warn(message = 'Missing files warning\n' + msg, 
                          category = RuntimeWarning)
    
    return query_stats, missing_flags


def get_unary_gdf(
    shapes_gdf:gpd.GeoDataFrame,
    crs,
):
    union_shape = shapely.unary_union(
        shapes_gdf.to_crs(crs)['geometry']
    )
    union_shape_gdf = gpd.GeoDataFrame(
        data = {'geometry': [union_shape]},
        crs = crs,
    )
    return union_shape_gdf


def compute_area_contribution(
    catalog_gdf:gpd.GeoDataFrame,
    shape_gdf:gpd.GeoDataFrame,
):
    union_shape_gdf = get_unary_gdf(shapes_gdf=shape_gdf, crs=catalog_gdf.crs)
    union_shape = union_shape_gdf['geometry'][0]

    filtered_catalog_gdf = \
    gpd.overlay(
        catalog_gdf,
        union_shape_gdf,
    )

    return filtered_catalog_gdf['geometry'].apply(
        lambda x: x.area / union_shape.area * 100 # area contribution in terms of %
    )


def load_images(
    catalog_gdf:gpd.GeoDataFrame,
    shape_gdf:gpd.GeoDataFrame,
    nodata,
    njobs:int = 1,
    print_messages:bool = True,
):
    data_profile_list = rsutils.modify_images.load_images(
        src_filepaths = catalog_gdf['filepath'],
        shapes_gdf = shape_gdf,
        raise_error = False,
        nodata = nodata,
        all_touched = True,
        njobs = njobs,
        print_messages = print_messages,
    )

    data_profile_index = np.array([
        index 
        if data_profile[0] is not None 
        else -1 
        for index, data_profile in enumerate(data_profile_list)
    ])

    if np.all(data_profile_index == -1):
        raise ValueError('No valid images found.')

    catalog_gdf['image_index'] = data_profile_index
    catalog_gdf['crs'] = [
        str(profile['crs'])
        if profile is not None
        else None
        for _, profile in data_profile_list
    ]

    catalog_gdf = catalog_gdf[catalog_gdf['image_index'] != -1]

    return catalog_gdf, data_profile_list


def get_dst_crs(catalog_gdf:gpd.GeoDataFrame):
    group_by_crs_df = catalog_gdf.groupby(by='crs')
    area_contribution_mean = group_by_crs_df['area_contribution'].mean()
    sorted_area_contribution_mean = area_contribution_mean.sort_values(ascending=False)
    max_area_contribution_crs = sorted_area_contribution_mean.index[0]
    dst_crs = max_area_contribution_crs

    return dst_crs


def select_items(
    indices:list,
    items:list,
):
    return [items[i] for i in indices]


def get_merged_profile(
    indices:list,
    data_profile_list:list,
    dst_crs:str,
    nodata = None,
    njobs:int = 1,
    print_messages:bool = True,
):
    selected_data_profile_list = select_items(indices=indices, items=data_profile_list)

    # get indices with different crs
    different_crs_indices = [
        index for index, data_profile in enumerate(selected_data_profile_list)
        if data_profile[1]['crs'] != dst_crs
    ]
    
    # reproject to dst_crs
    if len(different_crs_indices) > 0:
        diff_crs_data_profile_list = select_items(indices=different_crs_indices, items=selected_data_profile_list)
        reproj_diff_crs_data_profile_list = rsutils.modify_images.modify_images_inplace(
            data_profile_list = diff_crs_data_profile_list,
            sequence = [
                (rsutils.modify_images.reproject, dict(dst_crs=dst_crs))
            ],
            njobs = njobs,
            print_messages = print_messages,
        )
        for index, reproj_diff_crs_data_profile in zip(different_crs_indices, reproj_diff_crs_data_profile_list):
            selected_data_profile_list[index] = reproj_diff_crs_data_profile
    
    # merge the reference band images in dst_crs
    merged_data, merged_profile = rsutils.modify_images.merge_inplace(
        data_profile_list = selected_data_profile_list,
        nodata = nodata,
    )

    return merged_profile


def get_indices_to_resample(
    indices:list,
    data_profile_list:list,
    reference_profile:dict,
):
    resample_indices = []
    for image_index in indices:
        _, profile = data_profile_list[image_index]
        if profile['crs'] != reference_profile['crs'] or \
            profile['height'] != reference_profile['height'] or \
            profile['width'] != reference_profile['width']:
            resample_indices.append(image_index)
    return resample_indices


def resample_by_indices(
    indices:list,
    data_profile_list:list,
    reference_profile:dict,
    njobs:int = 1,
    print_messages:bool = True,
    resampling = rasterio.warp.Resampling.nearest, 
):
    data_profile_to_resample = [data_profile_list[i] for i in indices]

    data_profile_resampled = rsutils.modify_images.modify_images_inplace(
        data_profile_list = data_profile_to_resample,
        sequence = [
            (rsutils.modify_images.resample_by_ref_meta, dict(ref_meta=reference_profile, resampling=resampling)),
        ],
        njobs = njobs,
        print_messages = print_messages,
    )

    for index, data_profile_resampled_i in zip(indices, data_profile_resampled):
        data_profile_list[index] = data_profile_resampled_i
    
    return data_profile_list


def create_datacube(
    catalog_gdf:gpd.GeoDataFrame, # cols: filepaths, band, timestamp, geometry
    shape_gdf:gpd.GeoDataFrame,
    startdate:datetime,
    enddate:datetime,
    mosaic_days:int,
    scl_mask_classes:list[int],
    bands:list[str],
    reference_band:str,
    export_folderpath:str,
    logger:logging.Logger = None,
    print_messages:bool = False,
    if_missing_files = 'raise_error',
    max_timedelta_days:int = 5,
    njobs:int = 1,
    njobs_load_images:int = 1,
):
    NODATA = 0 # since the script is hardcoded for sentinel-2-l2a
    EXT = '.jp2' # since the script is hardcoded for sentinel-2-l2a
    MAX_TIMEDELTA_DAYS = 5 # since the script is hardcoded for sentinel-2-l2a

    timer.start('missing_files_action')
    
    missing_files_action(
        catalog_gdf = catalog_gdf,
        shapes_gdf = shape_gdf,
        startdate = startdate,
        enddate = enddate,
        if_missing_files = if_missing_files,
        max_timedelta_days = max_timedelta_days,
        bands = bands,
    )

    timer.stop('missing_files_action')

    timestamps = catalog_gdf['timestamp'].unique().to_numpy()
    timestamps.sort()
    
    timer.start('load_images')
    logger.info("Loading images")
    # adds crs, image_index, removes invalid rows
    catalog_gdf, data_profile_list = load_images(
        catalog_gdf = catalog_gdf,
        shape_gdf = shape_gdf,
        nodata = NODATA,
        njobs = njobs_load_images,
        print_messages = print_messages,
    )

    timer.stop('load_images')

    timer.start('get_dst_crs')
    # needs crs and area_contribution
    dst_crs = get_dst_crs(catalog_gdf = catalog_gdf)
    
    # print('dst_crs =', dst_crs)
    timer.stop('get_dst_crs')

    timer.start('generate_reference_profile')
    
    # generate reference profile = merged profile

    ref_band_indices = catalog_gdf[catalog_gdf['band'] == reference_band]['image_index']

    logger.info("Generating reference profile")
    reference_profile = get_merged_profile(
        indices = ref_band_indices,
        data_profile_list = data_profile_list,
        dst_crs = dst_crs,
        nodata = NODATA,
        print_messages = print_messages,
    )

    timer.stop('generate_reference_profile')

    # print(f'set(shapes) = {set([data.shape for data, _ in data_profile_list])}')
    # print(f"set(crs) = {set([profile['crs'] for _, profile in data_profile_list])}")

    logger.info("Resampling to reference profile")

    timer.start('get_indices_to_resample')
    resample_indices = get_indices_to_resample(
        indices = catalog_gdf['image_index'], 
        data_profile_list = data_profile_list, 
        reference_profile = reference_profile,
    )
    # print(f'len(resample_indices) = {len(resample_indices)}')
    timer.stop('get_indices_to_resample')

    timer.start('resample_by_indices')
    data_profile_list = resample_by_indices(
        indices = resample_indices,
        data_profile_list = data_profile_list,
        reference_profile = reference_profile,
        njobs = njobs,
        print_messages = print_messages,
    )
    timer.stop('resample_by_indices')
    # print(f'set(shapes) = {set([data.shape for data, _ in data_profile_list])}')
    # print(f"set(crs) = {set([profile['crs'] for _, profile in data_profile_list])}")

    logger.info("Stacking images into datacube")
    timer.start('create_datacube')
    # create datacube
    timestamps = catalog_gdf['timestamp'].unique().tolist()
    timestamps.sort()
    ts_band_index_dict = dict(zip(zip(catalog_gdf['timestamp'], catalog_gdf['band']), catalog_gdf['image_index']))
    ts_id_dict = dict(zip(catalog_gdf['timestamp'], catalog_gdf['id']))

    datacube = []
    ids = []
    for ts in timestamps:
        bands_stack = []
        for band in bands:
            if (ts, band) not in ts_band_index_dict.keys():
                band_data = np.full(shape=(reference_profile['height'], reference_profile['width']), fill_value=NODATA)
            else:
                index = ts_band_index_dict[(ts, band)]
                band_data = data_profile_list[index][0]
            bands_stack.append(band_data)
        bands_stack = np.stack(bands_stack, -1)
        datacube.append(bands_stack)
        ids.append(ts_id_dict[ts])
    datacube = np.concatenate(datacube, axis=0)

    metadata = {
        'geotiff_metadata': reference_profile,
        'timestamps': timestamps,
        'ids': ids,
        'bands': bands,
        'data_shape_desc': ('timestamps|ids', 'height', 'width', 'bands'),
        'geometry': {
            'shape': shape_gdf['geometry'].to_list(),
            'crs': shape_gdf.crs,
        }
    }

    timer.stop('create_datacube')

    # print('datacube.shape =', datacube.shape)

    logger.info("Cloud masking and median mosaicing the datacube")

    timer.start('cloud mask + median mosaic')

    datacube, metadata = datacube_ops.run_datacube_ops(
        datacube = datacube,
        metadata = metadata,
        sequence = [
            (datacube_ops.apply_cloud_mask_scl, dict(mask_classes = scl_mask_classes)),
            (datacube_ops.drop_bands, dict(bands_to_drop = ['SCL'])),
            (datacube_ops.median_mosaic, dict(startdate = startdate,
                                              enddate = enddate,
                                              mosaic_days = mosaic_days,))
        ],
        logger = logger,
    )

    # print('datacube.shape =', datacube.shape)

    timer.stop('cloud mask + median mosaic')

    logger.info("Exporting the datacube")
    timer.start('save datacube')
    os.makedirs(export_folderpath, exist_ok=True)
    np.save(os.path.join(export_folderpath, 'datacube.npy'), datacube)
    np.save(os.path.join(export_folderpath, 'metadata.pickle.npy'), metadata, allow_pickle=True)
    timer.stop('save datacube')


# helper

def flatten_catalog_gdf(
    catalog_gdf:gpd.GeoDataFrame,
):
    data = {
        'id': [],
        'filepath': [],
        'band': [],
        'timestamp': [],
        'geometry': [],
        'area_contribution': [],
    }

    for index, row in catalog_gdf.iterrows():
        tile_id = row['id']
        folderpath = row['local_folderpath']
        timestamp = row['timestamp']
        geometry = row['geometry']
        area_contribution = row['area_contribution']

        for file in row['files'].split(','):
            if not file.endswith('.jp2'):
                continue
            band = file.removesuffix('.jp2')
            
            data['id'].append(tile_id)
            data['filepath'].append(os.path.join(folderpath, file))
            data['band'].append(band)
            data['timestamp'].append(timestamp)
            data['geometry'].append(geometry)
            data['area_contribution'].append(area_contribution)

    flattened_catalog_gdf = gpd.GeoDataFrame(data = data, crs = catalog_gdf.crs)

    return flattened_catalog_gdf


def calculate_area_contribution(
    catalog_gdf:gpd.GeoDataFrame,
    shapes_gdf:gpd.GeoDataFrame,
):
    union_shape_gdf = get_unary_gdf(shapes_gdf=shapes_gdf, crs=catalog_gdf.crs)
    union_shape = union_shape_gdf['geometry'][0]

    filtered_catalog_gdf = \
    gpd.overlay(
        catalog_gdf,
        union_shape_gdf,
    )

    filtered_catalog_gdf['area_contribution'] = filtered_catalog_gdf['geometry'].apply(
        lambda x: x.area / union_shape.area * 100 # area contribution in terms of %
    )

    return filtered_catalog_gdf


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog = 'python create_datacube_inmemory_single.py',
        description = (
            "Script to create a single sentinel-2 L2A datacube inmemory. "
            "This script would be run with snakemake for parallelisation. "
            "Thus this script is to create a single datacube."
        ),
        epilog = f"--- Send your complaints to {','.join(config.MAINTAINERS)} ---",
    )

    parser.add_argument('shapefilepath', help="shapefile containing polygon of interest.")
    parser.add_argument('catalog_filepath', help="catalog filepath contains path to the satellite files relevant to createing the datacube.")
    parser.add_argument('startdate', help='style: 2024-08-20 or 2024-08-20T13:54:40.022Z')
    parser.add_argument('enddate', help='style: 2024-08-20 or 2024-08-20T13:54:40.022Z')
    parser.add_argument('export_folderpath', help='Folderpath where the datacube should be saved.')
    parser.add_argument('--mosaic-days', help='Mosaic interval in days. Recommended value = 20')
    parser.add_argument('--scl-mask-classes', help="Values to mask in the SCL layer. To be passed as a comma separated integers. Recommended: '0,1,3,7,8,9,10'")
    parser.add_argument('--njobs', help="Number of parallel operations to execute for parallelised steps.", type=int)
    parser.add_argument('--njobs-load-images', help="Number of parallel operations for loading images.", type=int)    
    parser.add_argument('-v', '--verbose', required=False, action='store_true')
    # parser.add_argument('--stagger', required=False, action='store', type=float, default=None, help='Upper limit to randomly choose seconds to sleep to stagger the script when multiple run in parallel')

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel('INFO')

    catalog_gdf = gpd.read_file(args.catalog_filepath)
    shape_gdf = gpd.read_file(args.shapefilepath)
    startdate = pd.Timestamp(str(args.startdate), tz='UTC')
    enddate = pd.Timestamp(str(args.enddate), tz='UTC')
    export_folderpath = str(args.export_folderpath)
    mosaic_days = int(args.mosaic_days)
    scl_mask_classes = [int(v) for v in args.scl_mask_classes.split(',')]
    njobs = args.njobs
    njobs_load_images = args.njobs_load_images
        

    timer.start('process the catalog')
    catalog_gdf = calculate_area_contribution(catalog_gdf=catalog_gdf, shapes_gdf=shape_gdf)
    flattened_catalog_gdf = flatten_catalog_gdf(catalog_gdf=catalog_gdf)
    timer.stop('process the catalog')

    timer.start('all')

    MAX_TIMEDELTA_DAYS = 10

    create_datacube(
        catalog_gdf = flattened_catalog_gdf,
        shape_gdf = shape_gdf,
        startdate = startdate,
        enddate = enddate,
        mosaic_days = mosaic_days,
        scl_mask_classes = scl_mask_classes,
        bands = cdseutils.constants.Bands.S2L2A.ALL,
        reference_band = cdseutils.constants.Bands.S2L2A.B08,
        logger = logger,
        njobs = njobs,
        njobs_load_images = njobs_load_images,
        export_folderpath = export_folderpath,
        max_timedelta_days = MAX_TIMEDELTA_DAYS,
    )
    timer.stop('all')

    timer.summary()


"""
Notes:
=====
1.  No dst_crs to be passed. If a dst_crs is passed then two steps needs to be performed 
    1: change crs, 2: resample to highest resolution. This is an unnecessary complication 
    thus crs would be determined insitu by finding the crs with the highest area contribution.

2.  njobs_load_images is added as a parameter, in addition to having a global njobs. This
    is because load_images is a time consuming operation but also a very light weight operation.
    For heavy operations, which which affect the snakemake parallelisation, global njobs would
    be used.

3.  get_reference_profile is a multi-step process.
    1: Find images of reference band
    2: Reproject the reference band images whose crs doesn't match max area contribution crs
    3: Merge the reprojected images and generate the reference profile
    
"""