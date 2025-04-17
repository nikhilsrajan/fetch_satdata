import argparse
import pandas as pd
import geopandas as gpd
import os
import rasterio
import rasterio.warp
import numpy as np
import tqdm
import time
import shapely
from memory_profiler import profile
import gc

import config
from init_logger import get_logger
import register_configuration as rc

import sys
sys.path.append('..')

import cdseutils.sentinel2
import cdseutils.constants
import create_datacube
import datacube_ops
import rsutils.modify_images
import rsutils.utils


logger = get_logger()


def get_datacube_id(
    roi:str,
    startdate:pd.Timestamp,
    enddate:pd.Timestamp,
    config_id:int,
):
    return '_'.join([
        roi,
        startdate.strftime('%Y%m%dT%H%M%S'),
        enddate.strftime('%Y%m%dT%H%M%S'),
        str(config_id)
    ])


def fetch_tile_catalog(
    catalog_db_filepath:str,
    startdate:pd.Timestamp,
    enddate:pd.Timestamp,
    tileid:str,
):
    catalog_gdf = create_datacube.filter_catalog_db(
        catalog_db_filepath = catalog_db_filepath,
        table_name = config.S2L2A_TABLE,
        startdate = startdate,
        enddate = enddate,
    )

    catalog_gdf['tile_number'] = catalog_gdf['id'].apply(
        lambda x: cdseutils.sentinel2.sentinel2_id_parser(x)['tile_number_field']
    )

    tile_catalog_gdf = catalog_gdf[catalog_gdf['tile_number'] == tileid].copy()
    del catalog_gdf

    return tile_catalog_gdf.sort_values(by='timestamp').reset_index(drop=True)


def create_filepaths_dict(
    catalog_gdf:gpd.GeoDataFrame,
    bands:list[str],
    ext:str = '.jp2',
):
    timestamp_band_filepath_dict = dict()

    for index, row in catalog_gdf.iterrows():
        timestamp = row['timestamp']
        local_folderpath = row['local_folderpath']
        files = row['files'].split(',')
        timestamp_band_filepath_dict[timestamp] = {
            f.removesuffix(ext) : os.path.join(local_folderpath, f)
            for f in files if f.endswith(ext)
            and f.removesuffix(ext) in set(bands)
        }
        
    return timestamp_band_filepath_dict


def get_mosaiced_stack(
    timestamps:list[pd.Timestamp],
    start_index:int,
    end_index:int,
    timestamp_band_filepath_dict:dict,
    ref_meta:dict,
    scl_mask_classes:list[int],
):
    ...


@profile
def create_tile_s2l2a_datacube(
    tileid:str,
    export_folderpath:str,
    catalog_db_filepath:str,
    startdate:pd.Timestamp,
    enddate:pd.Timestamp,
    bands:list[str],
    mosaic_days:int,
    scl_mask_classes:list[int],
    print_messages:bool = True,
    logger = None,
    overwrite:bool = False,
    override_gap_days:int = None,
):
    REFERENCE_BAND = 'B08'

    tile_catalog_gdf = fetch_tile_catalog(
        catalog_db_filepath = catalog_db_filepath,
        startdate = startdate,
        enddate = enddate,
        tileid = tileid,
    )

    timestamp_band_filepath_dict = create_filepaths_dict(
        catalog_gdf = tile_catalog_gdf,
        bands = bands,
    )

    ids = tile_catalog_gdf['id'].tolist()
    timestamps = tile_catalog_gdf['timestamp'].tolist()
    tile_geometry = shapely.unary_union(tile_catalog_gdf['geometry'])
    tile_geometry_crs = tile_catalog_gdf.crs

    timedelta_days, timestamp_range = \
    create_datacube.compute_timestamp_stats(
        timestamps = timestamps
    )

    """
    TO DO: Check gap days and raise error if too much.
    """

    ts_index_ranges = datacube_ops.get_mosaic_ts_index_ranges(
        timestamps = timestamps,
        startdate = startdate,
        enddate = enddate,
        mosaic_days = mosaic_days,
    )

    reference_filepath = timestamp_band_filepath_dict[timestamps[0]][REFERENCE_BAND]

    with rasterio.open(reference_filepath) as ref:
        ref_meta = ref.meta.copy()

    ref_meta['nodata'] = 0
    ref_meta['driver'] = 'GTiff'
    ref_meta = rsutils.utils.driver_specific_meta_updates(meta = ref_meta)

    metadata = {
        'geotiff_metadata': ref_meta,
        'mosaic_index_intervals': ts_index_ranges,
        'previous_timestamps': timestamps,
        'timestamps': [
            timestamps[ts_index_range[0]] for ts_index_range in ts_index_ranges
        ],
        'ids': ids,
        'bands': bands,
        'data_shape_desc': ('timestamps', 'height', 'width', 'bands'),
        'geometry': {
            'shape': [tile_geometry],
            'crs': tile_geometry_crs,
        }
    }

    H = ref_meta['height']
    W = ref_meta['width']
    B = len(bands) - 1 # without SCL
    dtype = ref_meta['dtype']
    band_indices = {
        band: band_index 
        for band_index, band in enumerate(bands) 
        if band != cdseutils.constants.Bands.S2L2A.SCL
    }

    # pbar = tqdm.tqdm(total = (B+1)*len(timestamps))

    mosaiced_stack = np.zeros(shape=(len(ts_index_ranges), H, W, B), dtype=dtype)
    for m_i, ts_index_range in enumerate(ts_index_ranges):
        start_index, end_index = ts_index_range
        T = end_index-start_index+1
        
        to_mosaic_stack = np.zeros(shape=(T, H, W, B), dtype=dtype)
        
        SCL = np.zeros(shape=(T, H, W), dtype=dtype)
        
        for tm_i, ts_i in enumerate(list(range(start_index, end_index + 1))):
            timestamp = timestamps[ts_i]
            for band in bands:
                b_i = -1

                if band != cdseutils.constants.Bands.S2L2A.SCL:
                    b_i = band_indices[band]

                band_filepath = timestamp_band_filepath_dict[timestamp][band]

                with rasterio.open(band_filepath) as src:
                    if src.meta['crs'] != ref_meta['crs']:
                        raise ValueError('Mismatching CRS in the tiles')

                    data = src.read()

                    if src.meta['height'] != ref_meta['height'] or src.meta['width'] != ref_meta['width']:
                        resampled_data, _ = rsutils.modify_images.resample_by_ref_meta(
                            data = data, 
                            profile = src.meta.copy(),
                            ref_meta = ref_meta,
                            resampling = rasterio.warp.Resampling.nearest,
                        )
                        if band != cdseutils.constants.Bands.S2L2A.SCL:
                            to_mosaic_stack[tm_i, :, :, b_i] = resampled_data
                        else:
                            SCL[tm_i] = resampled_data[0]

                        del resampled_data

                    else:
                        if band != cdseutils.constants.Bands.S2L2A.SCL:
                            to_mosaic_stack[tm_i, :, :, b_i] = data
                        else:
                            SCL[tm_i] = data[0]
                    
                    del data
                
                gc.collect()
                # pbar.update()

        print('start mosaic')
        s_time = time.time()
        to_mosaic_stack[np.where(np.isin(SCL, scl_mask_classes))] = 0

        masked_to_mosaic_stack = np.ma.masked_values(to_mosaic_stack, 0, copy=False) # copy = False to avoid duplicate memory

        mosaiced_stack[m_i] = np.ma.median(masked_to_mosaic_stack, axis=0).data.astype(dtype)

        e_time = time.time()
        print(f't_elapsed: {e_time - s_time:.2f} s')

        del to_mosaic_stack, masked_to_mosaic_stack, SCL

        gc.collect()

    os.makedirs(export_folderpath, exist_ok=True)
    np.save(os.path.join(export_folderpath, 'datacube.npy'), mosaiced_stack)
    np.save(os.path.join(export_folderpath, 'metadata.pickle.npy'), metadata, allow_pickle=True)



if __name__ == '__main__':
    start_time = time.time()

    parser = argparse.ArgumentParser(
        prog = 'python create_s2l2a_datacube.py',
        description = (
            'Script to create tile sentinel-2-l2a datacube.'
        ),
        epilog = f"--- Send your complaints to {','.join(config.MAINTAINERS)} ---",
    )
    parser.add_argument('tileid', help="Name of the Sentinel-2 L2A tile for which datacube needs to be created.")
    parser.add_argument('startdate', help='style: 2024-08-20 or 2024-08-20T13:54:40.022Z')
    parser.add_argument('enddate', help='style: 2024-08-20 or 2024-08-20T13:54:40.022Z')
    parser.add_argument('config_id', help='Make sure to have configuration registered using the script register_configurations.py - the ID that is printed is what needs to be passed.')
    parser.add_argument('export_folderpath', help='Folderpath where the datacube would be saved to.')
    # parser.add_argument('-j', '--njobs', required=False, action='store', default=1, help='[default = 1] Number of parallel jobs')
    parser.add_argument('-v', '--verbose', required=False, action='store_true')
    parser.add_argument('--overwrite', required=False, action='store_true')
    # parser.add_argument('--record', required=False, action='store_true', help='Whether to record the creation of the datacube to datacube catalog')
    parser.add_argument('--override-gap-days', default=None, action='store', required=False, help='Override the permitted time gap.')

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel('INFO')

    tileid = str(args.tileid)
    roi = f'tileid={tileid}'
    startdate = pd.to_datetime(str(args.startdate))
    enddate = pd.to_datetime(str(args.enddate))
    config_id = int(args.config_id)
    export_folderpath = str(args.export_folderpath)

    override_gap_days = None
    if args.override_gap_days is not None:
        override_gap_days = int(args.override_gap_days)

    config_desc = rc.fetch_s2l2a_config(config_id = config_id)[config_id]

    mosaic_days = config_desc[rc.COL_MOSAIC_DAYS]
    scl_masks = rc.str_to_list_int(config_desc[rc.COL_SCL_MASKS])

    create_tile_s2l2a_datacube(
        tileid = tileid,
        export_folderpath = export_folderpath,
        catalog_db_filepath = config.FILEPATH_SATELLITE_CATALOG_DB,
        startdate = startdate,
        enddate = enddate,
        bands = cdseutils.constants.Bands.S2L2A.ALL ,
        mosaic_days = mosaic_days,
        scl_mask_classes = scl_masks,
        print_messages = False,
        logger = logger,
        overwrite = args.overwrite,
        override_gap_days = override_gap_days,
    )

    end_time = time.time()

    print(f't_elapsed: {end_time - start_time:.2f}s')