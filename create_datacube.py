import geopandas as gpd
import datetime
import rasterio.merge
import shapely.ops
import pandas as pd
import rasterio
import rasterio.warp
import os
import functools
import multiprocessing as mp
import tqdm
import numpy as np
import shutil
import warnings

import rsutils.modify_images
import rsutils.utils
import exceptions


FILENAME_DATACUBE = 'datacube.npy'
FILENAME_METADATA = 'metadata.pickle.npy'


def dt2ts(
    dt:datetime.datetime, 
    tz='UTC',
):
    if dt.tzinfo is None:
        return pd.Timestamp(dt, tz=tz)
    
    return pd.Timestamp(dt)


def get_unary_gdf(
    shapes_gdf:gpd.GeoDataFrame,
    crs,
):
    union_shape = shapely.ops.unary_union(
        shapes_gdf.to_crs(crs)['geometry']
    )
    union_shape_gdf = gpd.GeoDataFrame(
        data = {'geometry': [union_shape]},
        crs = crs,
    )
    return union_shape_gdf


def filter_catalog(
    catalog_filepath:str,
    satellite:str, # catalog can contain multiple satellites
    shapes_gdf:gpd.GeoDataFrame,
    startdate:datetime,
    enddate:datetime,
):
    """
    Assumes catalog has following columns:
    - timestamp
    - geometry
    """
    catalog_gdf = gpd.read_file(catalog_filepath)

    catalog_gdf = catalog_gdf[catalog_gdf['satellite'] == satellite]

    union_shape_gdf = get_unary_gdf(shapes_gdf=shapes_gdf, crs=catalog_gdf.crs)
    union_shape = union_shape_gdf['geometry'][0]

    dt_filtered_catalog_gdf = catalog_gdf[
        (catalog_gdf['timestamp'] >= dt2ts(startdate))
        & (catalog_gdf['timestamp'] <= dt2ts(enddate))
    ].reset_index(drop=True)

    filtered_catalog_gdf = \
    gpd.overlay(
        dt_filtered_catalog_gdf,
        union_shape_gdf,
    )

    filtered_catalog_gdf['area_contribution'] = filtered_catalog_gdf['geometry'].apply(
        lambda x: x.area / union_shape.area * 100 # area contribution in terms of %
    )

    return filtered_catalog_gdf


def query_catalog_stats(
    catalog_filepath:str,
    satellite:str, # catalog can contain multiple satellites
    shapes_gdf:gpd.GeoDataFrame,
    startdate:datetime,
    enddate:datetime,
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

    filtered_catalog_gdf = filter_catalog(
        catalog_filepath = catalog_filepath,
        satellite = satellite,
        shapes_gdf = shapes_gdf,
        startdate = startdate,
        enddate = enddate,
    )

    tile_count = filtered_catalog_gdf.shape[0]

    stats['tile_count'] = tile_count

    if tile_count == 0:
        return stats

    target_shape = shapely.ops.unary_union(shapes_gdf['geometry'])
    queried_shape = shapely.ops.unary_union(filtered_catalog_gdf['geometry'])

    stats['area_coverage'] = 1 - (target_shape - queried_shape).area / target_shape.area

    timestamps = filtered_catalog_gdf['timestamp'].to_numpy()
    timestamps.sort()
    timedeltas_round = [td.round('d').days for td in (timestamps[1:] - timestamps[:-1])]

    stats['timedelta_days'] = dict(zip(*np.unique(timedeltas_round, return_counts=True)))

    stats['timestamp_range'] = (timestamps.min(), timestamps.max())

    files = []
    for _files in filtered_catalog_gdf['files']:
        files += _files.split(',')

    stats['file_counts'] = dict(zip(*np.unique(files, return_counts=True)))

    return stats


def check_if_there_are_files_missing(
    shapes_gdf:gpd.GeoDataFrame,
    catalog_filepath:str,
    satellite:str, # catalog can contain multiple satellites
    startdate:datetime.datetime,
    enddate:datetime.datetime,
    files:list[str],
    max_timedelta_days:int = 5,
):
    stats = query_catalog_stats(
        catalog_filepath = catalog_filepath,
        satellite = satellite,
        shapes_gdf = shapes_gdf,
        startdate = startdate,
        enddate = enddate,
    )

    missing_flags = {
        'all': False,
        'area': False,
        'time': False,
        'files': False,
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
        
        completely_missing_files = []
        partially_missing_files = []
        for file in files:
            if file not in stats['file_counts'].keys():
                missing_flags['files'] = True
                completely_missing_files.append(file)
            elif stats['file_counts'][file] < stats['tile_count']:
                missing_flags['files'] = True
                partially_missing_files.append(file)
        if len(completely_missing_files) > 0:
            msgs.append(f'Completely missing files: {completely_missing_files}')
        if len(partially_missing_files) > 0:
            msgs.append(f'Partially missing files: {partially_missing_files}')

    msg = '; '.join(msgs)

    return stats, missing_flags, msg


def get_intersecting_band_filepaths(
    shapes_gdf:gpd.GeoDataFrame,
    catalog_filepath:str,
    satellite:str, # catalog can contain multiple satellites
    startdate:datetime,
    enddate:datetime,
    bands:list[str],
    ext:str = '.jp2',
):
    """
    Assumes the catalog has following columns:
    - id
    - timestamp
    - local_folderpath
    - files
    """
    catalog_gdf = filter_catalog(
        catalog_filepath = catalog_filepath,
        satellite = satellite,
        shapes_gdf = shapes_gdf,
        startdate = startdate,
        enddate = enddate,
    )

    data = {
        'id': [],
        'timestamp': [],
        'band': [],
        'filepath': [],
        'area_contribution': [],
    }

    for _id, timestamp, local_folderpath, \
        files, area_contribution in zip(
        catalog_gdf['id'],
        catalog_gdf['timestamp'],
        catalog_gdf['local_folderpath'],
        catalog_gdf['files'],
        catalog_gdf['area_contribution'],
    ):
        filenames_of_interest = {
            xii + ext 
            for xii in {
                xi.removesuffix(ext) 
                for xi in files.split(',')
            } & set(bands)
        }
        
        # if no files of interest present, skip
        if len(filenames_of_interest) == 0:
            continue

        _filepaths = [
            os.path.join(local_folderpath, filename_of_interest)
            for filename_of_interest in filenames_of_interest
        ]

        data['id'] += [_id for _ in range(len(_filepaths))]
        data['timestamp'] += [timestamp for _ in range(len(_filepaths))]
        data['area_contribution'] += [area_contribution for _ in range(len(_filepaths))]
        data['band'] += [
            filename_of_interest.removesuffix(ext)
            for filename_of_interest in filenames_of_interest
        ]
        data['filepath'] += _filepaths

        del _filepaths

    band_filepaths_df = pd.DataFrame(data=data).sort_values(
        by = ['timestamp', 'band'],
        ascending = [True, True],
    ).reset_index(drop=True)

    return band_filepaths_df


def get_image_crs(filepath:str):
    with rasterio.open(filepath) as src:
        crs = src.crs
    return crs


def change_parent_folderpath(
    filepath:str,
    parent_folderpath:str,
    new_parent_folderpath:str,
):
    filepath = os.path.abspath(filepath)
    parent_folderpath = os.path.abspath(parent_folderpath)
    return os.path.join(
        new_parent_folderpath,
        os.path.relpath(path=filepath,
                        start=parent_folderpath)
    )


def crop_and_reproject(
    shapes_gdf:gpd.GeoDataFrame,
    catalog_filepath:str,
    satellite:str, # catalog can contain multiple satellites
    startdate:datetime.datetime,
    enddate:datetime.datetime,
    bands:list[str],
    out_folderpath:str,
    satellite_folderpath:str=None, # for maintaining the same folder structure
    nodata = 0,
    working_dir:str = None,
    njobs:int = 8,
    dst_crs = None,
    print_messages:bool = True,
    ext:str = '.jp2',
):
    band_filepaths_df = \
    get_intersecting_band_filepaths(
        shapes_gdf = shapes_gdf,
        startdate = startdate,
        enddate = enddate,
        catalog_filepath = catalog_filepath,
        satellite = satellite,
        bands = bands,
        ext = ext,
    )

    if band_filepaths_df.shape[0] == 0:
        raise exceptions.DatacubeException(
            'get_intersecting_band_filepaths returned 0 results.'
        )
    
    band_filepaths_df['crs'] = band_filepaths_df['filepath'].apply(lambda x: str(get_image_crs(filepath=x)))

    if dst_crs is None:
        group_by_crs_df = band_filepaths_df.groupby(by='crs')
        area_contribution_mean = group_by_crs_df['area_contribution'].mean()
        sorted_area_contribution_mean = area_contribution_mean.sort_values(ascending=False)
        max_area_contribution_crs = sorted_area_contribution_mean.index[0]
        dst_crs = max_area_contribution_crs

    
    os.makedirs(out_folderpath, exist_ok=True)

    sequence = [
        (rsutils.modify_images.crop, dict(shapes_gdf=shapes_gdf, nodata=nodata, all_touched=True)),
        (rsutils.modify_images.reproject, dict(dst_crs=dst_crs)),
        (rsutils.modify_images.crop, dict(shapes_gdf=shapes_gdf, nodata=nodata, all_touched=True)),
    ]

    if satellite_folderpath is not None:
        satellite_folderpath = os.path.abspath(satellite_folderpath)
        band_filepaths_df['out_filepath'] = band_filepaths_df['filepath'].apply(
            lambda filepath: change_parent_folderpath(
                filepath = filepath,
                parent_folderpath = satellite_folderpath,
                new_parent_folderpath = out_folderpath,
            )
        )
    else:
        band_filepaths_df['out_filepath'] = band_filepaths_df.apply(
            lambda row: rsutils.utils.modify_filepath(
                filepath = row['filepath'],
                new_folderpath = os.path.join(out_folderpath, row['id'])
            ),
            axis = 1,
        )

    successes = rsutils.modify_images.modify_images(
        src_filepaths = band_filepaths_df['filepath'],
        dst_filepaths = band_filepaths_df['out_filepath'],
        sequence = sequence,
        working_dir = working_dir,
        njobs = njobs,
        print_messages = print_messages,
    )

    band_filepaths_df.drop(columns=['crs', 'filepath'], inplace=True)
    band_filepaths_df.rename(columns={'out_filepath': 'filepath'}, inplace=True)

    return band_filepaths_df


def resample_to_selected_band_inplace(
    band_filepath_dict:dict[str,str],
    selected_band:str,
    shapes_gdf:gpd.GeoDataFrame,
    resampling = rasterio.warp.Resampling.nearest,
    nodata = 0,
    working_dir:str = None,
):
    ref_filepath = band_filepath_dict[selected_band]
    sequence = [
        (rsutils.modify_images.resample_by_ref, dict(ref_filepath = ref_filepath,
                                             resampling = resampling)),
        (rsutils.modify_images.crop, dict(shapes_gdf=shapes_gdf, nodata=nodata, all_touched=True))
    ]
    filepaths = []
    for band, band_filepath in band_filepath_dict.items():
        if band != selected_band:
            filepaths.append(band_filepath)
    filepaths.append(ref_filepath)

    for filepath in filepaths:
        rsutils.modify_images.modify_image(
            src_filepath = filepath,
            dst_filepath = filepath,
            sequence = sequence,
            working_dir = working_dir,
        )


def resample_to_selected_band_inplace_by_df(
    band_filepaths_df:pd.DataFrame,
    shapes_gdf:gpd.GeoDataFrame,
    nodata = 0,
    resampling_ref_band:str = 'B08',
    resampling = rasterio.warp.Resampling.nearest,
    working_dir:str = None,
    njobs:int = 8,
    print_messages:bool = True,
):
    list_of_band_filepath_dicts = band_filepaths_df.groupby('id')[
        ['band', 'filepath']
    ].apply(
        lambda g: dict(map(tuple, g.values.tolist()))
    ).to_list()

    resample_to_selected_band_inplace_partial = \
    functools.partial(
        resample_to_selected_band_inplace,
        selected_band = resampling_ref_band,
        shapes_gdf = shapes_gdf,
        resampling = resampling,
        nodata = nodata,
        working_dir = working_dir,
    )

    with mp.Pool(njobs) as p:
        if print_messages:
            list(tqdm.tqdm(
                p.imap(resample_to_selected_band_inplace_partial, list_of_band_filepath_dicts), 
                total=len(list_of_band_filepath_dicts)
            ))
        else:
            list(p.imap(resample_to_selected_band_inplace_partial, list_of_band_filepath_dicts))


def get_shape(filepath):
    with rasterio.open(filepath) as src:
        shape = (src.meta['height'], src.meta['width'])
    return shape


def create_merge_reference(
    filepaths_to_merge:list[str],
    working_dir:str = None,
    nodata = 0,
):
    if working_dir is None:
        working_dir = os.path.split(filepaths_to_merge[0])[0]

    merged_ndarray, merged_transform = rasterio.merge.merge(
        filepaths_to_merge, nodata = nodata,
    )

    merged_ndarray[:] = 0

    with rasterio.open(filepaths_to_merge[0]) as src:
        merged_meta = src.meta.copy()

    merged_meta.update({
        'count': merged_ndarray.shape[0],
        'height': merged_ndarray.shape[1],
        'width': merged_ndarray.shape[2],
        'transform': merged_transform,
    })

    merged_meta = rsutils.utils.driver_specific_meta_updates(
        meta=merged_meta, driver='GTiff'
    )

    merge_reference_filepath = os.path.join(
        working_dir, f'merge_reference_{rsutils.utils.get_epochs_str()}.tif'
    )

    with rasterio.open(merge_reference_filepath, 'w', **merged_meta) as dst:
        dst.write(merged_ndarray)
    
    return merge_reference_filepath, merged_meta['height'], merged_meta['width']


def resample_to_merge_master_inplace(
    band_filepaths_df:pd.DataFrame,
    shapes_gdf:gpd.GeoDataFrame,
    nodata = 0,
    merge_ref_band:str = 'B08',
    working_dir:str = None,
    resampling = rasterio.warp.Resampling.nearest,
    njobs:int = 8,
    print_messages:bool = True,
):
    if band_filepaths_df.shape[0] == 0:
        raise exceptions.DatacubeException('band_filepaths_df is empty.')

    band_filepaths_df['shape'] = band_filepaths_df['filepath'].apply(get_shape)
    
    filepaths_to_merge = []

    for shape in band_filepaths_df['shape'].unique():
        filepath = band_filepaths_df[
            (band_filepaths_df['band'] == merge_ref_band)
            & (band_filepaths_df['shape'] == shape)
        ]['filepath'].tolist()[0]
        filepaths_to_merge.append(filepath)

    merge_reference_filepath, height, width \
    = create_merge_reference(
        filepaths_to_merge = filepaths_to_merge,
        working_dir = working_dir,
        nodata = nodata,
    )

    sequence = [
        (rsutils.modify_images.resample_by_ref, dict(ref_filepath = merge_reference_filepath,
                                             resampling = resampling)),
        (rsutils.modify_images.crop, dict(shapes_gdf=shapes_gdf, nodata=nodata, all_touched=True))
    ]

    rsutils.modify_images.modify_images(
        src_filepaths = band_filepaths_df['filepath'],
        dst_filepaths = band_filepaths_df['filepath'],
        sequence = sequence,
        working_dir = working_dir,
        njobs = njobs,
        print_messages = print_messages,
    )

    return merge_reference_filepath, height, width


def save_datacube(
    datacube:np.ndarray,
    metadata:dict,
    folderpath:str,
):
    os.makedirs(folderpath, exist_ok=True)
    datacube_filepath = os.path.join(folderpath, FILENAME_DATACUBE)
    metadata_filepath = os.path.join(folderpath, FILENAME_METADATA)
    np.save(datacube_filepath, datacube)
    np.save(metadata_filepath, metadata, allow_pickle=True)
    return datacube_filepath, metadata_filepath


def load_datacube(folderpath:str)->tuple[np.ndarray, dict]:
    datacube_filepath = os.path.join(folderpath, FILENAME_DATACUBE)
    metadata_filepath = os.path.join(folderpath, FILENAME_METADATA)
    datacube = np.load(datacube_filepath)
    metadata = np.load(metadata_filepath, allow_pickle=True)[()]
    return datacube, metadata


def missing_files_action(
    catalog_filepath:str,
    satellite:str, # catalog can contain multiple satellites
    shapes_gdf:gpd.GeoDataFrame,
    startdate:datetime.datetime,
    enddate:datetime.datetime,
    bands:list[str],
    if_missing_files = 'raise_error',
    ext:str = '.jp2',
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
        catalog_filepath = catalog_filepath,
        satellite = satellite,
        shapes_gdf = shapes_gdf,
        startdate = startdate,
        enddate = enddate,
        files = [f'{band}{ext}' for band in bands],
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


def create_datacube(
    shapes_gdf:gpd.GeoDataFrame,
    catalog_filepath:str,
    satellite:str, # catalog can contain multiple satellites
    startdate:datetime.datetime,
    enddate:datetime.datetime,
    bands:list[str],
    out_folderpath:str,
    working_dir:str,
    nodata = 0,
    njobs:int = 8,
    dst_crs = None,
    resampling = rasterio.warp.Resampling.nearest,
    resampling_ref_band:str = 'B08',
    ext:str = '.jp2',
    delete_working_dir:bool = True,
    satellite_folderpath:str = None, # for maintaining the same folder structure
    print_messages:bool = True,
    if_missing_files:str = 'raise_error', # options: ['raise_error', 'warn', None]
    max_timedelta_days:int = 5,
):  
    missing_files_action(
        catalog_filepath = catalog_filepath,
        satellite = satellite,
        shapes_gdf = shapes_gdf,
        startdate = startdate,
        enddate = enddate,
        bands = bands,
        if_missing_files = if_missing_files,
        ext = ext,
        max_timedelta_days = max_timedelta_days,
    )

    if print_messages:
        print('Cropping tiles and reprojecting to common CRS:')
    band_filepaths_df = crop_and_reproject(
        shapes_gdf = shapes_gdf,
        catalog_filepath = catalog_filepath,
        satellite = satellite,
        startdate = startdate,
        enddate = enddate,
        bands = bands,
        out_folderpath = working_dir,
        satellite_folderpath = satellite_folderpath,
        nodata = nodata,
        working_dir = working_dir,
        njobs = njobs,
        dst_crs = dst_crs,
        print_messages = print_messages,
        ext = ext,
    )

    if print_messages:
        print(f'Resampling cropped images to resolution of {resampling_ref_band} band:')
    resample_to_selected_band_inplace_by_df(
        band_filepaths_df = band_filepaths_df,
        shapes_gdf = shapes_gdf,
        nodata = nodata,
        resampling_ref_band = resampling_ref_band,
        resampling = resampling,
        working_dir = working_dir,
        njobs = njobs,
        print_messages = print_messages,
    )

    if print_messages:
        print(f'Resampling cropped images to merged shape:')
    merge_reference_filepath, height, width \
    = resample_to_merge_master_inplace(
        band_filepaths_df = band_filepaths_df,
        shapes_gdf = shapes_gdf,
        nodata = nodata,
        merge_ref_band = resampling_ref_band,
        working_dir = working_dir,
        resampling = resampling,
        njobs = njobs,
        print_messages = print_messages,
    )

    os.remove(merge_reference_filepath)

    timestamps = band_filepaths_df['timestamp'].unique().tolist()
    timestamps.sort()

    timestamp_band_filepaths_dict = \
    band_filepaths_df.groupby(['timestamp'])[
        ['band', 'filepath']
    ].apply(
        lambda g: dict(map(tuple, g.values.tolist()))
    ).to_dict()

    timestamp_to_id_dict = dict(zip(
        band_filepaths_df['timestamp'],
        band_filepaths_df['id'],
    ))

    datacube = []
    meta = None
    ids = []
    for timestamp in timestamps:
        band_stack = []
        for band in bands:
            band_filepath = timestamp_band_filepaths_dict[timestamp][band]
            with rasterio.open(band_filepath) as src:
                if meta is None:
                    meta = src.meta.copy()
                band_stack.append(src.read())
        band_stack = np.stack(band_stack, axis=-1)
        datacube.append(band_stack)
        del band_stack
        ids.append(timestamp_to_id_dict[timestamp])
    datacube = np.concatenate(datacube, axis=0)

    meta['nodata'] = nodata
    meta['driver'] = 'GTiff'
    rsutils.utils.driver_specific_meta_updates(meta = meta)

    metadata = {
        'geotiff_metadata': meta,
        'timestamps': timestamps,
        'ids': ids,
        'bands': bands,
        'data_shape_desc': ('timestamps|ids', 'height', 'width', 'bands')
    }

    if delete_working_dir:
        shutil.rmtree(working_dir)

    return save_datacube(
        datacube = datacube,
        metadata = metadata,
        folderpath = out_folderpath,
    )

