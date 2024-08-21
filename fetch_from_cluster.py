import os
import paramiko
import geopandas as gpd
import functools
import multiprocessing as mp
import tqdm
import datetime
import pandas as pd
import shapely.ops

import sshcreds



FOLDERPATH_SATELLITE = '/gpfs/data1/cmongp2/sasirajann/fetch_satdata/data/satellite/'

# not using os.path.join as separator is set by local OS where as UMD cluster is a fixed OS
FILEPATH_SENTINEL2_CATALOG = FOLDERPATH_SATELLITE + 'Sentinel-2/catalog.geojson'


def download_file_from_cluster(
    sshcreds:sshcreds.SSHCredentials,
    remotepath:str,
    download_filepath:str = None,
    download_folderpath:str = None,
    enable_auto_add_policy:bool = True, # Trust all policy, perhaps best to keep it optional
    overwrite:bool = False,
):
    if download_folderpath is None and download_filepath is None:
        raise Exception(
            "Either 'download_folderpath' or 'download_filepath' " + \
            "should be non None."
        )

    if download_filepath is None:
        filename = remotepath.split('/')[-1]
        download_filepath = os.path.join(
            download_folderpath, filename,
        )
    else:
        download_folderpath = os.path.split(download_filepath)[0]
    
    os.makedirs(download_folderpath, exist_ok=True)

    if not os.path.exists(download_filepath) or overwrite:
        # https://medium.com/@keagileageek/paramiko-how-to-ssh-and-file-transfers-with-python-75766179de73
        ssh_client = paramiko.SSHClient()

        if enable_auto_add_policy:
            ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        ssh_client.connect(
            hostname = sshcreds.hostname,
            username = sshcreds.username,
            password = sshcreds.password,
        )

        ftp_client = ssh_client.open_sftp()

        ftp_client.get(
            remotepath = remotepath,
            localpath = download_filepath,
        )

        ftp_client.close()
    
    return download_filepath


def _download_file_from_cluster_by_tuple(
    remotepath_download_filepath_tuple:tuple[str, str],
    sshcreds:sshcreds.SSHCredentials,
    overwrite:bool = False,
):
    remotepath, download_filepath = remotepath_download_filepath_tuple

    download_file_from_cluster(
        sshcreds = sshcreds,
        remotepath = remotepath,
        download_filepath = download_filepath,
        enable_auto_add_policy = True,
        overwrite = overwrite,
    )
    download_success = os.path.exists(download_filepath)

    return download_success


def download_files_from_cluster(
    sshcreds:sshcreds.SSHCredentials,
    remotepaths:list[str],
    download_filepaths:list[str],
    overwrite:bool = False,
    njobs:int = 16,
):
    if len(remotepaths) != len(download_filepaths):
        raise ValueError('Size of s3paths and download_filepaths do not match.')
    
    download_file_from_cluster_by_tuple_partial = functools.partial(
        _download_file_from_cluster_by_tuple,
        sshcreds = sshcreds,
        overwrite = overwrite,
    )

    remotepath_download_filepath_tuples = list(zip(remotepaths, download_filepaths))

    with mp.Pool(njobs) as p:
        download_successes = list(tqdm.tqdm(
            p.imap(download_file_from_cluster_by_tuple_partial, remotepath_download_filepath_tuples), 
            total=len(remotepath_download_filepath_tuples)
        ))
    
    return download_successes


def remotepath_to_localpath(
    remotepath:str,
    remote_root_path:str,
    local_root_path:str,
):  
    if not remotepath.startswith(remote_root_path):
        raise ValueError(
            'remotepath does not start with remote_root_path\n'
            f"remotepath='{remotepath}'\n"
            f"remote_root_path='{remote_root_path}'"
        )
    return os.path.join(
        local_root_path,
        *remotepath.removeprefix(remote_root_path).split('/')
    )
    

def query_sentinel2_catalog(
    shapes_gdf:gpd.GeoDataFrame,
    startdate:datetime,
    enddate:datetime,
    sshcreds:sshcreds.SSHCredentials,
    satellite_folderpath:str,
    overwrite_catalog:bool=False,
):
    catalog_filepath = remotepath_to_localpath(
        remotepath = FILEPATH_SENTINEL2_CATALOG,
        remote_root_path = FOLDERPATH_SATELLITE,
        local_root_path = satellite_folderpath,
    )

    download_file_from_cluster(
        sshcreds = sshcreds,
        remotepath = FILEPATH_SENTINEL2_CATALOG,
        download_filepath = catalog_filepath,
        overwrite = overwrite_catalog,
    )

    catalog_gdf = gpd.read_file(catalog_filepath)

    union_shape = shapely.ops.unary_union(
        shapes_gdf.to_crs(catalog_gdf.crs)['geometry']
    )
    union_shape_gdf = gpd.GeoDataFrame(
        data = {'geometry': [union_shape]},
        crs = catalog_gdf.crs,
    )

    dt_filtered_catalog_gdf = catalog_gdf[
        (catalog_gdf['timestamp'] >= pd.Timestamp(startdate, tz='UTC'))
        & (catalog_gdf['timestamp'] <= pd.Timestamp(enddate, tz='UTC'))
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


def download_intersecting_sentinel2_tiles_from_cluster(
    shapes_gdf:gpd.GeoDataFrame,
    startdate:datetime,
    enddate:datetime,
    sshcreds:sshcreds.SSHCredentials,
    satellite_folderpath:str,
    bands:list[str],
    overwrite_catalog:bool=False,
    njobs:int = 16,
):
    EXT_JP2 = '.jp2'

    catalog_gdf = query_sentinel2_catalog(
        shapes_gdf = shapes_gdf,
        startdate = startdate,
        enddate = enddate,
        sshcreds = sshcreds,
        satellite_folderpath = satellite_folderpath,
        overwrite_catalog = overwrite_catalog,
    )

    catalog_gdf['files_to_download'] = catalog_gdf['files'].apply(
        lambda x: ','.join({
            xi + EXT_JP2 for xi in
            {xi.removesuffix(EXT_JP2) for xi in x.split(',')} & set(bands)
        })
    )

    data = {
        'id': [],
        'timestamp': [],
        'band': [],
        'remotepath': [],
        'download_filepath': [],
    }

    for _id, timestamp, local_folderpath, files in zip(
        catalog_gdf['id'],
        catalog_gdf['timestamp'],
        catalog_gdf['local_folderpath'],
        catalog_gdf['files'],
    ):
        filenames_of_interest = {
            xii + EXT_JP2 
            for xii in {
                xi.removesuffix(EXT_JP2) 
                for xi in files.split(',')
            } & set(bands)
        }
        
        # if no files of interest present, skip
        if len(filenames_of_interest) == 0:
            continue

        _remotepaths = [
            local_folderpath + '/' + filename_of_interest
            for filename_of_interest in filenames_of_interest
        ]
        _download_filepaths = [
            remotepath_to_localpath(
                remotepath = _remotepath,
                remote_root_path = FOLDERPATH_SATELLITE,
                local_root_path = satellite_folderpath,
            )
            for _remotepath in _remotepaths
        ]
        data['id'] += [_id for _ in range(len(_remotepaths))]
        data['timestamp'] += [timestamp for _ in range(len(_remotepaths))]
        data['band'] += [
            filename_of_interest.removesuffix(EXT_JP2)
            for filename_of_interest in filenames_of_interest
        ]
        data['remotepath'] += _remotepaths
        data['download_filepath'] += _download_filepaths

        del _remotepaths, _download_filepaths

    download_successes = download_files_from_cluster(
        sshcreds = sshcreds,
        remotepaths = data['remotepath'],
        download_filepaths = data['download_filepath'],
        overwrite = False,
        njobs = njobs,
    )

    data['download_success'] = download_successes

    return pd.DataFrame(data=data)
