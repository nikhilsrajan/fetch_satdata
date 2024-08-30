import os
import paramiko
import geopandas as gpd
import functools
import multiprocessing as mp
import tqdm
import datetime
import pandas as pd

import sshcreds
import create_stack



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
        temp_download_filepath = download_filepath + '.temp'

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
            localpath = temp_download_filepath,
        )

        ftp_client.close()

        if os.path.exists(download_filepath):
            os.remove(download_filepath)

        os.rename(temp_download_filepath, download_filepath)
    
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
        raise ValueError('Size of remotepaths and download_filepaths do not match.')
    
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


def download_intersecting_sentinel2_tiles_from_cluster(
    shapes_gdf:gpd.GeoDataFrame,
    startdate:datetime,
    enddate:datetime,
    sshcreds:sshcreds.SSHCredentials,
    satellite_folderpath:str,
    bands:list[str],
    overwrite_catalog:bool=False,
    njobs:int = 4,
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

    band_filepaths_df:pd.DataFrame = \
    create_stack.get_intersecting_band_filepaths(
        shapes_gdf = shapes_gdf,
        startdate = startdate,
        enddate = enddate,
        catalog_filepath = catalog_filepath,
        bands = bands,
    )

    band_filepaths_df.rename(columns={'filepath': 'remotepath'}, inplace=True)
    band_filepaths_df['download_filepath'] = \
    band_filepaths_df['remotepath'].apply(
        lambda remotepath: remotepath_to_localpath(
            remotepath = remotepath,
            remote_root_path = FOLDERPATH_SATELLITE,
            local_root_path = satellite_folderpath,
        )
    )

    download_successes = download_files_from_cluster(
        sshcreds = sshcreds,
        remotepaths = band_filepaths_df['remotepath'],
        download_filepaths = band_filepaths_df['download_filepath'],
        overwrite = False,
        njobs = njobs,
    )

    band_filepaths_df.rename(columns={'download_filepath': 'filepath'}, inplace=True)

    band_filepaths_df['download_success'] = download_successes

    return band_filepaths_df

