import os
import geopandas as gpd
import datetime
import pandas as pd
import json

import sshutils
import sshcreds
import create_datacube
import create_s2l1c_datacube


# not using os.path.join as separator is set by local OS where as UMD cluster is a fixed OS
FOLDERPATH_SATELLITE = '/gpfs/data1/cmongp2/sasirajann/fetch_satdata/data/satellite/'
FOLDERPATH_DATACUBE = '/gpfs/data1/cmongp2/sasirajann/fetch_satdata/data/datacubes/'
FOLDERPATH_DATACUBE_S2L1C = FOLDERPATH_DATACUBE + 's2l1c/'

FILEPATH_SATELLITE_SENTINEL2_CATALOG = FOLDERPATH_SATELLITE + 'Sentinel-2/catalog.geojson'
FILEPATH_DATACUBE_S2L1C_CATALOG = FOLDERPATH_DATACUBE_S2L1C + 'catalog.geojson'
FILEPATH_DATACUBE_S2L1C_CONFIGURATIONS = FOLDERPATH_DATACUBE_S2L1C + 'configurations.json'


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
        remotepath = FILEPATH_SATELLITE_SENTINEL2_CATALOG,
        remote_root_path = FOLDERPATH_SATELLITE,
        local_root_path = satellite_folderpath,
    )

    sshutils.download_file_from_cluster(
        sshcreds = sshcreds,
        remotepath = FILEPATH_SATELLITE_SENTINEL2_CATALOG,
        download_filepath = catalog_filepath,
        overwrite = overwrite_catalog,
    )

    band_filepaths_df:pd.DataFrame = \
    create_datacube.get_intersecting_band_filepaths(
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

    download_successes = sshutils.download_files_from_cluster(
        sshcreds = sshcreds,
        remotepaths = band_filepaths_df['remotepath'],
        download_filepaths = band_filepaths_df['download_filepath'],
        overwrite = False,
        njobs = njobs,
    )

    band_filepaths_df.rename(columns={'download_filepath': 'filepath'}, inplace=True)

    band_filepaths_df['download_success'] = download_successes

    return band_filepaths_df


def load_datacube_s2l1c_catalog(
    sshcreds:sshcreds.SSHCredentials,
    datacubes_folderpath:str,
    overwrite:bool = False,
) -> gpd.GeoDataFrame:
    catalog_filepath = remotepath_to_localpath(
        remotepath = FILEPATH_DATACUBE_S2L1C_CATALOG,
        remote_root_path = FOLDERPATH_DATACUBE,
        local_root_path = datacubes_folderpath,
    )

    sshutils.download_file_from_cluster(
        sshcreds = sshcreds,
        remotepath = FILEPATH_DATACUBE_S2L1C_CATALOG,
        download_filepath = catalog_filepath,
        overwrite = overwrite,
    )

    return gpd.read_file(catalog_filepath)


def load_datacube_s2l1c_configurations(
    sshcreds:sshcreds.SSHCredentials,
    datacubes_folderpath:str,
    overwrite:bool = False,
) -> gpd.GeoDataFrame:
    configs_filepath = remotepath_to_localpath(
        remotepath = FILEPATH_DATACUBE_S2L1C_CONFIGURATIONS,
        remote_root_path = FOLDERPATH_DATACUBE,
        local_root_path = datacubes_folderpath,
    )

    sshutils.download_file_from_cluster(
        sshcreds = sshcreds,
        remotepath = FILEPATH_DATACUBE_S2L1C_CONFIGURATIONS,
        download_filepath = configs_filepath,
        overwrite = overwrite,
    )

    with open(configs_filepath) as h:
        configs = json.load(h)

    return configs


def download_s2l1c_datacube(
    sshcreds:sshcreds.SSHCredentials,
    id:str,
    datacubes_folderpath:str,
    overwrite_catalog:bool = False,
    overwrite:bool = False,
):
    catalog_gdf = load_datacube_s2l1c_catalog(
        sshcreds = sshcreds,
        datacubes_folderpath = datacubes_folderpath,
        overwrite = overwrite_catalog,
    )

    if id not in set(catalog_gdf[create_s2l1c_datacube.COL_ID]):
        msg = f'id={id} not present in the s2l1c catalog.'
        if not overwrite_catalog:
            msg += ' Perhaps try overwrite_catalog=True in case the catalog is out of date.'
        raise KeyError(msg)

    selected_catalog_gdf = catalog_gdf[
        catalog_gdf[create_s2l1c_datacube.COL_ID] == id
    ]

    if selected_catalog_gdf.shape[0] != 1: # will not be 0 cause that case is covered.
        raise ValueError(
            'This was not supposed to happen. id is supposed to be unique. '
            'Something is not correct with the catalog creation process. '
            'Please resolve this ASAP.'
        )

    # selected_catalog_gdf should have only 1 element at this point.

    remote_datacube_folderpath = selected_catalog_gdf[
        create_s2l1c_datacube.COL_LOCAL_FOLDERPATH
    ].to_list()[0]

    files = selected_catalog_gdf[create_s2l1c_datacube.COL_FILES].to_list()[0].split(',')

    download_filepaths = {}

    for file in files:
        # adding '/' cause the local_folderpath in catalog does not
        # end with it
        remotepath = remote_datacube_folderpath + '/' + file

        _download_filepath = remotepath_to_localpath(
            remotepath = remotepath,
            remote_root_path = FOLDERPATH_DATACUBE,
            local_root_path = datacubes_folderpath,
        )

        sshutils.download_file_from_cluster(
            sshcreds = sshcreds,
            remotepath = remotepath,
            download_filepath = _download_filepath,
            overwrite = overwrite,
        )

        download_filepaths[file] = _download_filepath
    
    return download_filepaths
