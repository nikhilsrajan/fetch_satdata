import geopandas as gpd
import datetime
import warnings
import shapely
import os

import catalogmanager
import configsmanager
import create_datacube
import extract_metadata
import datacube_ops
import exceptions


"""
Planned folder structure:

datacubes
`-- s2l1c
    `-- catalog.geojson
    `-- configurations.json
    `-- {roi_name}
        `-- {actual_startdate}_{actual_enddate}
            `-- {config_id}
                `-- datacube.npy
                `-- metadata.pickle.npy
"""


COL_ID = catalogmanager.COL_ID
COL_ROI_NAME = 'roi_name' # unique id for geometry
COL_GEOMETRY = 'geometry'
COL_STARTDATE = 'startdate'
COL_ENDDATE = 'enddate'
COL_CONFIG_ID = 'config_id'
COL_LOCAL_FOLDERPATH = 'local_folderpath'
COL_FILES = 'files'

FILENAME_DATACUBE = 'datacube.npy'
FILENAME_METADATA = 'metadata.pickle.npy'
FILENAME_MEANSUNANGLE = 'mean_sun_angle.csv'

REF_BAND_ORDER = [
    'B08', 'B04', 'B03', 'B02', # 10m
    'B8A', 'B11', 'B12', 'B05', 'B06', 'B07', # 20m
    'B01', 'B09', 'B10', # 60m
]

DATACUBE_ALREADY_EXISTS = 'return::datacube-already-exists'
DATACUBE_CREATED = 'return::datacube-created'
DATACUBE_OVERWRITTEN  = 'return::datacube-overwritten'


class Configuration(object):
    CLOUD_THRESHOLD = 'cloud_threshold'
    MOSAIC_DAYS = 'mosaic_days'
    BANDS = 'bands'

    def __init__(
            self, 
            cloud_threshold:float, 
            mosaic_days:int,
            bands:list[str],
        ):

        self.cloud_threshold = cloud_threshold
        self.mosaic_days = mosaic_days
        self.bands = bands
    
    def to_dict(self):
        return {
            Configuration.CLOUD_THRESHOLD: self.cloud_threshold,
            Configuration.MOSAIC_DAYS: self.mosaic_days,
            Configuration.BANDS: self.bands,
        }

    @classmethod
    def from_dict(cls, d:dict):
        cls(
            cloud_threshold = d[Configuration.CLOUD_THRESHOLD],
            mosaic_days = d[Configuration.MOSAIC_DAYS],
            bands = d[Configuration.BANDS]
        )


def new_config_action(
    config:Configuration,
    configs_filepath:str,
    if_new_config = 'raise_error',
):
    VALID_IF_NEW_CONFIG_OPTIONS = ['raise_error', 'warn', None]
    if not any([if_new_config is x for x in VALID_IF_NEW_CONFIG_OPTIONS]):
        raise exceptions.DatacubeException(
            f'Invalid if_new_config={if_new_config}. '
            f'if_new_config must be from {VALID_IF_NEW_CONFIG_OPTIONS}'
        )

    configs_manager = configsmanager.ConfigsManager(configs_filepath=configs_filepath)
    config_id = configs_manager.get_config_id(config = config.to_dict())

    is_new_config =  config_id == configsmanager.ConfigsManager.NULL_ID
    
    if is_new_config:
        msg = f'New config received -- {config.to_dict()}'
        if if_new_config == 'raise_error':
            raise exceptions.DatacubeException(msg)
        elif if_new_config == 'warn':
            warnings.warn(message = msg, category = RuntimeWarning)

    return is_new_config


def log_and_get_config_id(
    config:Configuration,
    configs_filepath:str,
):
    configs_manager = configsmanager.ConfigsManager(configs_filepath=configs_filepath)
    config_id = configs_manager.log_config(config=config.to_dict())
    configs_manager.save()
    return config_id


# Use this not in run_create_datacube but in the script to let people find which configs
# are very similar.
def get_configs_diff_table(
    config:Configuration,
    configs_filepath:str,
):
    configs_manager = configsmanager.ConfigsManager(configs_filepath=configs_filepath)
    return configs_manager.get_diff_table(query = config.to_dict())


def load_datacube_catalog_manager(
    datacube_catalog_filepath:str,
):
    datacube_catalog_manager = catalogmanager.CatalogManager(
        catalog_filepath = datacube_catalog_filepath,
        cols_dtype_dict = {
            COL_ID: catalogmanager.DTYPE_STR,
            COL_ROI_NAME: catalogmanager.DTYPE_STR,
            COL_GEOMETRY: catalogmanager.DTYPE_MULTIPOLYGON,
            COL_STARTDATE: catalogmanager.DTYPE_TIMESTAMP,
            COL_ENDDATE: catalogmanager.DTYPE_TIMESTAMP,
            COL_CONFIG_ID: catalogmanager.DTYPE_INT,
            COL_LOCAL_FOLDERPATH: catalogmanager.DTYPE_STR,
            COL_FILES: catalogmanager.DTYPE_LIST_STR,
        }
    )
    return datacube_catalog_manager


def get_roi_geom_gdf(
    datacube_catalog_filepath:str,
):
    datacube_catalog_gdf = load_datacube_catalog_manager(
        datacube_catalog_filepath = datacube_catalog_filepath
    ).catalog_gdf

    roi_geom_gdf = gpd.GeoDataFrame(
        datacube_catalog_gdf[
            [COL_ROI_NAME, COL_GEOMETRY]
        ].drop_duplicates().reset_index(drop=True),
        crs = datacube_catalog_gdf.crs,
    )

    return roi_geom_gdf


def check_if_geom_present_exact(
    shapes_gdf:gpd.GeoDataFrame,
    datacube_catalog_filepath:str,
):
    """
    Returns string if same geometry is found else None.
    """
    roi_geom_gdf = get_roi_geom_gdf(
        datacube_catalog_filepath = datacube_catalog_filepath
    )

    unary_gdf = create_datacube.get_unary_gdf(
        shapes_gdf = shapes_gdf,
        crs = roi_geom_gdf.crs,
    )

    unary_geom = unary_gdf['geometry'][0]

    sjoin_roi_geom_gdf = gpd.sjoin(roi_geom_gdf, unary_gdf)
    
    intersecting_roi_geom_gdf = roi_geom_gdf[
        roi_geom_gdf[COL_ROI_NAME].isin(sjoin_roi_geom_gdf[COL_ROI_NAME])
    ]

    same_geom_roi_name = None
    for index, row in intersecting_roi_geom_gdf.iterrows():
        geom_i = row[COL_GEOMETRY]
        if shapely.equals(geom_i, unary_geom):
            same_geom_roi_name = row[COL_ROI_NAME]
            break
    
    return same_geom_roi_name


def check_if_roi_name_already_used(
    roi_name:str,
    datacube_catalog_filepath:str,
):
    """
    Returns True if name already present else False.
    """
    dcm = load_datacube_catalog_manager(
        datacube_catalog_filepath = datacube_catalog_filepath
    )
    return roi_name in set(dcm.catalog_gdf[COL_ROI_NAME].to_list())


# Use this not in run_create_datacube but in the script to let people find which roi_names
# have very similar shape to what they wish to create a datacube for.
def check_if_geom_present_iou(
    shapes_gdf:gpd.GeoDataFrame,
    datacube_catalog_filepath:str,
    iou_threshold:float = 0.97,
):
    """
    Returns GeoDataFrame with geometries with IOU greater than iou_threshold.
    If no geometries found, an empty GeoDataFrame is returned.
    """
    roi_geom_gdf = get_roi_geom_gdf(
        datacube_catalog_filepath = datacube_catalog_filepath
    )

    unary_gdf = create_datacube.get_unary_gdf(
        shapes_gdf = shapes_gdf,
        crs = roi_geom_gdf.crs,
    )

    unary_geom = unary_gdf['geometry'][0]

    overlay_roi_geom_gdf = gpd.overlay(roi_geom_gdf, unary_gdf).set_index(COL_ROI_NAME)

    intersecting_roi_geom_gdf = roi_geom_gdf[
        roi_geom_gdf[COL_ROI_NAME].isin(overlay_roi_geom_gdf.index)
    ].set_index(COL_ROI_NAME)

    intersecting_roi_geom_gdf.loc[
        overlay_roi_geom_gdf.index, 'intersection_area'
    ] = overlay_roi_geom_gdf.area

    intersecting_roi_geom_gdf['IOU'] = \
        intersecting_roi_geom_gdf['intersection_area'] / intersecting_roi_geom_gdf[COL_GEOMETRY].apply(
            lambda x: shapely.union(x, unary_geom).area
        )

    cleared_iou_roi_geom_gdf = intersecting_roi_geom_gdf[
        intersecting_roi_geom_gdf['IOU'] >= iou_threshold
    ].sort_values(by='IOU', ascending=False).reset_index()

    cleared_iou_roi_geom_gdf.drop(columns=['intersection_area'], inplace=True)

    return cleared_iou_roi_geom_gdf


def get_datacube_folderpath(
    root_folderpath:str,
    roi_name:str,
    actual_startdate:datetime.datetime,
    actual_enddate:datetime.datetime,
    config_id:int,
):
    return os.path.join(
        root_folderpath,
        roi_name,
        f"{actual_startdate.strftime('%Y%m%dT%H%M%S')}_{actual_enddate.strftime('%Y%m%dT%H%M%S')}",
        str(config_id),
    )


def get_datacube_filepaths(
    root_folderpath:str,
    roi_name:str,
    actual_startdate:datetime.datetime,
    actual_enddate:datetime.datetime,
    config_id:int,
):
    folderpath = get_datacube_folderpath(
        root_folderpath = root_folderpath,
        roi_name = roi_name,
        actual_startdate = actual_startdate,
        actual_enddate = actual_enddate,
        config_id = config_id,
    )
    
    datacube_filepath = os.path.join(folderpath, FILENAME_DATACUBE)
    metadata_filepath = os.path.join(folderpath, FILENAME_METADATA)

    return datacube_filepath, metadata_filepath


def check_if_datacube_already_present(
    datacube_catalog_filepath:str,
    roi_name:str,
    actual_startdate:datetime.datetime,
    actual_enddate:datetime.datetime,
    config_id:int,
):
    dcm = load_datacube_catalog_manager(
        datacube_catalog_filepath = datacube_catalog_filepath,
    )
    same_datacube_df = dcm.catalog_gdf[
        (dcm.catalog_gdf[COL_ROI_NAME] == roi_name) &
        (dcm.catalog_gdf[COL_STARTDATE] == create_datacube.dt2ts(dt=actual_startdate)) &
        (dcm.catalog_gdf[COL_ENDDATE] == create_datacube.dt2ts(dt=actual_enddate)) &
        (dcm.catalog_gdf[COL_CONFIG_ID] == config_id)
    ]

    if same_datacube_df.shape[0] == 1:
        folderpath = same_datacube_df[COL_LOCAL_FOLDERPATH].to_list()[0]
        # I know that COL_FILES is basically list of string joined by ',' 
        # cause of how CatalogManager is designed to function.
        # Should be a better way to do this so that it won't look like magic.
        # Perhaps by having a get method within CatalogManager
        files = same_datacube_df[COL_FILES].to_list()[0].split(',')
        catalog_mentioned_filepaths = [
            os.path.join(folderpath, file) for file in files
        ]
        if all(os.path.exists(filepath) for filepath in catalog_mentioned_filepaths):
            return True
        else:
            raise exceptions.MajorException(
                "This should have not happened. File mentioned in the catalog should "
                "not be absent from the machine.\n"
                f"{COL_ROI_NAME}: {roi_name}\n"
                f"{COL_STARTDATE}: {create_datacube.dt2ts(dt=actual_startdate)}\n"
                f"{COL_ENDDATE}: {create_datacube.dt2ts(dt=actual_enddate)}\n"
                f"{COL_CONFIG_ID}: {config_id}\n"
                f"{COL_LOCAL_FOLDERPATH}: {folderpath}\n"
                f"catalog_mentioned_filepaths: {catalog_mentioned_filepaths}"
            )

    elif same_datacube_df.shape[0] > 1:
        raise exceptions.MajorException(
            "This should have not happened. There is not shouldn't have been multiple entries "
            f"with same {COL_ROI_NAME}, {COL_STARTDATE}, {COL_ENDDATE}, and {COL_CONFIG_ID}.\n"
            f"{COL_ROI_NAME}: {roi_name}\n"
            f"{COL_STARTDATE}: {create_datacube.dt2ts(dt=actual_startdate)}\n"
            f"{COL_ENDDATE}: {create_datacube.dt2ts(dt=actual_enddate)}\n"
            f"{COL_CONFIG_ID}: {config_id}"
        )

    else:
        return False


def run_datacube_ops(
    folderpath:str,
    sequence:list,
    print_messages:bool = True,
):
    datacube, metadata = create_datacube.load_datacube(
        folderpath = folderpath
    )
    datacube, metadata = datacube_ops.run_datacube_ops(
        datacube = datacube,
        metadata = metadata,
        sequence = sequence,
        print_messages = print_messages,
    )
    create_datacube.save_datacube(
        datacube = datacube,
        metadata = metadata,
        folderpath = folderpath,
    )


def create_s2l1c_datacube(
    shapes_gdf:gpd.GeoDataFrame,
    export_folderpath:str,
    satellite_catalog_filepath:str,
    startdate:datetime.datetime,
    enddate:datetime.datetime,
    bands:list[str],
    njobs:int,
    s2cloudless_chunksize:int = None,
    cloud_threshold:float = 1, 
    mosaic_days = None,
    print_messages:bool = True,
    if_missing_files = 'raise_error',
):
    NODATA = 0 # since the script is hardcoded for sentinel-2-l1c
    EXT = '.jp2' # since the script is hardcoded for sentinel-2-l1c
    MAX_TIMEDELTA_DAYS = 5 # since the script is hardcoded for sentinel-2-l1c

    for ref_band_candidate in REF_BAND_ORDER:
        if ref_band_candidate in bands:
            resampling_ref_band = ref_band_candidate
            break

    working_dir = os.path.join(export_folderpath, 'temp')

    create_datacube.create_datacube(
        shapes_gdf = shapes_gdf,
        catalog_filepath = satellite_catalog_filepath,
        startdate = startdate,
        enddate = enddate,
        bands = bands,
        out_folderpath = export_folderpath,
        working_dir = working_dir,
        nodata = NODATA,
        njobs = njobs,
        resampling_ref_band = resampling_ref_band,
        delete_working_dir = True,
        satellite_folderpath = None,
        print_messages = print_messages,
        ext = EXT,
        if_missing_files = if_missing_files,
        max_timedelta_days = MAX_TIMEDELTA_DAYS,
    )

    if print_messages:
        print('Extracting mean_sun_angle:')
    mean_sun_angle_df = \
    extract_metadata.extract_s2l1c_mean_sun_angle(
        shapes_gdf = shapes_gdf,
        catalog_filepath = satellite_catalog_filepath,
        startdate = startdate,
        enddate = enddate,
        print_messages = print_messages,
    )
    mean_sun_angle_filepath = os.path.join(export_folderpath, FILENAME_MEANSUNANGLE)
    mean_sun_angle_df.to_csv(mean_sun_angle_filepath, index=False)

    datacube_ops_sequence = []

    if s2cloudless_chunksize is not None:
        datacube_ops_sequence.append((
            datacube_ops.run_s2cloudless, dict(chunksize = s2cloudless_chunksize, 
                                               njobs = njobs, 
                                               print_messages = print_messages)
        ))
    
    if cloud_threshold is not None and mosaic_days is not None:
        datacube_ops_sequence.append((
            datacube_ops.apply_cloud_mask, dict(cloud_threshold = cloud_threshold * 10000)
        ))
        datacube_ops_sequence.append((
            datacube_ops.drop_bands, dict(bands_to_drop = ['CMK'])
        ))
        datacube_ops_sequence.append((
            datacube_ops.median_mosaic, dict(startdate = startdate,
                                             enddate = enddate,
                                             mosaic_days = mosaic_days,)
        ))

    run_datacube_ops(
        folderpath = export_folderpath,
        sequence = datacube_ops_sequence,
        print_messages = print_messages,
    )


def get_datacube_id(
    roi_name:str,
    actual_startdate:datetime.datetime,
    actual_enddate:datetime.datetime,
    config_id:int,
):
    return '_'.join([
        roi_name,
        actual_startdate.strftime('%Y%m%dT%H%M%S'),
        actual_enddate.strftime('%Y%m%dT%H%M%S'),
        str(config_id)
    ])


def update_catalog(
    datacube_catalog_filepath:str,
    roi_name:str,
    actual_startdate:datetime.datetime,
    actual_enddate:datetime.datetime,
    config_id:int,
    datacube_folderpath:str,
    geometry_epsg_4326:shapely.Geometry,
):
    dcm = load_datacube_catalog_manager(
        datacube_catalog_filepath = datacube_catalog_filepath,
    )
    dcm.add(entry = {
        COL_ID: get_datacube_id(roi_name = roi_name,
                                actual_startdate = actual_startdate,
                                actual_enddate = actual_enddate,
                                config_id = config_id),
        COL_ROI_NAME: roi_name,
        COL_STARTDATE: actual_startdate,
        COL_ENDDATE: actual_enddate,
        COL_CONFIG_ID: config_id,
        COL_LOCAL_FOLDERPATH: datacube_folderpath,
        COL_FILES: [FILENAME_DATACUBE, FILENAME_METADATA, FILENAME_MEANSUNANGLE],
        COL_GEOMETRY: geometry_epsg_4326,
    })
    dcm.save()


def create_s2l1c_datacube_and_update_catalog(
    satellite_catalog_filepath:str,
    datacube_catalog_filepath:str,
    configs_filepath:str,
    datacubes_folderpath:str,
    roi_name:str,
    shapes_gdf:gpd.GeoDataFrame, 
    startdate:datetime.datetime, 
    enddate:datetime.datetime, 
    bands:list[str],
    njobs:int, 
    s2cloudless_chunksize:int = None,
    cloud_threshold:float = 1, 
    mosaic_days = None,
    print_messages:bool = True,
    if_missing_files:bool = 'raise_error', # options: ['raise_error', 'warn', None]
    ext:str = '.jp2',
    if_new_config = 'raise_error',
    overwrite:bool = False,
):
    config = Configuration(
        cloud_threshold = cloud_threshold,
        mosaic_days = mosaic_days,
        bands = bands,
    )

    is_new_config = new_config_action(
        config = config,
        configs_filepath = configs_filepath,
        if_new_config = if_new_config,
    )
    # we reach here when a user has decided not to raise_error
    config_id = log_and_get_config_id(
        config = config,
        configs_filepath = configs_filepath,
    )

    query_stats, missing_flags = \
    create_datacube.missing_files_action(
        catalog_filepath = satellite_catalog_filepath,
        shapes_gdf = shapes_gdf,
        startdate = startdate,
        enddate = enddate,
        bands = bands,
        if_missing_files = if_missing_files,
        ext = ext,
        max_timedelta_days = 5, # hard-coded for sentinel-2
    )

    actual_startdate = query_stats['timestamp_range'][0]
    actual_enddate = query_stats['timestamp_range'][1]

    same_shape_roi_name = check_if_geom_present_exact(
        shapes_gdf = shapes_gdf,
        datacube_catalog_filepath = datacube_catalog_filepath,
    )
    if same_shape_roi_name is not None:
        if same_shape_roi_name != roi_name:
            raise exceptions.DatacubeException(
                "The given geometry matches exactly with a previous entry "
                f"under a different roi_name='{same_shape_roi_name}'. "
                "Check with the other users of the create_s2l1c_datacube or "
                f"consider using '{same_shape_roi_name}' as the roi_name instead "
                f"of '{roi_name}'."
            )
    elif check_if_roi_name_already_used(
        roi_name=roi_name, 
        datacube_catalog_filepath=datacube_catalog_filepath,
    ):
        # same geom not present in the catalog but roi_name already used for another geom
        raise exceptions.DatacubeException(
            f"roi_name={roi_name} is already used for another geometry. "
            "Please try a different roi_name."
        )
    
    will_be_overwritten = False
        
    if check_if_datacube_already_present(
        datacube_catalog_filepath = datacube_catalog_filepath,
        roi_name = roi_name,
        actual_startdate = actual_startdate,
        actual_enddate = actual_enddate,
        config_id = config_id,
    ):
        if not overwrite:
            if print_messages:
                print(
                    f'Datacube already exists -- roi_name={roi_name}, actual_startdate={actual_startdate} '
                    f'actual_enddate={actual_enddate}, config_id={config_id}'
                )
            return DATACUBE_ALREADY_EXISTS
        else:
            will_be_overwritten = True
    
    
    datacube_folderpath = get_datacube_folderpath(
        root_folderpath = datacubes_folderpath,
        roi_name = roi_name,
        actual_startdate = actual_startdate,
        actual_enddate = actual_enddate,
        config_id = config_id,
    )

    create_s2l1c_datacube(
        shapes_gdf = shapes_gdf,
        export_folderpath = datacube_folderpath,
        satellite_catalog_filepath = satellite_catalog_filepath,
        startdate = startdate,
        enddate = enddate,
        bands = bands,
        njobs = njobs,
        s2cloudless_chunksize = s2cloudless_chunksize,
        cloud_threshold = cloud_threshold,
        mosaic_days = mosaic_days,
        print_messages = print_messages,
        if_missing_files = if_missing_files,
    )

    union_shape_gdf = create_datacube.get_unary_gdf(shapes_gdf=shapes_gdf, 
                                                    crs=catalogmanager.EPSG_4326)
    union_shape = union_shape_gdf['geometry'][0]

    if print_messages:
        print(f'Updating catalog: {datacube_catalog_filepath}')

    update_catalog(
        datacube_catalog_filepath = datacube_catalog_filepath,
        roi_name = roi_name,
        actual_startdate = actual_startdate,
        actual_enddate = actual_enddate,
        config_id = config_id,
        datacube_folderpath = datacube_folderpath,
        geometry_epsg_4326 = union_shape,
    )

    if print_messages:
        print(
            f'Datacube created -- roi_name={roi_name}, actual_startdate={actual_startdate} '
            f'actual_enddate={actual_enddate}, config_id={config_id}'
        )
    
    if will_be_overwritten:
        return DATACUBE_OVERWRITTEN
    
    return DATACUBE_CREATED
