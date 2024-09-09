import os

FOLDERPATH_SATELLITE = '../data/satellite'
FOLDERPATH_FETCH_CATALOG_CACHE = '../data/fetch_catalog_cache'
FOLDERPATH_DATACUBES = '../data/datacubes'
FOLDERPATH_DATACUBES_S2L1C = '../data/datacubes/s2l1c'

FILEPATH_SENTINEL2_LOCAL_CATALOG = os.path.join(FOLDERPATH_SATELLITE, 'Sentinel-2/catalog.geojson')
FILEPATH_S2L1C_DATACUBE_CATALOG = os.path.join(FOLDERPATH_DATACUBES_S2L1C, 'catalog.geojson')
FILEPATH_S2L1C_DATACUBE_CONFIG_TRACKER = os.path.join(FOLDERPATH_DATACUBES_S2L1C, 'configurations.json')
