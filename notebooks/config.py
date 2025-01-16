import os


CURRENT_DIRECTORY = os.path.dirname(__file__)

FOLDERPATH_SATELLITE = os.path.join(CURRENT_DIRECTORY, '../data/satellite')
FOLDERPATH_DATACUBES = os.path.join(CURRENT_DIRECTORY, '../data/datacubes')
FOLDERPATH_FETCH_CATALOG_CACHE = os.path.join(CURRENT_DIRECTORY, '../data/fetch_catalog_cache')

FILEPATH_SENTINEL2_LOCAL_CATALOG = os.path.join(FOLDERPATH_SATELLITE, 'Sentinel-2/catalog.geojson')
