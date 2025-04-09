import os


AUTHORS = [
    'nsasiraj@umd.edu',
]

MAINTAINERS = [
    'nsasiraj@umd.edu',
]

CURRENT_DIRECTORY = os.path.dirname(__file__)


"""
Folder structure:
----------------
.
`-- satellite/
|   `-- catalog.db
|   |   - table:sentinel-2-l1c, cols:
|   |   - table:sentinel-2-l2a, cols:
|   `-- Sentinel-2/
|   `-- ...
|   `-- ...
`-- datacubes/
    `-- catalog.db
    |   - table:sentinel-2-l1c, cols: 
    |   - table:sentinel-2-l2a, cols:
    `-- configurations.db
    |   - table:sentinel-2-l1c, cols:
    |   - table:sentinel-2-l2a, cols:
    `-- geometries.db
    |   - table:geometries, cols: shape_name, geometry
    `-- sentinel-2-l2a/
    |  `-- {shape-name}
    |      `-- {actual_startdate}_{actual_enddate}
    |          `-- {config_id}
    |              `-- datacube.npy
    |              `-- metadata.pickle.npy
    `-- sentinel-2-l1c/
       `-- {shape-name}
           `-- {actual_startdate}_{actual_enddate}
               `-- {config_id}
                   `-- datacube.npy
                   `-- metadata.pickle.npy
"""


FOLDERPATH_SATELLITE = os.path.join(CURRENT_DIRECTORY, '../data/satellite')
FOLDERPATH_FETCH_CATALOG_CACHE = os.path.join(CURRENT_DIRECTORY, '../data/fetch_catalog_cache')
FOLDERPATH_DATACUBES = os.path.join(CURRENT_DIRECTORY, '../data/datacubes')
FOLDERPATH_DATACUBES_S2L1C = os.path.join(CURRENT_DIRECTORY, '../data/datacubes/sentinel-2-l1c')
FOLDERPATH_DATACUBES_S2L2A = os.path.join(CURRENT_DIRECTORY, '../data/datacubes/sentinel-2-l2a')
FOLDERPATH_ESA = os.path.join(CURRENT_DIRECTORY, '../data/esa')

FILEPATH_SATELLITE_CATALOG_DB = os.path.join(FOLDERPATH_SATELLITE, 'catalog.db')
FILEPATH_DATACUBE_CATALOG_DB = os.path.join(FOLDERPATH_DATACUBES, 'catalog.db')
FILEPATH_CONFIG_REGISTRY_DB = os.path.join(FOLDERPATH_DATACUBES, 'configurations.db')

# Tables in the FILEPATH_SATELLITE_CATALOG_DB, FILEPATH_DATACUBE_CATALOG_DB and FILEPATH_CONFIG_REGISTRY_DB
S2L1C_TABLE = 'sentinel-2-l1c'
S2L2A_TABLE = 'sentinel-2-l2a'

FILEPATH_POLYGONS_REGISTRY_DB = os.path.join(FOLDERPATH_DATACUBES, 'polygons.db')
GEOMETRIES_TABLE = 'geometries'

