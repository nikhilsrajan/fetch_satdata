import geopandas as gpd
import datetime

import sys
sys.path.append('..')

import config
import crop_and_zip


if __name__ == '__main__':
    """
    Arguments:
    - ROI shape filepath
    - startdate (YYYY-MM-DD)
    - enddate (YYYY-MM-DD)
    - bands separated by comma, eg B02,B03,B04,B08
    - zip filepath without extension
    - njobs
    """
    NODATA = 0 # since the script is hardcoded for sentinel-2-l1c

    shapes_gdf = gpd.read_file(sys.argv[1])
    startdate = datetime.datetime.strptime(sys.argv[2], '%Y-%m-%d')
    enddate = datetime.datetime.strptime(sys.argv[3], '%Y-%m-%d')
    bands = sys.argv[4].split(',')
    zip_filepath = sys.argv[5]
    njobs = int(sys.argv[6])

    crop_and_zip.crop_and_zip(
        shapes_gdf = shapes_gdf,
        catalog_filepath = config.FILEPATH_SENTINEL2_LOCAL_CATALOG,
        startdate = startdate,
        enddate = enddate,
        bands = bands,
        zip_filepath = zip_filepath,
        satellite_folderpath = config.FOLDERPATH_SATELLITE,
        nodata = NODATA,
        njobs = njobs,
    )
