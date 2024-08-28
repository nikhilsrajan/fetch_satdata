import os
import geopandas as gpd
import datetime
import shutil

import sys
sys.path.append('..')

import config
import create_stack
import extract_metadata


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

    create_stack.create_stack(
        shapes_gdf = shapes_gdf,
        catalog_filepath = config.FILEPATH_SENTINEL2_LOCAL_CATALOG,
        startdate = startdate,
        enddate = enddate,
        bands = bands,
        out_folderpath = zip_filepath,
        working_dir = zip_filepath,
        nodata = NODATA,
        njobs = njobs,
        resampling_ref_band = 'B08',
        delete_working_dir = True,
        satellite_folderpath = None,
    )

    print('Extracting mean_sun_angle:')
    mean_sun_angle_df = \
    extract_metadata.extract_s2l1c_mean_sun_angle(
        shapes_gdf = shapes_gdf,
        catalog_filepath = config.FILEPATH_SENTINEL2_LOCAL_CATALOG,
        startdate = startdate,
        enddate = enddate,
    )
    mean_sun_angle_df.to_csv(os.path.join(zip_filepath, 'mean_sun_angle.csv'), index=False)

    print('Zipping files...')
    final_zip_filepath = shutil.make_archive(
        zip_filepath,
        'zip',
        zip_filepath,
    )

    shutil.rmtree(zip_filepath)

    print(f'Outputs zipped and saved at: {os.path.abspath(final_zip_filepath)}')
