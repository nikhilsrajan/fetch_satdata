import os
import geopandas as gpd
import datetime
import shutil
import time
import argparse

import sys
sys.path.append('..')

import config
import rsutils.s2_grid_utils
import create_s2l1c_datacube


def main(
    shapes_gdf:gpd.GeoDataFrame, 
    startdate:datetime.datetime, 
    enddate:datetime.datetime, 
    bands:list[str],
    zip_filepath:str, 
    njobs:int, 
    s2cloudless_chunksize:int = None,
    cloud_threshold:float = 1, 
    mosaic_days = None,
    print_messages:bool = True,
    if_missing_files = 'raise_error',
):
    if print_messages:
        print('--- run ---')

    out_folderpath = zip_filepath.removesuffix('.zip')

    create_s2l1c_datacube.create_s2l1c_datacube(
        shapes_gdf = shapes_gdf,
        export_folderpath = out_folderpath,
        satellite_catalog_filepath = config.FILEPATH_SENTINEL2_LOCAL_CATALOG,
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

    if print_messages:
        print('Zipping files...')
    final_zip_filepath = shutil.make_archive(
        out_folderpath,
        'zip',
        out_folderpath,
    )

    shutil.rmtree(out_folderpath)

    if print_messages:
        print(f'Outputs zipped and saved at: {os.path.abspath(final_zip_filepath)}')


if __name__ == '__main__':
    start_time = time.time()

    parser = argparse.ArgumentParser(
        prog = 'python create_s2l1c_datacube_and_zip.py',
        description = (
            'Script to create a datacube of S2L1C data for given shapefile '
            'or s2grid id, and daterange. The script also runs s2cloudless to '
            'compute cloud probability if requested, and performs median '
            'mosaicing. The datacube is then saved into the specified folderpath '
            'and zipped.'    
        ),
        epilog = f"--- Send your complaints to {','.join(config.MAINTAINERS)} ---",
    )
    parser.add_argument('roi', help='filepath=path/to/shapefile | s2gridid=S2GridID')
    parser.add_argument('startdate', help='YYYY-MM-DD')
    parser.add_argument('enddate', help='YYYY-MM-DD')
    parser.add_argument('bands', default='all', help='all, or bands comma separated B02,B03,B04,B08 (default = all)')
    parser.add_argument('out', help='/path/to/output')
    parser.add_argument('--njobs', default=4, help='Number of cores to use. (default = 4)')
    parser.add_argument('-s2c', '--s2cloudless', action='store', required=False, help='Whether to run s2cloudless and if so at what chunk size.')
    parser.add_argument('-m', '--mosaic', action='store', required=False, help='Whether to perform median mosaicing and if so in intervals of how many days (most used interval: 20.')
    parser.add_argument('-c', '--cloud-threshold', action='store', default=0, required=False, help='The probabiliy threshold at and above which the pixel is considered cloud. Must be from 0 to 1.')
    parser.add_argument('--silent', action='store_true', help='To run the script without any print statements.')
    parser.add_argument('--ignore-missing-files', action='store_true', help='If there are missing files for requested region and date range, this option ignores the error and proceeds, except when there are no files present.')
    parser.add_argument('--warn-missing-files', action='store_true', help='If there are missing files for requested region and date range, this option raises a warning and proceeds, except when there are no files present.')
    args = parser.parse_args()

    if args.roi.startswith('filepath='):
        roi_filepath = args.roi.removeprefix('filepath=')
        shapes_gdf = gpd.read_file(roi_filepath)
    elif args.roi.startswith('s2gridid='):
        s2gridid = args.roi.removeprefix('s2gridid=')
        s2grid_geom = \
        rsutils.s2_grid_utils.get_grid_geometry_from_id(
            grid_id=s2gridid, scale_fact=1.1,
        )
        shapes_gdf = gpd.GeoDataFrame(
            data={'geometry':[s2grid_geom]}, 
            crs='epsg:4326',
        )
    else:
        raise ValueError(f"Invalid ROI input='{args.roi}'. Must start with 'filepath=' or 's2gridid='.")

    startdate = datetime.datetime.strptime(args.startdate, '%Y-%m-%d')
    
    enddate = datetime.datetime.strptime(args.enddate, '%Y-%m-%d')

    if args.bands == 'all':
        bands = [
            'B01', 'B02', 'B03', 'B04',
            'B05', 'B06', 'B07', 'B08',
            'B8A', 'B09', 'B10', 'B11',
            'B12',
        ]
    else:
        bands = args.bands.upper().split(',')

    zip_filepath = args.out.removesuffix('.zip') + '.zip'

    njobs = int(args.njobs)
    
    s2cloudless_chunksize = args.s2cloudless
    if s2cloudless_chunksize is not None:
        s2cloudless_chunksize = int(s2cloudless_chunksize)
        if s2cloudless_chunksize < 0:
            raise ValueError('s2cloudless_chunksize can not be negative.')
        
    cloud_threshold = float(args.cloud_threshold)
    if cloud_threshold < 0 or cloud_threshold > 1:
        raise ValueError('cloud_threshold must be from 0-1')
    
    mosaic_days = args.mosaic
    if mosaic_days is not None:
        mosaic_days = int(mosaic_days)
        if mosaic_days < 0:
            raise ValueError('mosaic_days can not be negative.')
        
    print_messages = not args.silent

    if_missing_files = 'raise_error'
    if args.ignore_missing_files:
        if_missing_files = None
    if args.warn_missing_files:
        if_missing_files = 'warn'

    if print_messages:
        print('--- inputs ---')
        print(f'roi: {args.roi}')
        print(f'startdate: {args.startdate}')
        print(f'enddate: {args.enddate}')
        print(f'bands: {bands}')
        print(f'out: {zip_filepath}')
        print(f'njobs: {njobs}')
        if s2cloudless_chunksize is not None:
            print(f's2cloudless_chunksize: {s2cloudless_chunksize}')
        if mosaic_days is not None:
            print(f'cloud_threshold: {cloud_threshold}')
            print(f'mosaic_days: {mosaic_days}')
        if if_missing_files is not None:
            print(f'if_missing_files: {if_missing_files}')

    main(
        shapes_gdf = shapes_gdf,
        startdate = startdate,
        enddate = enddate,
        bands = bands,
        zip_filepath = zip_filepath,
        njobs = njobs,
        s2cloudless_chunksize = s2cloudless_chunksize,
        cloud_threshold = cloud_threshold,
        mosaic_days = mosaic_days,
        print_messages = print_messages,
        if_missing_files = if_missing_files,
    )

    end_time = time.time()

    if print_messages:
        print(f'--- t_elapsed: {round(end_time - start_time, 2)} secs ---')
