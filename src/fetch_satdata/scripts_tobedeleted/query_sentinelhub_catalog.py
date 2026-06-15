import os
import geopandas as gpd
import sentinelhub
import datetime
import argparse

import config
from init_logger import get_logger

import sys
sys.path.append(os.path.join(config.CURRENT_DIRECTORY, '..'))

import cdseutils.mydataclasses
import cdseutils.utils
import cdseutils.constants


VALID_SATELLITES = [
    cdseutils.constants.Bands.S2L1C.NAME,
    cdseutils.constants.Bands.S2L2A.NAME,
]


logger = get_logger()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog = 'python query_sentinelhub_catalog.py',
        description = (
            'Script to query sentinelhub catalog for a given satellite, '
            'startdate, enddate, and shapefile.'
        ),
        epilog = f"--- Send your complaints to {','.join(config.MAINTAINERS)} ---",
    )

    parser.add_argument('cdse_creds_json', help='/path/to/cdse_credentials.json')
    parser.add_argument('satellite', help=f'Satellite source to make the query for. Options: {VALID_SATELLITES}')
    parser.add_argument('roi_filepath', help='/path/to/shapefile')
    parser.add_argument('startdate', help='YYYY-MM-DD')
    parser.add_argument('enddate', help='YYYY-MM-DD')
    parser.add_argument('export_filepath', help='Filepath where the queried catalog should be saved')
    parser.add_argument('--cache', action='store', required=False, default=None, help='Folderpath to cache the sentinelhub catalog queries.')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()

    cdse_creds = cdseutils.utils.cdse_credentials_from_json(str(args.cdse_creds_json))

    satellite = str(args.satellite)

    if satellite not in VALID_SATELLITES:
        raise ValueError(f"Invalid satellite='{satellite}'. Valid options: {VALID_SATELLITES}")
    
    collection = {
        cdseutils.constants.Bands.S2L1C.NAME : sentinelhub.DataCollection.SENTINEL2_L1C,
        cdseutils.constants.Bands.S2L2A.NAME : sentinelhub.DataCollection.SENTINEL2_L2A,
    }[satellite]
    
    roi_filepath = str(args.roi_filepath)

    shapes_gdf = gpd.read_file(roi_filepath)

    startdate = datetime.datetime.strptime(str(args.startdate), '%Y-%m-%d')
    enddate = datetime.datetime.strptime(str(args.enddate), '%Y-%m-%d')

    export_filepath = str(args.export_filepath)

    cache_folderpath = None
    if args.cache is not None:
        cache_folderpath = str(args.cache)
    else:
        logger.warning(
            "It is advised to use --cache option to reduce the number of "
            "queries made to the sentinelhub API and thus the credit consumption."
        )

    if args.verbose:
        logger.setLevel('INFO')

    logger.info('Input parameters:')
    logger.info('-----------------')
    logger.info(f"roi_filepath : {roi_filepath}")
    logger.info(f"startdate : {startdate.strftime('%Y-%m-%d')}")
    logger.info(f"enddate : {enddate.strftime('%Y-%m-%d')}")
    logger.info(f"satellite : {satellite}")
    logger.info(f"export_filepath : {export_filepath}")

    if cache_folderpath is not None:
        logger.info(f"cache_folderpath : {cache_folderpath}")

    catalog_gdf = cdseutils.utils.query_catalog(
        shapes_gdf = shapes_gdf,
        sh_creds = cdse_creds.sh_creds,
        collection = collection,
        startdate = startdate,
        enddate = enddate,
        cache_folderpath = cache_folderpath,
    )

    logger.info(f"Number of tiles found: {catalog_gdf.shape[0]}")

    catalog_gdf['satellite'] = satellite

    catalog_gdf.to_file(export_filepath)
