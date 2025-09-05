import argparse
import geopandas as gpd
import pandas as pd
import os
import tqdm
import multiprocessing as mp
import functools

import config

import sys
sys.path.append('..')

import rsutils.utils


COL_ID = 'id'
COL_LABEL = 'label'

NODATA_PLANET = 0


def setup_single(
    shape_gdf:gpd.GeoDataFrame,
    catalog_gdf:gpd.GeoDataFrame,
    startdate:pd.Timestamp,
    enddate:pd.Timestamp,
    timestamp_col:str,
    # filepath_col:str,
    run_folderpath:str,
    # check_cropability:bool = False,
    # raise_error:bool = True,
):
    # forced convertion of invalid geometry to valid
    shape_gdf['geometry'] = shape_gdf['geometry'].buffer(0)

    daterange_catalog_gdf = catalog_gdf[
        (catalog_gdf[timestamp_col] >= startdate) &
        (catalog_gdf[timestamp_col] <= enddate)
    ]
    filtered_catalog_gdf = daterange_catalog_gdf[
        daterange_catalog_gdf.intersects(
            shape_gdf.to_crs(catalog_gdf.crs)['geometry'][0]
        )
    ]

    images_count = filtered_catalog_gdf.shape[0]

    actual_startdate = filtered_catalog_gdf[timestamp_col].min()
    actual_enddate = filtered_catalog_gdf[timestamp_col].max()

    export_folderpath = os.path.join(
        run_folderpath, 
        f"{actual_startdate.strftime('%Y%m%d')}_{actual_enddate.strftime('%Y%m%d')}",
        str(shape_gdf[COL_ID][0])
    )
    shapefile_filepath = os.path.join(export_folderpath, 'geometry.geojson')
    subset_catalog_filepath = os.path.join(export_folderpath, 'catalog.geojson')

    os.makedirs(export_folderpath, exist_ok=True)

    shape_gdf.to_file(shapefile_filepath)
    filtered_catalog_gdf.to_file(subset_catalog_filepath)

    # cropped_shape = None
    # croppable = None
    # if check_cropability:
    #     try:
    #         cropped_imarray, cropped_meta = rsutils.utils.crop_tif(
    #             src_filepath = filtered_catalog_gdf[filepath_col].to_list()[0],
    #             shapes_gdf = shape_gdf,
    #             nodata = NODATA_PLANET,
    #             all_touched = True,
    #         )
    #         cropped_shape = cropped_imarray.shape
    #         del cropped_imarray, cropped_meta
    #     except Exception as e:
    #         if raise_error:
    #             raise e
    #         print('Crop Test Failed.')

    return os.path.abspath(shapefile_filepath), \
        actual_startdate, actual_enddate, \
        subset_catalog_filepath, \
        os.path.abspath(export_folderpath), \
        images_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = 'python setup_planetdatacube_run.py',
        description = (
            "Script to create input csv for create_planet_datacube.py snakemake run. "
            "Note that unlike 'generate_s2l2a_datacube_input_csv.py' there is no rigorous "
            "checks to see if the geometry repeats under a different unique id. "
            "If one uses the script with different set of geometries, but prexisting ids, and "
            "same export-folderpath, they would run the risk of overwriting the older datacubes. "
            "But since later snakemake is used, it would try not to redo whats already done, "
            "forcing the person to choose a new export-folderpath, which is great."
        ),
        epilog = f"--- Send your complaints to {','.join(config.MAINTAINERS)} ---",
    )

    parser.add_argument('catalog_filepath', help='Filepath to the catalog of satellite images.')
    parser.add_argument('shapefilepath', help="shapefile containing polygons along with a unique id for which to create input csv.")
    parser.add_argument('id_col', help="id column in the shapefile")
    parser.add_argument('timestamp_col', help="Timestamp column in the catalog")
    # parser.add_argument('filepath_col', help="Filepath column in the catalog")
    parser.add_argument('startdate', help='style: 2024-08-20 or 2024-08-20T13:54:40.022Z')
    parser.add_argument('enddate', help='style: 2024-08-20 or 2024-08-20T13:54:40.022Z')
    parser.add_argument('run_folderpath', help='Folderpath where the outputs are to be created.')
    parser.add_argument('--label_col', required=False, default=None, help="Label column in the shapefile")
    parser.add_argument('-j', '--njobs', required=False, action='store', default=1, help='[default = 1] Number of parallel jobs')
    parser.add_argument('--UTC', required=False, action='store_true', help='Add timezone to UTC')
    parser.add_argument('--mosaic-days', required=False, action='store', default=None, type=int, help='Option to add mosaic days. Recommended value = 20.')
    parser.add_argument('--scl-mask-classes', required=False, action='store', nargs='+', type=int, default=None, help='Values in SCL layer to be masked across bands.')

    args = parser.parse_args()

    catalog_gdf = gpd.read_file(args.catalog_filepath)
    shapes_gdf = gpd.read_file(args.shapefilepath)
    id_col = str(args.id_col)
    label_col = args.label_col
    timestamp_col = str(args.timestamp_col)
    # filepath_col = str(args.filepath_col)
    startdate = pd.to_datetime(str(args.startdate), utc=args.UTC)
    enddate = pd.to_datetime(str(args.enddate), utc=args.UTC)
    run_folderpath = str(args.run_folderpath)
    njobs = int(args.njobs)
    mosaic_days = args.mosaic_days
    scl_mask_classes = args.scl_mask_classes

    os.makedirs(run_folderpath, exist_ok=True)

    csv_filepath = os.path.join(run_folderpath, 'input.csv')

    shape_gdfs = [
        gpd.GeoDataFrame(data = {
            'geometry': [row['geometry']],
            COL_ID: [row[id_col]],
        }, crs=shapes_gdf.crs)
        for _, row in shapes_gdf.iterrows()
    ]

    if label_col is not None:
        labels = shapes_gdf[label_col].to_list()
        for i in range(len(shape_gdfs)):
            shape_gdfs[i][COL_LABEL] = labels[i]
 
    setup_single_partial = functools.partial(
        setup_single,
        catalog_gdf = catalog_gdf,
        startdate = startdate,
        enddate = enddate,
        timestamp_col = timestamp_col,
        # filepath_col = filepath_col,
        run_folderpath = run_folderpath,
        # check_cropability = True,
        # raise_error = False,
    )

    with mp.Pool(njobs) as p:
        outputs = list(tqdm.tqdm(
            p.imap(setup_single_partial, shape_gdfs), total=len(shape_gdfs)
        ))

    data = {
        'shapefilepath': [],
        'startdate': [],
        'enddate': [],
        'catalog_filepath': [],
        'export_folderpath': [],
        'datacube_filepath': [],
        'images_count': [],
        COL_ID: [],
    }

    if label_col is not None:
        data[COL_LABEL] = []

    i = 0
    for shape_filepath, actual_startdate, \
        actual_enddate, subcatalog_filepath, \
        export_folderpath, images_count in outputs:

        data['shapefilepath'].append(shape_filepath)
        data['startdate'].append(actual_startdate.strftime('%Y-%m-%d'))
        data['enddate'].append(actual_enddate.strftime('%Y-%m-%d'))
        data['catalog_filepath'].append(subcatalog_filepath)
        data['export_folderpath'].append(export_folderpath)
        data['datacube_filepath'].append(os.path.join(export_folderpath, 'datacube.npy'))
        data['images_count'].append(images_count)
        # data['cropped_shape'].append(cropped_shape)

        data[COL_ID].append(shape_gdfs[i][COL_ID][0])
        if label_col is not None:
            data[COL_LABEL].append(shape_gdfs[i][COL_LABEL][0])

        i += 1
    
    input_df = pd.DataFrame(data=data)
    input_df['added_on'] = pd.Timestamp.now(tz='UTC')
    
    if mosaic_days is not None:
        input_df['mosaic_days'] = mosaic_days
    if scl_mask_classes is not None:
        input_df['scl_mask_classes'] = ','.join([str(v) for v in scl_mask_classes])

    if os.path.exists(csv_filepath):
        existing_df = pd.read_csv(csv_filepath)
        input_df = pd.concat([existing_df, input_df])

    input_df.to_csv(csv_filepath, index=False)
