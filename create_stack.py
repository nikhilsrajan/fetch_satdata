import geopandas as gpd
import datetime
import shapely.ops
import pandas as pd


def filter_catalog(
    catalog_filepath:str,
    shapes_gdf:gpd.GeoDataFrame,
    startdate:datetime,
    enddate:datetime,
):
    """
    Assumes catalog has following columns:
    - timestamp
    - geometry
    """
    catalog_gdf = gpd.read_file(catalog_filepath)

    union_shape = shapely.ops.unary_union(
        shapes_gdf.to_crs(catalog_gdf.crs)['geometry']
    )
    union_shape_gdf = gpd.GeoDataFrame(
        data = {'geometry': [union_shape]},
        crs = catalog_gdf.crs,
    )

    dt_filtered_catalog_gdf = catalog_gdf[
        (catalog_gdf['timestamp'] >= pd.Timestamp(startdate, tz='UTC'))
        & (catalog_gdf['timestamp'] <= pd.Timestamp(enddate, tz='UTC'))
    ].reset_index(drop=True)

    filtered_catalog_gdf = \
    gpd.overlay(
        dt_filtered_catalog_gdf,
        union_shape_gdf,
    )

    filtered_catalog_gdf['area_contribution'] = filtered_catalog_gdf['geometry'].apply(
        lambda x: x.area / union_shape.area * 100 # area contribution in terms of %
    )

    return filtered_catalog_gdf


def get_intersecting_band_filepaths(
    shapes_gdf:gpd.GeoDataFrame,
    catalog_filepath:str,
    startdate:datetime,
    enddate:datetime,
    bands:list[str],
    ext:str = '.jp2',
):
    """
    Assumes the catalog has following columns:
    - id
    - timestamp
    - local_folderpath
    - files
    """
    catalog_gdf = filter_catalog(
        catalog_filepath = catalog_filepath,
        shapes_gdf = shapes_gdf,
        startdate = startdate,
        enddate = enddate,
    )

    data = {
        'id': [],
        'timestamp': [],
        'band': [],
        'filepath': [],
        'area_contribution': [],
    }

    for _id, timestamp, local_folderpath, \
        files, area_contribution in zip(
        catalog_gdf['id'],
        catalog_gdf['timestamp'],
        catalog_gdf['local_folderpath'],
        catalog_gdf['files'],
        catalog_gdf['area_contribution'],
    ):
        filenames_of_interest = {
            xii + ext 
            for xii in {
                xi.removesuffix(ext) 
                for xi in files.split(',')
            } & set(bands)
        }
        
        # if no files of interest present, skip
        if len(filenames_of_interest) == 0:
            continue

        _filepaths = [
            local_folderpath + '/' + filename_of_interest
            for filename_of_interest in filenames_of_interest
        ]

        data['id'] += [_id for _ in range(len(_filepaths))]
        data['timestamp'] += [timestamp for _ in range(len(_filepaths))]
        data['area_contribution'] += [area_contribution for _ in range(len(_filepaths))]
        data['band'] += [
            filename_of_interest.removesuffix(ext)
            for filename_of_interest in filenames_of_interest
        ]
        data['filepath'] += _filepaths

        del _filepaths

    band_filepaths_df = pd.DataFrame(data=data).sort_values(
        by = ['timestamp', 'band'],
        ascending = [True, True],
    ).reset_index(drop=True)

    return band_filepaths_df
