import rasterio
import rasterio.features
import geopandas as gpd
import glob
import os
import shapely
import tqdm
import multiprocessing as mp


def get_all_tif_filepaths(
    folderpath:str,
):
    # https://stackoverflow.com/questions/18394147/how-to-do-a-recursive-sub-folder-search-and-return-files-in-a-list
    filepaths =  [
        os.path.abspath(y) for x in os.walk(folderpath) 
        for y in glob.glob(os.path.join(x[0], '*')) 
        if os.path.isfile(y)
    ]
    filepaths = [
        fp for fp in filepaths if fp.endswith('.tif')
    ]
    return filepaths


def get_raster_details(raster_filepath, alpha_band='alpha', alpha_nodata=0):
    with rasterio.open(raster_filepath) as src:
        bands = src.descriptions
        desc_index = {desc: i for i, desc in enumerate(bands, 1)}
        data = src.read(desc_index[alpha_band])
        crs = src.crs
        meta = src.meta.copy()

    results = rasterio.features.shapes(data, data!=alpha_nodata, transform=meta['transform'])
    geom = shapely.unary_union([shapely.geometry.shape(geom) for geom, _ in results])
    geom = geom.simplify(100, preserve_topology=True)
    
    return geom, crs, bands


def get_info(raster_filepath):
    startdate_str, enddate_str = raster_filepath.split('/')[-3].split('_')
        
    geom, crs, bands = get_raster_details(
        raster_filepath = raster_filepath,
    )

    return startdate_str, enddate_str, geom, crs, bands, raster_filepath


if __name__ == '__main__':
    root_folderpath = '/gpfs/data1/cmongp2/PlanetData/malawi/ps_biweekly_8band/'
    planetcatalog_geojson_filepath = '/gpfs/data1/cmongp2/PlanetData/malawi/planetdata_catalog.geojson'
    NJOBS = 100

    tif_filepaths = get_all_tif_filepaths(
        folderpath = root_folderpath
    )

    data = {
        'filepath': [],
        'startdate': [],
        'enddate': [],
        'bands': [],
        'geometry': [],
    }

    with mp.Pool(NJOBS) as p:
        results = list(tqdm.tqdm(
            p.imap(get_info, tif_filepaths), 
            total=len(tif_filepaths)
        ))

    crs = None
    for startdate_str, enddate_str, geom, _crs, bands, raster_filepath in results:
        if crs is None:
            crs = _crs
        elif crs != _crs:
            raise ValueError(f'CRS mismatch: {crs} != {_crs}')

        data['startdate'].append(startdate_str)
        data['enddate'].append(enddate_str)
        data['bands'].append(','.join(bands))
        data['filepath'].append(raster_filepath)
        data['geometry'].append(geom)

    planet_catalog_gdf = gpd.GeoDataFrame(data = data, crs = crs)

    planet_catalog_gdf['tile_number'] = planet_catalog_gdf['filepath'].apply(
        lambda x: x.split('/')[-1].split('.')[0]
    )

    for tile_number in planet_catalog_gdf['tile_number'].unique():
        tile_catalog_gdf = planet_catalog_gdf[
            planet_catalog_gdf['tile_number'] == tile_number
        ]
        max_area_index = tile_catalog_gdf.area.argmax()
        max_area_geometry = tile_catalog_gdf.iloc[max_area_index]['geometry']
        planet_catalog_gdf.loc[tile_catalog_gdf.index, 'geometry'] = max_area_geometry

    planet_catalog_gdf.to_file(planetcatalog_geojson_filepath)
