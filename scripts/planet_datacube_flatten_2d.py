import argparse
import geopandas as gpd
import tqdm
import numpy as np
import os
import rasterio.transform

import config


PLANET_NODATA = 0


def get_easting_northing_array(
    width:int,
    height:int,
    transform
):
    xs = np.arange(width)
    ys = np.arange(height)
    X, Y = np.meshgrid(xs, ys)
    XY_flat = np.column_stack((X.ravel(), Y.ravel()))
    eastings, northings = rasterio.transform.xy(
        transform = transform, 
        rows = XY_flat[:,1], 
        cols = XY_flat[:,0], 
        offset = 'center',
    )
    eastings_northings = np.array([(e, n) for e, n in zip(eastings, northings)])
    easting_northing_array = eastings_northings.reshape(*X.shape, 2)
    return easting_northing_array


def read_datacube_convert_to_2d(
    filepath:str,
    nodata = 0,
):
    datacube = np.load(filepath)

    metadata_filepath = os.path.join(os.path.dirname(filepath), 'metadata.pickle.npy')

    nt, nh, nw, nb = datacube.shape # checking if datacube is 4d

    # finding height and width indices whose nodata sum across time and band is less than nt * nb
    # basically height and width pixels which are not completely nodata
    h_indices, w_indices = np.where((datacube == nodata).sum(axis=(0, 3)) < nt*nb)

    flattened_datacube = datacube[:, h_indices, w_indices, :].swapaxes(0, 1)
    n_pixels, _, _ = flattened_datacube.shape

    metadata = np.load(metadata_filepath, allow_pickle=True)[()]
    geotiff_metadata = metadata['geotiff_metadata']
    EN = get_easting_northing_array(
        width = geotiff_metadata['width'],
        height = geotiff_metadata['height'],
        transform = geotiff_metadata['transform'],
    )

    flattened_coords = EN[h_indices, w_indices, :]

    return flattened_datacube, flattened_coords


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog = 'python planet_datacube_agg_2d.py',
        description = (
            'Script to aggregate planet datacubes into a single numpy array. '
            'The purpose of this is to convert datacubes of farm level polygons '
            'into data that could be use for training a model. The user passes '
            'a file with the datacube ids they want to aggregate, a list of '
            'operations to be run in sequence, and the export folderpath where a '
            'data.npy, ids.npy, and labels.npy would be saved.'
        ),
        epilog = f"--- Send your complaints to {','.join(config.MAINTAINERS)} ---",
    )
    parser.add_argument('filepath', action='store', help='Filepath of the file containing datacube filepaths, along with ids, and optional labels. Can be geojson or csv.')
    parser.add_argument('filepath_col', action='store', help='Column in the file containing filepaths to the datacubes.')
    parser.add_argument('id_col', action='store', help='Column in the file containing ids.')
    parser.add_argument('export_folderpath', action='store', help='Folderpath to save the outputs to.')
    parser.add_argument('-l', '--label_col', action='store', default=None, required=False, help='Column in the file containing ')
    
    args = parser.parse_args()

    filepaths_df = gpd.read_file(args.filepath)
    filepath_col = args.filepath_col
    id_col = args.id_col
    label_col = args.label_col

    os.makedirs(args.export_folderpath, exist_ok=True)

    datacube_2d_list = []
    coords_2d_list = []
    id_list = []
    if label_col is not None:
        label_list = []

    for index, row in tqdm.tqdm(filepaths_df.iterrows(), 
                                total=filepaths_df.shape[0]):
        flattened_datacube, flattened_coords = read_datacube_convert_to_2d(
            filepath = row[filepath_col], 
            nodata = PLANET_NODATA,
        )
        datacube_2d_list.append(flattened_datacube)
        coords_2d_list.append(flattened_coords)

        n_pixels, _, _ = flattened_datacube.shape

        del flattened_datacube

        id_list += [row[id_col] for _ in range(n_pixels)]

        if label_col is not None:
            label_list += [row[label_col] for _ in range(n_pixels)]
    
    datacubes_2d = np.concatenate(datacube_2d_list, axis=0)
    coords_2d = np.concatenate(coords_2d_list, axis=0)
    ids = np.array(id_list)

    if label_col is not None:
        labels = np.array(label_list)

    np.save(os.path.join(args.export_folderpath, 'data.npy'), datacubes_2d)
    np.save(os.path.join(args.export_folderpath, 'coords.npy'), coords_2d)
    np.save(os.path.join(args.export_folderpath, 'ids.npy'), ids)

    if label_col is not None:
        np.save(os.path.join(args.export_folderpath, 'labels.npy'), labels)
