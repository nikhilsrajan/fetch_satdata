import os
import numpy as np
import joblib
import argparse

import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import Affine

from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles

import rsutils.modify_bands


MODEL_FILEPATH = '/Users/nikhilsrajan/NASA-Harvest/project/fetch_satdata/data/at_eurocrops/rf_model.joblib'

NODATA = 255
DTYPE = 'uint8'

def modify_datacube(datacube, metadata):
    sentinel_2_l2a_bands = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B11', 'B12']

    bands = metadata.get('bands')
    band_indices = {band: index for index, band in enumerate(bands)}

    modified_datacube, modified_band_indices = \
    rsutils.modify_bands.modify_bands(
        bands = np.expand_dims(datacube, axis=(0)).astype(float),
        band_indices = band_indices,
        sequence = [
            (rsutils.modify_bands.mask_invalid_and_interpolate, {}),
            (rsutils.modify_bands.compute_bands, dict(bands_to_compute = ['NDVI', 'NDRE', 'GCVI', 'SAVI'])),
            (rsutils.modify_bands.remove_bands, dict(bands_to_remove = sentinel_2_l2a_bands))
        ]
    )

    modified_datacube = np.squeeze(modified_datacube)

    return modified_datacube


def get_model_output(modified_datacube):
    clf, le = joblib.load(MODEL_FILEPATH)

    n_ts, n_h, n_w, n_bands = modified_datacube.shape
    X = modified_datacube.reshape(n_ts, n_h*n_w, n_bands).swapaxes(0, 1).reshape(n_h*n_w, n_ts*n_bands)

    nonnan_indices = np.where(~np.isnan(X).any(axis = 1))
    nonnan_X = X[nonnan_indices]

    output_flattened = np.full(shape=(n_h*n_w), fill_value=NODATA, dtype=DTYPE)
    output_flattened[nonnan_indices] = clf.predict(nonnan_X)

    output = output_flattened.reshape(n_h, n_w)

    return output


def write_cog(array: np.ndarray, transform: Affine, crs, out_path: str, nodata=NODATA, dtype=DTYPE):
    if array.ndim == 2:
        array = array[np.newaxis, ...]  # (bands, height, width)

    bands, height, width = array.shape
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": bands,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
    }

    with MemoryFile() as memfile:
        with memfile.open(**profile) as dst:
            dst.write(array)

        cog_translate(
            memfile.name,
            out_path,
            cog_profiles.get("deflate"),
            in_memory = False,
            quiet = True,
            overview_level = 5,              # force pyramid levels
            overview_resampling = "nearest", # or bilinear/cubic
        )


def main(datacube_folderpath, output_filepath):
    datacube_filepath = os.path.join(datacube_folderpath, 'datacube.npy')
    metadata_filepath = os.path.join(datacube_folderpath, 'metadata.pickle.npy')

    datacube = np.load(datacube_filepath)
    metadata = np.load(metadata_filepath, allow_pickle=True)[()]
    
    modified_datacube = modify_datacube(datacube=datacube, metadata=metadata)

    output = get_model_output(modified_datacube=modified_datacube)

    write_cog(
        array = output,
        transform = metadata['geotiff_metadata']['transform'],
        crs = metadata['geotiff_metadata']['crs'],
        out_path = output_filepath,
        nodata = NODATA,
        dtype = DTYPE,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog = 'python demo_model_deploy.py',
        description = "Demo model deploy.",
    )

    parser.add_argument('datacube_folderpath', help="Folderpath containing datacube.npy and metadata.pickle.npy")
    parser.add_argument('output_filepath', help="Filepath where the output is to be stored.")

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_filepath), exist_ok=True)

    main(
        datacube_folderpath = args.datacube_folderpath,
        output_filepath = args.output_filepath,
    )

