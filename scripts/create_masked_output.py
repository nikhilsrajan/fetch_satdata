import os

CUR_DIR = os.path.dirname(__file__)

import pandas as pd
import geopandas as gpd
import rasterio
import numpy as np
import argparse

import sys
sys.path.append(os.path.join(CUR_DIR, '..'))

import rsutils.utils
import rsutils.modify_images


ESA_WORLDCOVER_CROPLAND = 40
OUTPUT_NODATA = 255

# TEST_ROI = '192047c'
# TEST_DATERANGE = '20221003_20231030'
# OUTPUT_FILENAME = 'output_698.tiff'



LULC_TIF_FILEPATH = f'/gpfs/data1/cmongp1/sasirajann/malawi/data/malawi_worldcover_2021.tif'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog = 'python create_masked_output.py',
        description = (
            "Urgent script for creating masked planetdata malawi output - masked with ESA WorldCover LULC"
        ),
    )

    parser.add_argument('input_filepath', help="Tif file that needs to be masked.")
    parser.add_argument('shapefilepath', help="ShapeFile of the ROI")
    parser.add_argument('output_filepath', help="Where to save the masked tif filepath")

    args = parser.parse_args()

    # output_tif_filepath = f'/gpfs/data1/cmongp1/nair/decifr/malawi_250617_bt/inference/99_fsl_20250620-100316_4e434211-ab93-4bb8-b7da-f03218e51cbd/{TEST_DATERANGE}/{TEST_ROI}/{OUTPUT_FILENAME}'
    # roi_shape_filepath = f'/gpfs/data1/cmongp1/sasirajann/malawi/inference_datacubes/{TEST_DATERANGE}/{TEST_ROI}/geometry.geojson'
    # masked_output_tif_filepath = f'/gpfs/data1/cmongp1/sasirajann/malawi/outputs/test/masked_output.tif'

    output_tif_filepath = args.input_filepath
    roi_shape_filepath = args.shapefilepath
    masked_output_tif_filepath = args.output_filepath

    os.makedirs(os.path.dirname(masked_output_tif_filepath), exist_ok=True)

    shapes_gdf = gpd.read_file(roi_shape_filepath)

    cropped_lulc_imarray, cropped_lulc_meta = \
    rsutils.utils.crop_tif(
        src_filepath = LULC_TIF_FILEPATH,
        shapes_gdf = shapes_gdf,
        all_touched = True,
    )

    with rasterio.open(output_tif_filepath) as src:
        output_imarray = src.read()
        output_meta = src.meta.copy()

    resampled_lulc_imarray, resampled_lulc_meta = \
    rsutils.modify_images.resample_by_ref_meta(
        data = cropped_lulc_imarray,
        profile = cropped_lulc_meta,
        ref_meta = output_meta,
    )

    noncrop_h_pixels, noncrop_w_pixels = np.where(resampled_lulc_imarray[0] != ESA_WORLDCOVER_CROPLAND)
    output_imarray[:, noncrop_h_pixels, noncrop_w_pixels] = OUTPUT_NODATA
    output_meta['nodata'] = OUTPUT_NODATA
    output_meta['compress'] = 'lzw'

    with rasterio.open(masked_output_tif_filepath, 'w', **output_meta) as dst:
        dst.write(output_imarray)

