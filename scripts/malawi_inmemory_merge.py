import os
import pandas as pd
import rasterio.merge

import sys
sys.path.append('..')

import rsutils.utils
import rsutils.modify_images


FILEPATHS_CSV = '/gpfs/data1/cmongp1/nair/decifr/malawi_250812_v4_ft/inference/99_fsl_20250901-125337_d4a5f023-8910-4662-891d-9d3302cf5944/output_catalog.csv'
FILEPATHS_COL = 'masked_filepath'

NODATA = 255
NJOBS = 100

GROUPBY = 'daterange'
OUTPUT_FOLDERPATH = '/gpfs/data1/cmongp1/nair/decifr/malawi_250812_v4_ft/inference/99_fsl_20250901-125337_d4a5f023-8910-4662-891d-9d3302cf5944/masked+merged_outputs'

filepaths_df = pd.read_csv(FILEPATHS_CSV)

os.makedirs(OUTPUT_FOLDERPATH, exist_ok=True)

unique_group_ids = filepaths_df[GROUPBY].unique()

"""
NOTE: ASSUMING ALL IMAGES HAVE SAME CRS
"""

for group_id in unique_group_ids:
    print(f'Merging for {GROUPBY}={group_id}...')

    group_filepaths_df = filepaths_df[filepaths_df[GROUPBY] == group_id]

    data_profile_list = rsutils.modify_images.load_images(
        src_filepaths = group_filepaths_df[FILEPATHS_COL],
        njobs = NJOBS,
    )

    merged_profile = data_profile_list[0][1].copy()

    memfiles = rsutils.modify_images.images_to_memfiles(
        data_profile_list = data_profile_list,
    )

    merged_ndarray, merged_transform = rasterio.merge.merge(
        [memfile.open() for memfile in memfiles], 
        nodata = NODATA,
    )

    merged_profile.update({
        'count': merged_ndarray.shape[0],
        'height': merged_ndarray.shape[1],
        'width': merged_ndarray.shape[2],
        'transform': merged_transform,
        'compress': 'lzw',
    })

    merged_profile = rsutils.utils.driver_specific_meta_updates(
        meta=merged_profile, driver='GTiff'
    )

    with rasterio.open(
        os.path.join(OUTPUT_FOLDERPATH, f'{group_id}.tiff'), 
        'w', 
        **merged_profile,
    ) as dst:
        dst.write(merged_ndarray)

    del memfiles, data_profile_list
