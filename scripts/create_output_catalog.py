import os

CUR_DIR = os.path.dirname(__file__)

import pandas as pd

import sys
sys.path.append(os.path.join(CUR_DIR, '..'))

import rsutils.utils


ROOT_FOLDERPATH = '/gpfs/data1/cmongp1/nair/decifr/malawi_250617_bt/inference/99_fsl_20250620-100316_4e434211-ab93-4bb8-b7da-f03218e51cbd/'
CATALOG_FILEPATH = '/gpfs/data1/cmongp2/sasirajann/fetch_satdata/data/malawi/output_catalog.csv'


filepaths = rsutils.utils.get_all_files_in_folder(
    folderpath = ROOT_FOLDERPATH,
    keep_extensions = ['.tif']
)


data = {
    'filepath': [],
    'daterange': [],
    'roi': [],
}


for filepath in filepaths:
    prefix_rm_filepath = filepath.removeprefix(ROOT_FOLDERPATH)
    daterange, roi, filename = prefix_rm_filepath.split('/')
    data['daterange'].append(daterange)
    data['roi'].append(roi)
    data['filepath'].append(filepath)


pd.DataFrame(data = data).to_csv(CATALOG_FILEPATH, index=False)
