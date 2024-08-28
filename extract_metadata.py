import os
import geopandas as gpd
import datetime
import xml.etree.ElementTree as ET
import pandas as pd
import tqdm

import create_stack


def extract_s2l1c_mean_sun_angle(
    shapes_gdf:gpd.GeoDataFrame,
    catalog_filepath:str,
    startdate:datetime.datetime,
    enddate:datetime.datetime,
):
    METADATA_FILENAME = 'MTD_TL.xml'

    catalog_gdf = create_stack.filter_catalog(
        catalog_filepath = catalog_filepath,
        shapes_gdf = shapes_gdf,
        startdate = startdate,
        enddate = enddate,
    )

    data = {
        'id': [],
        'filepath': [],
        'azimuth (deg)': [],
        'zenith (deg)': [],
    }

    for _id, local_folderpath, files in tqdm.tqdm(zip(
        catalog_gdf['id'],
        catalog_gdf['local_folderpath'],
        catalog_gdf['files'],
    ), total=catalog_gdf.shape[0]):
        filenames_of_interest = set(files.split(',')) & {METADATA_FILENAME}

        data['id'].append(_id)

        # MTD_TL.xml was not downloaded for id
        if len(filenames_of_interest) == 0:
            data['filepath'].append(None)
            data['azimuth (deg)'].append(None)
            data['zenith (deg)'].append(None)
            continue
        
        filepath = os.path.join(local_folderpath, METADATA_FILENAME)
        data['filepath'].append(filepath)

        root = ET.parse(filepath).getroot()

        azimuth_angle_element = root.findall('.//Mean_Sun_Angle/AZIMUTH_ANGLE')[0]
        azimuth_angle_units = azimuth_angle_element.attrib['unit']
        if azimuth_angle_units != 'deg':
            raise ValueError(f"azimuth_angle_units={azimuth_angle_units}, which is not 'deg' as it was expected.")
        azimuth_angle_value = float(azimuth_angle_element.text)
       
        zenith_angle_element = root.findall('.//Mean_Sun_Angle/ZENITH_ANGLE')[0]
        zenith_angle_units = zenith_angle_element.attrib['unit']
        if zenith_angle_units != 'deg':
            raise ValueError(f"zenith_angle_units={zenith_angle_units}, which is not 'deg' as it was expected.")
        zenith_angle_value = float(zenith_angle_element.text)
        
        data['azimuth (deg)'].append(azimuth_angle_value)
        data['zenith (deg)'].append(zenith_angle_value)

    return pd.DataFrame(data=data)

