from typing import List

import geopandas as gpd
import numpy as np
import xarray as xr
from shapely.geometry import Polygon


def get_selected_utm_epsg(aoi) -> int:
    """
    Get the closest UTM EPSG code for given bounds.
    """
    if isinstance(aoi, Polygon):
        minx, miny, maxx, maxy = aoi.bounds
    else:
        minx, miny, maxx, maxy = aoi
    lon = np.mean([minx, maxx])
    lat = np.mean([miny, maxy])

    # Calculation from https://stackoverflow.com/a/40140326/4556479
    utm_band = str((np.floor((lon + 180) / 6) % 60) + 1)
    if len(utm_band) == 1:
        utm_band = "0" + utm_band
    if lat >= 0:
        epsg_code = "326" + utm_band
        return int(epsg_code)
    epsg_code = "327" + utm_band
    return int(epsg_code)


def find_modal_epsg(epsg_codes):
    epsg_code_dict = dict()
    for code in epsg_codes:
        if code not in epsg_code_dict.keys():
            epsg_code_dict[code] = 1
        else:
            epsg_code_dict[code] += 1

    keys = list(epsg_code_dict.keys())
    frequency = [epsg_code_dict[code] for code in keys]
    modal_epsg_code = keys[np.argmax(frequency)]

    return modal_epsg_code
