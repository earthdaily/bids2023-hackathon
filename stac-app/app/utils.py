from typing import List

import geopandas as gpd
import numpy as np
import xarray as xr
from shapely.geometry import Polygon


def normalize(array: np.array, normalize_type="max") -> np.array:
    """
    Normalize an array

    Args:
        array (np.array):
        normalize_type (str): max or quantile

    Returns:
        normalized array
    """
    if normalize_type == "max":
        array_min, array_max = np.nanmin(array), np.nanmax(array)
        array = (array - array_min) / (array_max - array_min)

    elif normalize_type == "quantile":
        array_min, array_max = np.quantile(array, 0.02), np.quantile(array, 0.98)
        array[array > array_max] = array_max
        array[array < array_min] = array_min
        array = (array - array_min) / (array_max - array_min)
    else:
        return array

    return array


def calculate_index(data: xr.Dataset, ndiff, time=None) -> np.array:
    """
    Calculates Arbitrary Index

    Args:
        data (xarray.Dataset): dataset
        ndiff (str): index to calculate, one of [NWDI, MNDWI]
        time (numpy.datetime64): if not selected will calculate across all time.

    Returns:
        array (array)

    """
    if ndiff == "NDWI":
        if time is not None:
            green = data.msi.sel({"time": [time], "band": "B03"})
            nir = data.msi.sel({"time": [time], "band": "B08"})
        else:
            green = data.msi.sel({"band": "B03"})
            nir = data.msi.sel({"band": "B08"})

        array = (green - nir) / (green + nir)

    elif ndiff == "MNDWI":
        if time is not None:
            green = data.msi.sel({"time": [time], "band": "B03"})
            swir = data.msi.sel({"time": [time], "band": "B11"})
        else:
            green = data.msi.sel({"band": "B03"})
            swir = data.msi.sel({"band": "B11"})

        array = (green - swir) / (green + swir)

    else:
        raise NotImplementedError(f"{ndiff} is not implemented.")

    return array


def rescale_index(array: np.array, desired_min=0, desired_max=1) -> np.array:
    """
    Rescales index such as NDWI to within a specified range
    Args:
        array (array): array of floating point values
        desired_min (int): desired minimum value to scale to
        desired_max (int): desired maximum value to scale to
    Returns:
        y (array) array scaled between desired min and desired max
    """
    array_min, array_max = np.nanmin(array), np.nanmax(array)
    y = (array - array_min) / (array_max - array_min) * (
        desired_max - desired_min
    ) + desired_min

    return y


def convert_wgs_to_utm(lon: float, lat: float) -> str:
    """
    Predicts most likely utm zone based on a long-lat
    Args:
        lon:
        lat:

    Returns:

    """

    utm_band = str(int((np.floor((lon + 180) / 6) % 60) + 1))
    if len(utm_band) == 1:
        utm_band = "0" + utm_band
    if lat >= 0:
        epsg_code = "326" + utm_band
    else:
        epsg_code = "327" + utm_band

    return epsg_code


def bounds_to_wgs84(bounds: List, crs: str) -> List:
    """
    Converts UTM bounds to WGS84 bounds
    Args:
        bounds:
        crs:

    Returns:

    """
    xmin, ymin, xmax, ymax = (
        np.float64(bounds[0]),
        np.float64(bounds[1]),
        np.float64(bounds[2]),
        np.float64(bounds[3]),
    )
    bounds_poly = Polygon([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)])
    gdf = gpd.GeoDataFrame(geometry=[bounds_poly])
    gdf.crs = crs
    gdf = gdf.to_crs("EPSG:4326")
    bounds = gdf.iloc[0].geometry.bounds
    xmin, ymin, xmax, ymax = (
        np.float64(bounds[0]),
        np.float64(bounds[1]),
        np.float64(bounds[2]),
        np.float64(bounds[3]),
    )
    bounds = [[ymin, xmin], [ymax, xmax]]

    return bounds
