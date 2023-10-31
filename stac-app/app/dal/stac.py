import json
from typing import Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import planetary_computer as pc
import stackstac
import xarray as xr
from pystac_client import Client

from app.config import STAC_API_URL
from app.utils import convert_wgs_to_utm
from app.app import app
from shapely.geometry import Polygon


class SCLMaskLabel:
    # Note: This is for Sentinel2 L2A only
    NO_DATA = 0
    SATURED_OR_DEFECTED = 1
    DARK_AREA_PIXELS = 2
    CLOUD_SHADOWS = 3
    VEGETATION = 4
    NOT_VEGETATED = 5
    WATER = 6
    UNCLASSIIFED = 7
    CLOUD_MEDIUM_PROBABILITY = 8
    CLOUD_HIGH_PROBABILITY = 9
    THIN_CIRRUS = 10
    SNOW = 11


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


def get_stac_items(geom, start_date, end_date) -> xr.Dataset:
    catalog = Client.open(STAC_API_URL)
    geometry = gpd.GeoDataFrame.from_features(geom["features"], crs="EPSG:4326")

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects=geom["features"][0]["geometry"],
        datetime=f"{start_date}/{end_date}",
    )

    array_bounds = geometry.unary_union.bounds
    app.logger.info(array_bounds)

    epsg_codes = list()
    items = [item.to_dict() for item in search.get_items()]
    for item in items:
        epsg = item["properties"].get("proj:epsg", "") or get_selected_utm_epsg(
            item["bbox"]
        )
        epsg_codes.append(epsg)

    epsg = find_modal_epsg(epsg_codes)
    app.logger.info(f"Number of items returned: {len(items)}")

    app.logger.info(items)
    app.logger.info(epsg)
    app.logger.info(array_bounds)
    stacked_data = stackstac.stack(
        items,
        epsg=epsg,
        bounds_latlon=array_bounds,
        resolution=10,
        assets=[
            "B02",
            "B03",
            "B04",
            "B05",
            "B06",
            "B07",
            "B08",
            "B8A",
            "B09",
            "B11",
            "B12",
            "SCL",
        ],
    )

    data = stacked_data.to_dataset()
    data = data.rename({[i for i in data.data_vars][0]: "msi"})
    clean_ind = ~pd.Index(data.time).duplicated(keep="first")
    data = data.sel({"time": clean_ind})

    """
    NODATA = 0
    MASK_BAND_NAME = "SCL"

    data = data.where(data > NODATA)

    scl_band = data.sel(band=[MASK_BAND_NAME])

    mask_values = np.logical_or(
        scl_band == SCLMaskLabel.CLOUD_SHADOWS,
        scl_band >= SCLMaskLabel.CLOUD_MEDIUM_PROBABILITY,
    )

    data = data.where(mask_values.squeeze() == 0)
    app.logger.info(data.time)
    """
    temporally_agg_data_array = data.groupby('time.date').min()
    app.logger.info(temporally_agg_data_array.date)

    temporally_agg_data_array.attrs["crs"] = stacked_data.attrs["crs"]
    app.logger.info(temporally_agg_data_array.date)

    return temporally_agg_data_array
