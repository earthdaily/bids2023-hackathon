import os
from typing import Dict

import geopandas as gpd
import pandas as pd
import stackstac
import xarray as xr
from pystac_client import Client

from app.app import app
from app.config import STAC_API_URL
from app.constants import EOProductType
from app.utils import find_modal_epsg, get_selected_utm_epsg
from app.auth import get_new_token


def get_stac_items(
    collection_name: str, geom: Dict, start_date: str, end_date: str
) -> xr.Dataset:

    catalog = Client.open(
        os.getenv("EDS_API_URL"), headers={"Authorization": f"bearer {get_new_token()}"}
    )

    product_type = EOProductType(collection_name)
    app.logger.info(
        f'Query is: collection:{product_type.value} \n geom: {geom["features"][0]["geometry"]} \n time:{start_date}/{end_date}'
    )
    query_result = catalog.search(
        collections=[product_type.value],
        intersects=geom["features"][0]["geometry"],
        datetime=f"{start_date}/{end_date}",
    )

    items = [item.to_dict() for item in query_result.get_items()]
    app.logger.info(f"Number of items returned: {len(items)}")

    epsg_codes = list()
    for item in items:
        epsg = item["properties"].get("proj:epsg", "") or get_selected_utm_epsg(
            item["bbox"]
        )
        epsg_codes.append(epsg)

    epsg = find_modal_epsg(epsg_codes)

    geometry = gpd.GeoDataFrame.from_features(geom["features"], crs="EPSG:4326")
    array_bounds = geometry.unary_union.bounds

    stacked_data = stackstac.stack(
        items,
        epsg=epsg,
        bounds_latlon=array_bounds,
        resolution=product_type.RESOLUTION,
        rescale=False,
        assets=product_type.DEFAULT_ASSETS,
    )

    data = stacked_data.to_dataset()
    data = data.rename({[i for i in data.data_vars][0]: "msi"})
    clean_ind = ~pd.Index(data.time).duplicated(keep="first")
    data = data.sel({"time": clean_ind})

    temporally_agg_data_array = data.groupby("time.date").min()
    app.logger.info(temporally_agg_data_array.date)

    temporally_agg_data_array.attrs["crs"] = stacked_data.attrs["crs"]
    app.logger.info(temporally_agg_data_array.date)

    return temporally_agg_data_array
