import datetime

import dash
import numpy as np
import pandas as pd
import rioxarray  # noqa
import xarray as xr
import xrspatial.multispectral as ms
from dash_extensions.enrich import Input, Output, ServersideOutput

from app.app import app
from app.constants import EOProductType
from app.services import get_image_url

DEFAULT_RESPONSE = (
    None,
    None,
    None,
    "Select a baseline and comparison date to view images",
)


@app.callback(
    [
        ServersideOutput(component_id="color-bounds", component_property="data"),
        ServersideOutput(component_id="color-baseline-url", component_property="data"),
        ServersideOutput(
            component_id="color-comparison-url", component_property="data"
        ),
        Output(component_id="loading-color", component_property="children"),
    ],
    [
        Input(component_id="dataset-store", component_property="data"),
        Input(component_id="date-selector-baseline", component_property="value"),
        Input(component_id="date-selector-comparison", component_property="value"),
        Input(component_id="eds-collection-selector", component_property="value"),
    ],
)
def get_color_images(
    data: xr.Dataset, baseline_time: int, comparison_time: int, collection_name: str
):
    """ """
    ctx = dash.callback_context
    app.logger.info("Color retrieval triggered...")
    app.logger.info(baseline_time)
    app.logger.info(comparison_time)

    # if triggered by site-selector, then wipe layers

    if not ctx.triggered:
        app.logger.info("Returning default render urls because color not triggered.")
        return DEFAULT_RESPONSE

    if (
        data is not None
        and baseline_time is not None
        and comparison_time is not None
        and collection_name is not None
    ):
        try:
            app.logger.info(f"Color render initiated")
            product_type = EOProductType(collection_name)

            st = datetime.datetime.now()
            baseline_date = pd.Timestamp(baseline_time)
            baseline_date = datetime.date(
                baseline_date.year, baseline_date.month, baseline_date.day
            )

            comparison_date = pd.Timestamp(comparison_time)
            comparison_date = datetime.date(
                comparison_date.year, comparison_date.month, comparison_date.day
            )
            (
                color_bounds,
                baseline_color_image,
                comparison_color_image,
            ) = get_color(data, baseline_date, comparison_date, product_type)

            app.logger.info(f"Loading images...")

            baseline_color_url = get_image_url(baseline_color_image)
            comparison_color_url = get_image_url(comparison_color_image)

            et = datetime.datetime.now()
            delta = (et - st).seconds
            app.logger.info(f"Color data rendered - {delta}s")

            return (
                color_bounds,
                baseline_color_url,
                comparison_color_url,
                f"Analysis loaded in {delta}s ",
            )

        except RuntimeError as e:
            app.logger.info(str(e))
            return (
                dash.no_update,
                dash.no_update,
                dash.no_update,
                "Error communicating with servers.",
            )

    else:
        app.logger.info("Returning default map because triggered but data not ready.")

        return DEFAULT_RESPONSE


def get_color(data, baseline_date, comparison_date, product_type):
    baseline_color_image, color_map_bounds = compute_color_at_date(
        data, baseline_date, product_type
    )
    comparison_color_image, color_map_bounds = compute_color_at_date(
        data, comparison_date, product_type
    )

    return color_map_bounds, baseline_color_image, comparison_color_image


def compute_color_at_date(data, baseline_date, product_type):
    data_at_date = data.sel({"date": baseline_date})
    color = ms.true_color(
        data_at_date.sel({"band": product_type.RGB[0]}).msi,
        data_at_date.sel({"band": product_type.RGB[1]}).msi,
        data_at_date.sel({"band": product_type.RGB[2]}).msi,
    )

    color = color.to_dataset().rio.write_crs(data.rio.crs)
    color = color.transpose("band", "y", "x")
    color = color.rio.reproject("EPSG:4326")
    color = color.transpose("y", "x", "band")

    fill_value = color.true_color._FillValue

    color_image = color.true_color.values
    r_fill_index = color_image[:, :, 0] == fill_value
    g_fill_index = color_image[:, :, 1] == fill_value
    b_fill_index = color_image[:, :, 2] == fill_value
    fill_index = np.logical_and(
        np.logical_and(r_fill_index, g_fill_index), b_fill_index
    )
    # set no alpha on fill_value pixels
    color_image[fill_index, 3] = 0
    xmin, ymin, xmax, ymax = color.rio.bounds()
    color_map_bounds = [[ymin, xmin], [ymax, xmax]]

    return color_image, color_map_bounds
