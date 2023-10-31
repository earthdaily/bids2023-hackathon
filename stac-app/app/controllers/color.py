import datetime
import os

import dash
import geopandas as gpd
import rasterio
import rioxarray  # noqa
import xarray as xr
from dash_extensions.enrich import Input, Output, ServersideOutput

from app.app import app
from app.controllers.map import render_layers  # noqa
from app.exceptions import NotComputedError
from app.services import get_image_url

DEFAULT_RESPONSE = (
    None,
    None,
    None,
    None,
    None,
    None,
    "Please select a site followed by two dates.",
)


@app.callback(
    [
        ServersideOutput(component_id="color-bounds", component_property="data"),
        ServersideOutput(
            component_id="true-color-baseline-url", component_property="data"
        ),
        ServersideOutput(
            component_id="true-color-comparison-url", component_property="data"
        ),
        Output(component_id="loading-color", component_property="children"),
    ],
    [
        Input(component_id="dataset-store", component_property="data"),
        Input(component_id="date-picker-range", component_property="start_date"),
        Input(component_id="date-picker-range", component_property="end_date"),
    ],
)
def get_color_images(
    data: xr.Dataset, baseline_time_idx: int, comparison_time_idx: int
):
    """ """
    ctx = dash.callback_context
    app.logger.info("Color retrieval triggered...")
    app.logger.info(baseline_time_idx)
    app.logger.info(comparison_time_idx)

    # if triggered by site-selector, then wipe layers

    if not ctx.triggered:
        app.logger.info(
            "Returning default render urls because water detection not triggered."
        )
        return DEFAULT_RESPONSE

    else:
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id in ["site-selector"]:
        app.logger.info("Returning default render urls because site changed.")
        return DEFAULT_RESPONSE

    if (
        data is not None
        and baseline_time_idx is not None
        and comparison_time_idx is not None
    ):
        return DEFAULT_RESPONSE

        try:
            app.logger.info(
                f"water detection initiated with following settings: "
                f"{[detection_settings.__dict__[key] for key in detection_settings.__dict__.keys()]}"
            )

            st = datetime.datetime.now()

            (
                color_bounds,
                baseline_color_image,
                comparison_color_image,
            ) = get_color(data, baseline_time_idx, comparison_time_idx)

            baseline_color_url = get_image_url(baseline_color_image)
            comparison_color_url = get_image_url(comparison_color_image)

            et = datetime.datetime.now()
            delta = (et - st).seconds
            app.logger.info(f"Water data rendered - {delta}s")

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
                dash.no_update,
                dash.no_update,
                dash.no_update,
                "Error communicating with Microsoft servers.",
            )

    else:
        app.logger.info("Returning default map because triggered but data not ready.")

        return DEFAULT_RESPONSE


