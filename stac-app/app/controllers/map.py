import datetime
from typing import List, Tuple

import dash
import dash_leaflet as dl
from dash_extensions.enrich import Input, Output

from app.app import app
from app.layers import base_layer
from app import utils


@app.callback(
    [
        Output(component_id="raster-layers", component_property="children"),
        Output(component_id="vector-layers", component_property="children"),
        Output(component_id="loading-render", component_property="children"),
    ],
    [
        Input(component_id="color-bounds", component_property="data"),
        Input(component_id="color-baseline-url", component_property="data"),
    ],
)
def render_layers(color_bounds, color_url) -> Tuple[List, List, None]:
    """
    This controls the rendering of layers in the store
    """
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    app.logger.info(f"rendering map")

    if color_url is not None:
        st = datetime.datetime.now()

        layers = [
            dl.Overlay(
                dl.ImageOverlay(
                    className="img",
                    id="true-color-baseline",
                    url=color_url,
                    bounds=color_bounds,
                ),
                name="True Color Baseline",
                checked=True,
            ),
        ]

        et = datetime.datetime.now()
        delta = (et - st).seconds
        app.logger.info(f"Map drawn - {delta}s")

        return [base_layer] + layers, [], dash.no_update
    else:
        app.logger.info(f"Not drawing any map layers because no urls available.")

        return [base_layer], [], dash.no_update
