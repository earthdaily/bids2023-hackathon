import datetime
from typing import List, Tuple

import dash
import dash_leaflet as dl
from dash_extensions.enrich import Input, Output

from app.app import app
from app.layers import base_layer, venus_layer
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
        Input(component_id="color-comparison-url", component_property="data"),
    ],
)
def render_layers(
    color_bounds, color_baseline_url, color_comparison_url
) -> Tuple[List, List, None]:
    """
    This controls the rendering of layers in the store
    """
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    app.logger.info(f"Rendering map...")

    if color_baseline_url is not None:
        st = datetime.datetime.now()

        layers = [
            dl.Overlay(
                dl.ImageOverlay(
                    className="img",
                    id="color-baseline",
                    url=color_baseline_url,
                    bounds=color_bounds,
                ),
                name="True Color Baseline",
                checked=True,
            ),
            dl.Overlay(
                dl.ImageOverlay(
                    className="img",
                    id="color-comparison",
                    url=color_comparison_url,
                    bounds=color_bounds,
                ),
                name="True Color Comparison",
                checked=True,
            ),
        ]

        et = datetime.datetime.now()
        delta = (et - st).seconds
        app.logger.info(f"Map drawn - {delta}s")
        app.logger.info(f"Bounds - {color_bounds}")

        return [base_layer] + layers, [venus_layer], dash.no_update
    else:
        app.logger.info(f"Not drawing any map layers because no urls available.")

        return [base_layer], [venus_layer], dash.no_update
