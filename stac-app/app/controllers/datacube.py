import datetime
from typing import Dict, Tuple, Union, List

import dash
import pandas as pd
import xarray as xr
import rioxarray as rio  # noqa
from dash_extensions.enrich import Input, Output, ServersideOutput

from app.app import app
from app.dal.stac import get_stac_items


@app.callback(
    [
        ServersideOutput(component_id="dataset-store", component_property="data"),
        ServersideOutput(component_id="dates-store", component_property="data"),
        Output(component_id="loading-data-cube", component_property="children"),
    ],
    [
        Input(component_id="search-btn", component_property="n_clicks"),
        Input(component_id="edit-control", component_property="geojson"),
        Input(component_id="date-picker-range", component_property="start_date"),
        Input(component_id="date-picker-range", component_property="end_date"),
    ],
)
def get_datacube(
    search_trigger, search_geom, start_date, end_date
) -> Tuple[Union[xr.Dataset, None], Union[Dict, None], str]:
    """
    Generates a dataset for the selected query
    """

    ctx = dash.callback_context
    app.logger.info("Dataset search triggered...")
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    dates = list()
    if search_geom is not None and start_date is not None and end_date is not None:
        if trigger_id in ["search-btn"]:
            app.logger.info("Getting data triggered by geom+start-date+end-date")
            app.logger.info(start_date)
            app.logger.info(end_date)

            st = datetime.datetime.now()
            data = get_stac_items(search_geom, start_date, end_date)
            time_values = data.date.values
            for time in time_values:
                dates.append(
                    {"label": pd.Timestamp(time).strftime("%Y-%m-%d"), "value": time}
                )

            cube_epsg = f"EPSG:{data.epsg.values}"
            data = data.rio.write_crs(cube_epsg)

            et = datetime.datetime.now()
            delta = (et - st).seconds
            app.logger.info(f"Data loaded - {delta}s")

            return data, dates, f"Data cube loaded in {delta}s"
        else:
            if len(search_geom["features"]) != 0:
                return dash.no_update, dash.no_update, "Search ready"

            return (
                dash.no_update,
                dash.no_update,
                "Please define start date, end date, and geometry.",
            )
    else:
        app.logger.info(f"Search not ready: some data fields not filled.")
        return (
            dash.no_update,
            dash.no_update,
            "Please define start date, end date, and geometry.",
        )


@app.callback(
    [
        Output(component_id="date-selector-baseline", component_property="options"),
    ],
    [Input(component_id="dates-store", component_property="data")],
)
def get_baseline_times(dates) -> List[Dict]:
    app.logger.info("Get baseline times triggered.")
    app.logger.info(dates)

    return dates


@app.callback(
    [
        Output(component_id="date-selector-comparison", component_property="options"),
    ],
    [
        Input(component_id="date-selector-baseline", component_property="options"),
        Input(component_id="date-selector-baseline", component_property="value"),
    ],
)
def get_comparison_times(dates, baseline_date) -> Tuple[Dict, None]:
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    app.logger.info("Get comparison times triggered.")
    app.logger.info(dates)
    app.logger.info(baseline_date)

    comparison_dates = list()

    for date in dates:
        if date["value"] != baseline_date:
            comparison_dates.append(date)

    return comparison_dates
