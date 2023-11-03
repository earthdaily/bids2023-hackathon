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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


DEFAULT_RESPONSE = (
    None,
    None,
    None,
    "Select a baseline and comparison date to view images",
)


@app.callback(
    [
        ServersideOutput(component_id="segmentation-bounds", component_property="data"),
        ServersideOutput(component_id="segmentation-baseline-url", component_property="data"),
        ServersideOutput(
            component_id="segmentation-comparison-url", component_property="data"
        ),
        Output(component_id="loading-segmentation", component_property="children"),
    ],
    [
        Input(component_id="segmentation-btn", component_property="n_clicks"),
        Input(component_id="n-classes-selector", component_property="value"),
        Input(component_id="dataset-store", component_property="data"),
        Input(component_id="date-selector-baseline", component_property="value"),
        Input(component_id="date-selector-comparison", component_property="value"),
        Input(component_id="eds-collection-selector", component_property="value"),
    ],
)
def get_segmentation_images(
    n_clicks: int, n_classes: int, data: xr.Dataset, baseline_time: int, comparison_time: int, collection_name: str
):
    """ """
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    app.logger.info("Segmentation calculation triggered...")
    app.logger.info(baseline_time)
    app.logger.info(comparison_time)

    if not ctx.triggered:
        app.logger.info("Returning default render urls because segmentation not triggered.")
        return DEFAULT_RESPONSE

    if (
        data is not None
        and baseline_time is not None
        and comparison_time is not None
        and collection_name is not None
    ):
        if trigger_id in ['segmentation-btn']:
            try:
                app.logger.info(f"Segmentation render initiated")
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
                    baseline_segmentation_image,
                    comparison_segmentation_image,
                ) = get_segmentation(data, baseline_date, comparison_date, n_classes)

                app.logger.info(f"Loading images...")

                baseline_segmentation_url = get_image_url(baseline_segmentation_image, colormap='random_classes')
                comparison_segmentation_url = get_image_url(comparison_segmentation_image, colormap='random_classes')

                et = datetime.datetime.now()
                delta = (et - st).seconds
                app.logger.info(f"Color data rendered - {delta}s")

                return (
                    color_bounds,
                    baseline_segmentation_url,
                    comparison_segmentation_url,
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
            app.logger.info("Returning default map because data ready but not triggered.")

            return (None, None, None, "Press Compute to initiate unsupervised segmentation")
    else:
        app.logger.info("Returning default map because triggered but data not ready.")

        return DEFAULT_RESPONSE


def get_segmentation(data, baseline_date, comparison_date, n_classes):
    baseline_color_image, color_map_bounds = compute_segmentation_at_date(
        data, baseline_date, n_classes
    )
    comparison_color_image, color_map_bounds = compute_segmentation_at_date(
        data, comparison_date, n_classes
    )

    return color_map_bounds, baseline_color_image, comparison_color_image


def compute_segmentation_at_date(data, baseline_date, n_classes):
    data_at_date = data.sel({"date": baseline_date})

    data = data_at_date.rio.write_crs(data_at_date.rio.crs)
    data = data.transpose("band", "y", "x")
    data = data.rio.reproject("EPSG:4326")
    data = data.transpose("y", "x", "band")
    data = data.rename({[i for i in data.data_vars][0]: "msi"})

    vals = data.msi.values
    vals[np.isnan(vals)] = 0
    vals[np.isinf(vals)] = 0
    vals[vals > 65535] = 0
    for ii in range(vals.shape[-1]):
        vals[:,:,ii] = (vals[:,:,ii] - vals[:,:,ii].min()) /  (vals[:,:,ii].max() - vals[:,:,ii].min())

    features = vals.reshape(vals.shape[0] * vals.shape[1],-1)

    shp = vals.shape[:2]

    pca = PCA(n_components=3)
    try:
        features = pca.fit_transform(features)
    except:
        app.logger.info(np.isnan(features).sum())
        app.logger.info(np.isinf(features).sum())
        app.logger.info(np.mean(features))
        raise

    kmeans = KMeans(n_clusters=n_classes).fit(features)
    classes = kmeans.predict(features)
    cat_map = classes.reshape(shp)

    xmin, ymin, xmax, ymax = data.rio.bounds()
    color_map_bounds = [[ymin, xmin], [ymax, xmax]]
    #cat_map = np.expand_dims(cat_map, axis=2)

    return cat_map, color_map_bounds
