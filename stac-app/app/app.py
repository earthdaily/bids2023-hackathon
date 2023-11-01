import datetime
import logging

import dash_bootstrap_components as dbc
import diskcache
from dash_extensions.enrich import (
    DashProxy,
    Input,
    Output,
    ServersideOutputTransform,
    dcc,
    html,
)
from dask.distributed import Client, LocalCluster  # noqa

from app import config, services
from app.layouts import get_main_page

st = datetime.datetime.now()

log = logging.getLogger(__name__)


app = DashProxy(
    __name__,
    transforms=[ServersideOutputTransform()],
    external_stylesheets=[dbc.themes.PULSE],
    assets_url_path="app/assets",
    title="EarthDaily",
    meta_tags=[
        {
            "name": "viewport",
            "content": "width=device-width, height=device-height, initial-scale=1.0, maximum-scale=1.2",
        }
    ],
)

log.info(f"Initializing app... {(datetime.datetime.now() - st).seconds}s")
cache = diskcache.Cache("./cache")

client = Client(processes=False)
log.info(f"Cluster loaded... {(datetime.datetime.now() - st).seconds}s")

server = app.server
log.info(f"Server loaded... {(datetime.datetime.now() - st).seconds}s")

app.layout = html.Div(
    [dcc.Location(id="url", refresh=False), html.Div(id="page-content")]
)
log.info(f"Layout loaded.... {(datetime.datetime.now() - st).seconds}s")

# Important note about call order: Must assign the `layout` property before assigning callbacks.
callbacks = services.get_callbacks()
log.info(f"Callbacks loaded... {(datetime.datetime.now() - st).seconds}s")


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def display_page(pathname):
    return get_main_page()


log.info(f"App loaded. {(datetime.datetime.now() - st).seconds}s")

if __name__ == "__main__":
    app.run_server(debug=True)
