import dash_leaflet as dl

from app import config

base_layer = dl.BaseLayer(
    dl.TileLayer(
        id="basemap",
        url=config.topo_url,
    ),
    name="Basemap",
    checked=True,
)
