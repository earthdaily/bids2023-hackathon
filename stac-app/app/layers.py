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


venus_layer = dl.Overlay(dl.TileLayer(id='venus-sites',
                                         url='https://api.mapbox.com/styles/v1/rschueder/clof3c3e6002d01pwgb8y9i92/tiles/256/{z}/{x}/{y}@2x?access_token=pk.eyJ1IjoicnNjaHVlZGVyIiwiYSI6ImNrdHJwYmZqcTBtejcydXFpcDZhaWhrcXYifQ.iQqW4uQ3KWVIQh6OfxkbzQ',
                                         ),
                            name='VENUS Sites', checked=True)