import os

local = True
n_workers = 2
memory = 32
worker_memory = int(memory / n_workers)

STAC_API_URL = "https://edc-skyfox.eds.earthdaily.com/archive/v1/stac/v1"
static_asset_path = "/app/assets/"

osm_url = "https://tile.openstreetmap.org/${z}/${x}/${y}.png"
topo_url = "http://services.arcgisonline.com/arcgis/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}"

init_bounds = [[48.78, -123.755], [49.7611, -122.5]]
