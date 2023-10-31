. ./.env
docker build -t stac-app .
docker run -it --rm -p 8050:8050 stac-app
