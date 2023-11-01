. ./.env
docker build -t stac-app . 
docker run \
-e AWS_DEFAULT_REGION="$AWS_DEFAULT_REGION" \
-e EDS_AUTH_URL="$EDS_AUTH_URL" \
-e EDS_CLIENT_ID="$EDS_CLIENT_ID" \
-e EDS_SECRET="$EDS_SECRET" \
-e EDS_API_URL="$EDS_API_URL" \
-it --rm -p 8050:8050 stac-app
