. ./.env
docker build -t stac-app . 
docker run \
-e EDS_AUTH_URL="$EDS_AUTH_URL" \
-e EDS_CLIENT_ID="$EDS_CLIENT_ID" \
-e EDS_SECRET="$EDS_SECRET" \
-e EDS_API_URL="$EDS_API_URL" \
-e AWS_NO_SIGN_REQUEST="$AWS_NO_SIGN_REQUEST" \
-v ./:/code \
-it --rm -p 8050:8050 stac-app
