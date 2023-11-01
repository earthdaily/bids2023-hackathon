 ### STAC-App
 This demonstration app enables users to make EDS queries and render data within an AOI for two dates to enable quick comparison. To run this app on your local machine, run the following commands:
 
 ```
 cd stac-app
 ./run_app.sh
 ```

 Note: requires the file `stac-app/.env` to be populated with the following environment variables:
 
 ```
 AWS_DEFAULT_REGION=us-east-1
 EDS_AUTH_URL="https://2023-bids-hackathon-earthdaily.auth.us-east-1.amazoncognito.com/oauth2/token"
 EDS_API_URL="https://api.eds.earthdaily.com/archive/v1/stac/v1"
 EDS_CLIENT_ID=<id>
 EDS_SECRET=<secret>
 AWS_NO_SIGN_REQUEST=YES
 ```
