import json
import os
import warnings

import boto3
import rasterio
import requests
from rasterio.session import AWSSession


def get_new_token():
    auth_server_url = os.getenv("EDS_AUTH_URL")
    client_id = os.getenv("EDS_CLIENT_ID")
    client_secret = os.getenv("EDS_SECRET")
    token_req_payload = {"grant_type": "client_credentials"}

    token_response = requests.post(
        auth_server_url,
        data=token_req_payload,
        verify=False,
        allow_redirects=False,
        auth=(client_id, client_secret),
    )
    token_response.raise_for_status()

    tokens = json.loads(token_response.text)
    return tokens["access_token"]


def authenticate_rasterio(func):
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            boto3.setup_default_session()
            if boto3.DEFAULT_SESSION is not None:
                credentials = boto3.DEFAULT_SESSION.get_credentials()
                assert credentials is not None
            else:
                raise Exception("Boto3 default session not found!")
            sess = AWSSession(
                aws_unsigned=False,
                aws_access_key_id=credentials.access_key,
                aws_secret_access_key=credentials.secret_key,
                aws_session_token=credentials.token,
                region_name=os.environ["AWS_DEFAULT_REGION"],
                requester_pays=True,
            )
        with rasterio.Env(
            session=sess,
        ) as env:
            result = func(*args, **kwargs)

        return result

    return wrapper
