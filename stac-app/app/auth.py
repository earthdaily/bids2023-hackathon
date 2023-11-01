import json
import os

import requests


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
