import os
import zipfile

import requests

# Define the URL of the ZIP file to download
url = "https://huggingface.co/datasets/earthdaily/venus-change-detection/resolve/main/VENUS_CHANGE_DATASET.zip"

# Define the folder where you want to store the downloaded ZIP file
download_folder = "/teamspace/studios/this_studio/data/"

# Ensure the download folder exists
os.makedirs(download_folder, exist_ok=True)

# Define the path to save the downloaded ZIP file
zip_file_path = os.path.join(download_folder, "VENUS_CHANGE_DATASET.zip")

# Download the ZIP file from the URL
response = requests.get(url)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Save the content of the response to the ZIP file
    with open(zip_file_path, "wb") as zip_file:
        zip_file.write(response.content)

    # Unzip the downloaded ZIP file to the 'data' folder
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(download_folder)

    print(f'Successfully downloaded and unzipped to {os.path.abspath("data")}')
else:
    print(f"Failed to download the ZIP file. Status code: {response.status_code}")
