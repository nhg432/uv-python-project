# This script is used to install pip.
# Download the latest version from https://bootstrap.pypa.io/get-pip.py if needed.

import os
import sys
import urllib.request

def download_get_pip():
    url = "https://bootstrap.pypa.io/get-pip.py"
    filename = "get-pip-latest.py"
    print(f"Downloading get-pip.py from {url} ...")
    urllib.request.urlretrieve(url, filename)
    print(f"Downloaded as {filename}.")
    print("Run it with: python get-pip-latest.py")

if __name__ == "__main__":
    download_get_pip()