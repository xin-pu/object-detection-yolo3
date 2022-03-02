import os

import requests
from tqdm import tqdm


def download_file(filename, url):
    print("Download {} from {}".format(filename, url))

    chunk_size = 1024
    r = requests.get(url, stream=True)
    with open(filename, 'wb') as f:
        pbar = tqdm(unit="B", total=int(r.headers['Content-Length']))
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:  # filter out keep-alive new chunks
                pbar.update(len(chunk))
                f.write(chunk)
    return filename


def download_if_not_exists(filename, url):
    if not os.path.exists(filename):
        download_file(filename, url)
        return True
    return False
