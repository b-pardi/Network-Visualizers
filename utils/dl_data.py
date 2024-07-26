import numpy as np
import requests
import gzip
import os

def download_mnist(fp):
    base_url = 'http://yann.lecun.com/exdb/mnist/'
    files = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz',
        'test_images': 't10k-images-idx3-ubyte.gz',
        'test_labels': 't10k-labels-idx1-ubyte.gz'
    }
    
    os.makedirs('mnist', exist_ok=True)
    
    for key, filename in files.items():
        data_dir = os.path.join(fp, key+'/')
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            url = base_url + filename
            response = requests.get(url)
            with open(os.path.join(data_dir, filename), 'wb') as f:
                f.write(response.content)