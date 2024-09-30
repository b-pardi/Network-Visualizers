import numpy as np
import requests
import gzip
import os
import hashlib

def compute_md5_checksum(fp):
    md5_hash = hashlib.md5() # md5 hash object
    with open(fp, 'rb') as f:

        # iterate through file in 8KB chunks to avoid memory issues
        byte_block_iter = iter(lambda: f.read(8192), b'') # iterates until sentinel value of empty byte string
        for byte_block in byte_block_iter:
            md5_hash.update(byte_block)
        
        full_hash = md5_hash.hexdigest()
        return full_hash

def download_mnist(fp):
    base_url = ' https://storage.googleapis.com/cvdf-datasets/mnist/'
    files = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz',
        'test_images': 't10k-images-idx3-ubyte.gz',
        'test_labels': 't10k-labels-idx1-ubyte.gz'
    }
    
    data_dir_dict = {}
    for key, filename in files.items():
        data_dir = os.path.join(fp, key+'/')
        data_dir_dict[key] = os.path.join(data_dir, filename)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            url = base_url + filename
            response = requests.get(url)
            with open(os.path.join(data_dir, filename), 'wb') as f:
                f.write(response.content)

    return data_dir_dict

def load_mnist(data_dir_dict, mnist_md5_dict):
    for data_type, data_fp in data_dir_dict.items():
        # check file integrity
        checksum = compute_md5_checksum(data_fp)
        if checksum != mnist_md5_dict[data_type]:
            raise ValueError(f"FILE INTEGRITY ERROR. Mnist {data_type} expected md5 checksum of {mnist_md5_dict[data_type]} but received {checksum}")

    x_train = load_mnist_images(data_dir_dict['train_images'])
    y_train = load_mnist_labels(data_dir_dict['train_labels'])
    x_test = load_mnist_images(data_dir_dict['test_images'])
    y_test = load_mnist_labels(data_dir_dict['test_labels'])

    return x_train, y_train, x_test, y_test
        

def load_mnist_images(data_fp):
    with gzip.open(data_fp, 'rb') as f:
        # mnist downloads as idx files, 
        # sample byte outputs of each header section
        # magic number b'\x00\x00\x08\x03' first 2 bytes always 0 indicating idx file, 8 is dtype of file (unsigned int), 3 is n_dimensions
        magic_number = int.from_bytes(f.read(4), byteorder='big')
        num_images = int.from_bytes(f.read(4), byteorder='big')
        num_rows = int.from_bytes(f.read(4), byteorder='big')
        num_cols = int.from_bytes(f.read(4), byteorder='big')

        # Read image data
        buffer = f.read(num_images * num_rows * num_cols) # byte obj containing all bytes of image data (after header bytes)
        images = np.frombuffer(buffer, dtype=np.uint8) # create 1D array from byte buffer
        images = images.reshape(num_images, num_rows, num_cols) # reshape 1d array into 3D array of imgs
        return images

def load_mnist_labels(data_fp):
    # open gzip as bytes object
    with gzip.open(data_fp, 'rb') as f:
        magic_number = int.from_bytes(f.read(4), byteorder='big')
        num_labels = int.from_bytes(f.read(4), byteorder='big')

        # Read label data
        buffer = f.read(num_labels)
        labels = np.frombuffer(buffer, dtype=np.uint8)
        return labels