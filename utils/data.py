import numpy as np
import requests
import gzip
import os
import hashlib

def onehot_encode(labels, num_classes):
    """
    Convert an array of labels to one-hot encoded format using NumPy.
    
    Args:
    - labels (ndarray): Array of label integers.
    - num_classes (int): Number of classes (default is 10 for MNIST).
    
    Returns:
    - one_hot (ndarray): One-hot encoded labels.
    """
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot

def stratified_split(x_train, y_train, val_size=0.1, num_classes=10, random_state=42):
    """
    Splits x_train and y_train into training and validation sets using stratified sampling.

    Args:
    - x_train (np.ndarray): Input features for training.
    - y_train (np.ndarray): One-hot encoded labels for training.
    - val_size (float): Proportion of the training set to be used for validation (e.g., 0.1 for 10%).
    - num_classes (int): Number of classes (default 10 for MNIST).
    - random_state (int): Seed for reproducibility.

    Returns:
    - x_train_strat: Stratified training set after splitting.
    - y_train_strat: Stratified training labels after splitting.
    - x_val_strat: Stratified validation set.
    - y_val_strat: Stratified validation labels.
    """
    np.random.seed(random_state)
    y_train_labels = np.argmax(y_train, axis=1)

    # Initialize arrays for final splits
    x_train_strat = np.empty((0, *x_train.shape[1:]), dtype=x_train.dtype)
    y_train_strat = np.empty((0, y_train.shape[1]), dtype=y_train.dtype)
    x_val_strat = np.empty((0, *x_train.shape[1:]), dtype=x_train.dtype)
    y_val_strat = np.empty((0, y_train.shape[1]), dtype=y_train.dtype)

    # grab data from each class and split them evenly
    for i in range(num_classes):
        # find indices for current class and shuffle them
        class_idxs = np.where(y_train_labels == i)[0]
        np.random.shuffle(class_idxs)

        # determine size of val set based on val_size ratio argument
        val_count = int(len(class_idxs) * val_size)

        # split data into training and val sets
        val_idxs = class_idxs[:val_count]
        train_idxs = class_idxs[val_count:]

        # append data
        x_train_strat = np.vstack((x_train_strat, x_train[train_idxs]))
        y_train_strat = np.vstack((y_train_strat, y_train[train_idxs]))
        x_val_strat = np.vstack((x_val_strat, x_train[val_idxs]))
        y_val_strat = np.vstack((y_val_strat, y_train[val_idxs]))

    # shuffle the stratified training and validation sets to randomize order
    train_shuffle_indices = np.random.permutation(x_train_strat.shape[0])
    val_shuffle_indices = np.random.permutation(x_val_strat.shape[0])

    x_train_strat = x_train_strat[train_shuffle_indices]
    y_train_strat = y_train_strat[train_shuffle_indices]
    x_val_strat = x_val_strat[val_shuffle_indices]
    y_val_strat = y_val_strat[val_shuffle_indices]

    return x_train_strat, y_train_strat, x_val_strat, y_val_strat


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