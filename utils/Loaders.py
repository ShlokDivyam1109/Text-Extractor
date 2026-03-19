import os
import idx2numpy
import numpy as np
import torch
import time
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

classes = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

class EMNISTDataset(Dataset):

    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        self.classes = [classes[l] for l in labels]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img = self.images[idx]
        label_id = int(self.labels[idx])
        class_char = self.classes[idx]

        # fix orientation
        img = np.rot90(img, 3)
        img = np.fliplr(img)

        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        return (
            torch.tensor(img),
            class_char
        )

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def get_loaders(DATA_DIR = os.path.join(BASE_DIR,"emnist_source_files"),log = True):
    if log:
        log_dir = os.path.join(BASE_DIR, "logs")

        os.makedirs(log_dir, exist_ok=True)

        log = os.path.join(
            log_dir,
            "log_" + time.strftime("%Y%m%d_%H%M%S") + ".txt"
        )
    if log:
        with open(log,'w') as f:
            f.write("Loading the Dataset...")

    try:
        train_images = idx2numpy.convert_from_file(
            os.path.join(DATA_DIR, "emnist-byclass-train-images-idx3-ubyte")
        )
    except Exception as e:
        if log:
            with open(log,'a') as f:
                f.write(str(e))
        else:
            raise e

    if log:
        with open(log,'a') as f:
            f.write("Training images loaded\n")

    try:
        train_labels = idx2numpy.convert_from_file(
            os.path.join(DATA_DIR, "emnist-byclass-train-labels-idx1-ubyte")
        )
    except Exception as e:
        if log:
            with open(log,'a') as f:
                f.write(str(e))
        else:
            raise e

    if log:
        with open(log,'a') as f:
            f.write("Training labels loaded\n")

    try:
        test_images = idx2numpy.convert_from_file(
            os.path.join(DATA_DIR, "emnist-byclass-test-images-idx3-ubyte")
        )
    except Exception as e:
        if log:
            with open(log,'a') as f:
                f.write(str(e))
        else:
            raise e

    if log:
        with open(log,'a') as f:
            f.write("Testing images loaded\n")

    try:
        test_labels = idx2numpy.convert_from_file(
            os.path.join(DATA_DIR, "emnist-byclass-test-labels-idx1-ubyte")
        )
    except Exception as e:
        if log:
            with open(log,'a') as f:
                f.write(str(e))
        else:
            raise e

    if log:
        with open(log,'a') as f:
            f.write("Testing labels loaded\n")

    try:
        images = np.concatenate((train_images, test_images))
        labels = np.concatenate((train_labels, test_labels))
    except Exception as e:
        if log:
            with open(log,'a') as f:
                f.write(str(e))
        else:
            raise e

    if log:
        with open(log,'a') as f:
            f.write("Dataset Loading Complete\nSplitting Dataset\n")

    try:
        X_train, X_temp, y_train, y_temp = train_test_split(
            images, labels, test_size=0.2, random_state=42
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )
    except Exception as e:
        if log:
            with open(log,'a') as f:
                f.write(str(e))
        else:
            raise e

    if log:
        with open(log,'a') as f:
            f.write("Test-Train Splitting Complete\n")

    try:
        train_dataset = EMNISTDataset(X_train, y_train)
        val_dataset = EMNISTDataset(X_val, y_val)
        test_dataset = EMNISTDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64)
        test_loader = DataLoader(test_dataset, batch_size=64)
    except Exception as e:
        if log:
            with open(log,'a') as f:
                f.write(str(e))
        else:
            raise e

    if log:
        with open(log,'a') as f:
            f.write("Loader Creation Complete\n")

    return train_loader, val_loader, test_loader
