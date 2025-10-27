import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from skimage.feature import hog
import config


def list_images(data_dir):
    images = []
    labels = []
    for label_name in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label_name)
        if not os.path.isdir(label_dir):
            continue
        for fname in os.listdir(label_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                images.append(os.path.join(label_dir, fname))
                labels.append(label_name)
    return images, labels


def load_and_preprocess(path, size=config.IMG_SIZE):
    img = Image.open(path).convert('L')  # grayscale
    img = img.resize(size)
    arr = np.array(img).astype(np.float32) / 255.0
    return arr


def extract_features(paths, method='flatten'):
    feats = []
    for p in paths:
        img = load_and_preprocess(p)
        if method == 'flatten':
            feats.append(img.flatten())
        elif method == 'hog':
            h = hog(img, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True)
            feats.append(h)
        else:
            raise ValueError('Unknown method')
    return np.vstack(feats)


def train_test_split_paths(paths, labels):
    X_train, X_test, y_train, y_test = train_test_split(
        paths, labels, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED, stratify=labels)
    return X_train, X_test, y_train, y_test
