import numpy as np
import os
import cv2
from tqdm import tqdm

def load_train_data(train_dir, img_size=256):
    X_train = []
    y_train = []

    for folder in tqdm(os.listdir(train_dir)):
        img = cv2.imread(train_dir + folder + '/images/' + folder + '.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size))
        X_train.append(img)

        mask = cv2.imread(train_dir + folder + '/masks/' + folder + '.png')
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = cv2.resize(mask, (img_size, img_size))
        y_train.append(mask)

    return np.array(X_train), np.array(y_train, dtype=np.float32)