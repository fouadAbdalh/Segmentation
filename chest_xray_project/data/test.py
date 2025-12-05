import numpy as np
import os
import cv2
from tqdm import tqdm

def load_test_data(test_dir, img_size=256):
    X_test = []

    for folder in tqdm(os.listdir(test_dir)):
        img = cv2.imread(test_dir + folder + '/' + folder + '.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size))
        X_test.append(img)

    return np.array(X_test)