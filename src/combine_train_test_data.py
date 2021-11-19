import numpy as np
import pandas as pd
import glob
from utils import logger

TRAIN_DIR = "data/training-keypoints/"
TEST_DIR = "data/testing-keypoints/"
classes = ["downdog", "goddess", "plank", "tree", "warrior2"]

train_files = np.array([])
test_files = np.array([])
no_of_train_files, no_of_test_files = 0, 0
for c in classes:
    train = np.array(glob.glob(TRAIN_DIR + c + "/*.npy"))
    test = np.array(glob.glob(TEST_DIR + c + "/*.npy"))
    train_files = np.append(train_files, train)
    test_files = np.append(test_files, test)
    no_of_train_files += len(train)
    no_of_test_files += len(test)

one_hot_classes = dict(pd.get_dummies(classes))

batch_size = len(train_files)
test_batch_size = len(test_files)
input_dim = 34 # Number of Keypoints / Input Dimension
no_of_classes = len(classes) # Number of Classes / Output Dimension
data_shape = (batch_size, input_dim)
test_data_shape = (test_batch_size, input_dim)
labels_shape = (batch_size, no_of_classes)
test_labels_shape = (test_batch_size, no_of_classes)

data = np.zeros(data_shape)
test_data = np.zeros(test_data_shape)
labels = np.zeros(labels_shape)
test_labels = np.zeros(test_labels_shape)

for i, c in enumerate(classes):
    logger("Combining data for class: {}".format(c))
    
    for j, f in enumerate(train_files):
        class_of_f = f.split("/")[-2]
        if class_of_f == c:
            data_example = np.load(f)
            data[j, :] = data_example.ravel()
            labels[j, :] = one_hot_classes[c].values

    for j, f in enumerate(test_files):
        class_of_f = f.split("/")[-2]
        if class_of_f == c:
            test_data_example = np.load(f)
            test_data[j, :] = test_data_example.ravel()
            test_labels[j, :] = one_hot_classes[c].values

np.save("data/train_data.npy", data)
np.save("data/train_labels.npy", labels)
np.save("data/test_data.npy", test_data)
np.save("data/test_labels", test_labels)