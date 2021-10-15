import numpy as np
import pandas as pd
import glob
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense

LEARNING_RATE = 0.01
EPOCHS = 10

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
    
files = np.append(train_files, test_files)
no_of_files = no_of_train_files + no_of_test_files

one_hot_classes = dict(pd.get_dummies(classes))

batch_size = len(files)
input_dim = 34 # Number of Keypoints / Input Dimension
no_of_classes = len(classes) # Number of Classes / Output Dimension
data_shape = (batch_size, input_dim)
labels_shape = (batch_size, no_of_classes)

data = np.zeros(data_shape)
labels = np.zeros(labels_shape)

for i, c in enumerate(classes):
    for j, f in enumerate(files):
        class_of_f = f.split("/")[-2]
        if class_of_f == c:
            data_example = np.load(f)
            data[j, :] = data_example.ravel()
            labels[j, :] = one_hot_classes[c].values

print("Shape of Data = {}".format(data.shape))
print("Shape of Labels = {}".format(labels.shape))

X_train, X_test_val, y_train, y_test_val = train_test_split(data, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test_val, y_test_val, test_size = 0.4, random_state = 42)

print("Shape of Training Data = {}".format(X_train.shape))
print("Shape of Training Labels = {}".format(y_train.shape))
print("Shape of Validation Data = {}".format(X_val.shape))
print("Shape of Validation Labels = {}".format(y_val.shape))
print("Shape of Test Data = {}".format(X_test.shape))
print("Shape of Test Labels = {}".format(y_test.shape))

model = keras.Sequential([
    Input(shape = (34,)),
    Dense(20, activation = 'relu'),
    Dense(5, activation = 'softmax')
])

opt = keras.optimizers.Adam(learning_rate = LEARNING_RATE)
model.compile(optimizer=opt, loss="categorical_crossentropy", metrics = "accuracy")

history = model.fit(x = X_train, y = y_train, epochs = EPOCHS)

evaluation_metrics = model.evaluate(X_val, y_val)

print("Evaluation Loss = {}".format(evaluation_metrics[0]))
print("Evaluation Accuracy = {}".format(evaluation_metrics[1]))

fig, axs = plt.subplots(1, 2)
plt.suptitle("Training Metrics")
axs[0].plot(history.history["accuracy"])
axs[0].set_title("Training Accuracy")
axs[0].set_xlabel("Epochs")
axs[1].plot(history.history["loss"])
axs[1].set_title("Training Loss")
axs[1].set_xlabel("Epochs")

plt.show()
