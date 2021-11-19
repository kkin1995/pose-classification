from pose_classifier_utils import pose_classifier, get_prediction
import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd
import tensorflow as tf
from utils import logger

LEARNING_RATE = 0.01
EPOCHS = 10

data = np.load("data/train_data.npy")
labels = np.load("data/train_labels.npy")
test_data = np.load("data/test_data.npy")
test_labels = np.load("data/test_labels.npy")

logger("Shape of Data = {}".format(data.shape))
logger("Shape of Labels = {}".format(labels.shape))
logger("Shape of Test Data = {}".format(test_data.shape))
logger("Shape of Test Labels = {}".format(test_labels.shape))

model, history, test_metrics = pose_classifier(data, labels, test_data, test_labels, learning_rate = LEARNING_RATE, epochs = EPOCHS)
#tf.keras.utils.plot_model(model, to_file="models/model.png", show_shapes=True, show_layer_names=True)
model.save("models/pose_classifier")

logger("Training Accuracy = {}".format(history.history["accuracy"][-1]))
logger("Training Loss = {}".format(history.history["loss"][-1]))
logger("Validation Accuracy = {}".format(history.history["val_accuracy"][-1]))
logger("Validation Loss = {}".format(history.history["val_loss"][-1]))
logger("Testing Accuracy = {}".format(test_metrics[0]))
logger("Testing Loss = {}".format(test_metrics[1]))

fig, axs = plt.subplots(2, 2)
plt.suptitle("Training Metrics")
plt.tight_layout()

axs[0, 0].plot(history.history["accuracy"])
axs[0, 0].set_title("Training Accuracy")
axs[0, 0].set_xlabel("Epochs")
axs[0, 1].plot(history.history["loss"])
axs[0, 1].set_title("Training Loss")
axs[0, 1].set_xlabel("Epochs")
axs[1, 0].plot(history.history["val_accuracy"])
axs[1, 0].set_title("Validation Accuracy")
axs[1, 0].set_xlabel("Epochs")
axs[1, 1].plot(history.history["val_loss"])
axs[1, 1].set_title("Validation Loss")
axs[1, 1].set_xlabel("Epochs")

plt.savefig("plots/training_validation_metrics.png")
#plt.show()