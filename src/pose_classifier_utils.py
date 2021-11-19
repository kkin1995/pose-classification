import numpy as np
import pandas as pd
import glob
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense

def pose_classifier(data, labels, test_data, test_labels, learning_rate = 0.01, epochs = 10):
    """
    Takes the data and labels as input, creates a keras model using the Functional API
    and returns the trained model, training / validation metrics and testing metrics.

    Parameters:
    data: A Numpy Array of shape (batch_size, input_dim)
    labels: A Numpy Array of shape (batch_size, no_of_classes)
    learning_rate: A floating point value defining the step size of the learning algorithm
    epochs: An integer defining the number of iterations
    """
    X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.3, random_state=42)

    print("Shape of Training Data = {}".format(X_train.shape))
    print("Shape of Training Labels = {}".format(y_train.shape))
    print("Shape of Validation Data = {}".format(X_val.shape))
    print("Shape of Validation Labels = {}".format(y_val.shape))

    inputs = Input(shape = (34,))
    x = Dense(100, activation = 'relu')(inputs)
    outputs = Dense(5, activation = 'softmax')(x)
    model = keras.Model(inputs = inputs, outputs = outputs, name = "pose_classifier")
    opt = keras.optimizers.Adam(learning_rate = learning_rate)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics = ["accuracy"])

    history = model.fit(x = X_train, y = y_train, validation_data = (X_val, y_val), epochs = epochs)

    test_metrics = model.evaluate(test_data, test_labels)

    return model, history, test_metrics

def get_prediction(keypoints_array, model):
    predictions = model.predict(keypoints_array)
    classes = ["downdog", "goddess", "plank", "tree", "warrior2"]
    prediction_index = np.argmax(predictions)
    return classes[prediction_index]