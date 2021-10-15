import numpy as np
import pandas as pd
import glob
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense

def pose_classifier(data, labels):
    """
    Takes the data and labels as input, creates a keras model using the Functional API
    and returns the trained model, training / validation metrics and testing metrics.

    Parameters:
    data: A Numpy Array of shape (batch_size, input_dim)
    labels: A Numpy Array of shape (batch_size, no_of_classes)
    """
    X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.3, random_state=42)

    print("Shape of Training Data = {}".format(X_train.shape))
    print("Shape of Training Labels = {}".format(y_train.shape))
    print("Shape of Validation Data = {}".format(X_val.shape))
    print("Shape of Validation Labels = {}".format(y_val.shape))

    inputs = Input(shape = (34,))
    x = Dense(20, activation = 'relu')(inputs)
    outputs = Dense(5, activation = 'softmax')(x)
    model = keras.Model(inputs = inputs, outputs = outputs, name = "pose_classifier")
    opt = keras.optimizers.Adam(learning_rate = LEARNING_RATE)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics = "accuracy")

    history = model.fit(x = X_train, y = y_train, validation_data = (X_val, y_val), epochs = EPOCHS)

    test_metrics = model.evaluate(test_data, test_labels)

    return model, history, test_metrics

if __name__ == '__main__':
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
        for j, f in enumerate(train_files):
            class_of_f = f.split("/")[-2]
            if class_of_f == c:
                data_example = np.load(f)
                data[j, :] = data_example.ravel()
                labels[j, :] = one_hot_classes[c].values

    for i, c in enumerate(classes):
        for j, f in enumerate(test_files):
            class_of_f = f.split("/")[-2]
            if class_of_f == c:
                test_data_example = np.load(f)
                test_data[j, :] = test_data_example.ravel()
                test_labels[j, :] = one_hot_classes[c].values

    print("Shape of Data = {}".format(data.shape))
    print("Shape of Labels = {}".format(labels.shape))
    print("Shape of Test Data = {}".format(test_data.shape))
    print("Shape of Test Labels = {}".format(test_labels.shape))

    model, history, test_metrics = pose_classifier(data, labels)
    tf.keras.utils.plot_model(model, to_file="models/model.png", show_shapes=True, show_layer_names=True)
    model.save("models/pose_classifier")

    print("Test Loss = {}".format(test_metrics[0]))
    print("Test Accuracy = {}".format(test_metrics[1]))

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
