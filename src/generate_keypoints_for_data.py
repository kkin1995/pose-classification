import os
import glob
import cv2
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from posenet import return_keypoints_from_image_with_model
import tensorflow as tf
from tqdm import tqdm
import time

def logger(message):
    """
    Logs a message with timestamp to log file

    Parameters:
    message: String Representation of message to be logged
    """
    logfile = "../logs/" + time.asctime().split()[2] + "-" + time.asctime().split()[1] + "-" + time.asctime().split()[-1] + ".log"
    with open(logfile, "a") as f:
        f.write("[" + time.asctime().split()[3] + "] " + message + "\n")

def generate_keypoints(source_data_folder, target_data_folder, classes, model):
    """
    Parameters:
    source_data_folder: String representation of path with input images in respective class folder
    Example
    TRAIN
        - downdog
            - Image Files
        - goddess
            - Image Files
        - plank
            - Image Files
        - tree
            - Image Files
        - warrior2
            - Image Files
    
    target_data_folder: String representation of path to save keypoints generated from input images
    classes: Python List of classes
    model: Tensorflow Signature of Posenet Model
    
    """
    pbar = tqdm(classes)
    for c in pbar:
        pbar.set_description("Generating Keypoints for " + c)
        logger("Generating Keypoints for " + c)
        source_files = glob.glob(source_data_folder + c + "/*.jpg")
        target_class_folder = target_data_folder + c + "/"
        if os.path.isdir(target_class_folder) == False:
            os.makedirs(target_class_folder)
        for filename in source_files:
            image_number = filename.split("/")[-1].split(".")[0]
            try:
                _ = io.imread(filename)
                image = cv2.imread(filename)
            except Exception as e:
                logger(str(e))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            keypoints_dict = return_keypoints_from_image_with_model(image, model)
            keypoints = np.array(list(keypoints_dict.values()))[:, 0:2].reshape((34,1))
            np.save(target_class_folder + image_number + ".npy", keypoints)


if __name__ == '__main__':
    # Posenet Model Download URL: https://tfhub.dev/google/movenet/singlepose/lightning/4
    posenet = tf.saved_model.load("models/movenet_singlepose_lightning_4")
    posenet_model = posenet.signatures["serving_default"]

    classes = os.listdir("data/TRAIN/")
    training_folder = "data/TRAIN/"
    testing_folder = "data/TEST/"
    training_keypoints_folder = "data/training-keypoints/"
    testing_keypoints_folder = "data/testing-keypoints/"

    generate_keypoints(training_folder, training_keypoints_folder, classes, posenet_model)
    generate_keypoints(testing_folder, testing_keypoints_folder, classes, posenet_model)