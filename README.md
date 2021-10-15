# Pose Detection and Classification

A project that uses the Posenet pretrained model from Tensorflow to detect keypoints on the human body
and uses those keypoints to classify the pose.

Yoga pose classification dataset is from Kaggle. [Link to Dataset](https://www.kaggle.com/niharika41298/yoga-poses-dataset)

The file "posenet.py" inside the src/ directory contains functions to compute keypoints and plot keypoints on a single image
provided as a Numpy Array

The file "generate_keypoints_for_data.py" contains a function to compute keypoints for many images for many classes.

The file "pose_classifier.py" contains a function that takes as input the data and labels, creates a keras model using the functional API, and returns the trained model, training and validation metrics. The shape of data should be (batch_size, input_dim) and the shape of the labels should be (batch_size, no_of_classes).

## Install Dependencies
```
pip3 install -r requirements.txt
```

## Install Dependencies for Development (Includes Jupyter Notebook)
```
pip3 install -r requirements-dev.txt
```