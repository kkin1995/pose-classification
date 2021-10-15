# Pose Detection and Classification

A project that uses the Posenet pretrained model from Tensorflow to detect keypoints on the human body
and uses those keypoints to classify the pose.

Yoga pose classification dataset is from Kaggle. [Link to Dataset](https://www.kaggle.com/niharika41298/yoga-poses-dataset)

The file "posenet.py" inside the src/ directory contains functions to compute keypoints and plot keypoints on a single image
provided as a Numpy Array

The file "generate_keypoints_for_data.py" contains a function to compute keypoints for many images for many classes.

## Install Dependencies
```
pip3 install -r requirements.txt
```

## Install Dependencies for Development (Includes Jupyter Notebook)
```
pip3 install -r requirements-dev.txt
```