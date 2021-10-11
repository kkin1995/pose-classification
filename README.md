# Pose Detection and Classification

A project that uses the Posenet pretrained model from Tensorflow to detect keypoints on the human body
and uses those keypoints to classify the pose.

Yoga pose classification dataset is from Kaggle. [Link to Dataset](https://www.kaggle.com/niharika41298/yoga-poses-dataset)

The file "posenet.py" inside the src/ directory contains functions to compute keypoints and plot keypoints on a single image
provided as a Numpy Array

The file "generate_keypoints_for_data.py" contains a function to compute keypoints for many images for many classes.

## Note
The "pyproject.toml" and "poetry.lock" files are invalid. Following are the list of dependencies:
1. Tensorflow
2. OpenCV
3. Numpy
4. Matplotlib
5. Scikit-Image
6. tqdm