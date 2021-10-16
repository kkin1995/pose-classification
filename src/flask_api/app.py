from flask import Flask, jsonify, request
import numpy as np
import os
import sys
# Append path of parent directory to PATH to import pose_classifier and posenet
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from pose_classifier import get_prediction
from posenet import return_keypoints_from_image
from tensorflow.keras.models import load_model
import cv2

app = Flask(__name__)
@app.route("/predict", methods=['POST'])
def predict():
    file = request.files['image'].read()
    image = np.fromstring(file, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    keypoints_dict = return_keypoints_from_image(image)
    keypoints = np.array(list(keypoints_dict.values()))[:, 0:2].reshape((1,34))
    model = load_model("/Users/karankinariwala/Dropbox/KARAN/5-Projects/pose-detection/src/models/pose_classifier")
    prediction = get_prediction(keypoints, model)
    return jsonify({'predictions': prediction})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')