import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as tfhub
import cv2
import matplotlib.pyplot as plt
from matplotlib import animation

total_time_counter = 100

posenet = tfhub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")

posenet_model = posenet.signatures["serving_default"]

def plot_keypoint_on_image(image):
    orig_image_shape = image.shape
    
    image_to_process = image.copy()
    
    image = tf.expand_dims(image, axis = 0)
    image = tf.cast(tf.image.resize_with_pad(image, orig_image_shape[1], orig_image_shape[1]), dtype = tf.int32)
    
    image_to_process = tf.expand_dims(image_to_process, axis=0)
    image_to_process = tf.cast(tf.image.resize_with_pad(image_to_process, 192, 192), dtype=tf.int32)

    outputs = posenet_model(image_to_process)
    keypoints = outputs["output_0"]
    points = keypoints[:,:,:,0:2][0][0].numpy() * orig_image_shape[1]

    image_numpy = image.numpy()
    image_numpy = image_numpy.reshape((orig_image_shape[1], orig_image_shape[1], 3))
    for y, x in points:
        image_numpy = cv2.circle(image_numpy, (int(x),int(y)), radius=5, color=(255, 0, 0), thickness=-1)
    return image_numpy

counter = 0

video = cv2.VideoCapture(0)
_, frame = video.read()
fig = plt.figure()
im = plt.imshow(frame)

def init():
    im.set_data(frame)

def animate(i):
    _, frame = video.read()
    frame = plot_keypoint_on_image(frame)
    im.set_data(frame)
    return im

while counter < total_time_counter:
    anim = animation.FuncAnimation(fig, animate, init_func = init, frames = 128)
    counter += 1

anim.save("animation.mp4")
video.release()