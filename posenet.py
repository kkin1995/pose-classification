import tensorflow as tf
import cv2

def return_keypoints_from_image(image):
    """
    Passes an image through the Pose Net model and returns the 17 keypoints in a dictionary

    Parameters:
    image - Numpy Array representing the image

    Return:
    keypoints_dict - A dictionary with the keys being the names of the body parts and the values being a list containing the
    y - coordinate, x - coordinate and the confidence score.
    """

    keypoint_dict = {}

    keypoint_names =  ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", \
        "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]

    image = tf.expand_dims(image, axis=0)
    image = tf.cast(tf.image.resize_with_pad(image, 192, 192), dtype=tf.int32)

    outputs = posenet_model(image)
    keypoints = outputs["output_0"].numpy()
    keypoints = keypoints.reshape((17, 3))

    for i in range(17):
        keypoint_dict[keypoint_names[i]] = keypoints[i, :]
    
    return keypoint_dict

def plot_keypoint_on_image(image):
    """
    Passes and image through the Pose Net model and plots the keypoints on a rescaled image

    Parameters:
    image - Numpy array representing the image

    Returns:
    image_numpy - Rescaled numpy array with the keypoints plotted
    """
    orig_image_shape = image.shape
    
    image_to_process = image.copy()
    
    image = tf.expand_dims(image, axis = 0)
    image = tf.cast(tf.image.resize_with_pad(image, orig_image_shape[1], orig_image_shape[1]), dtype = tf.int32)
    
    image_to_process = tf.expand_dims(image_to_process, axis=0)
    image_to_process = tf.cast(tf.image.resize_with_pad(image_to_process, 192, 192), dtype=tf.int32)

    outputs = posenet_model(image_to_process)
    keypoints = outputs["output_0"].numpy()
    keypoints = keypoints.reshape((17, 3))
    points = keypoints[:, 0:2] * orig_image_shape[1]

    image_numpy = image.numpy()
    image_numpy = image_numpy.reshape((orig_image_shape[1], orig_image_shape[1], 3))
    for i in range(17):
        if keypoints[i, 2] >= 0.3:
            image_numpy = cv2.circle(image_numpy, (int(points[i, 1]), int(points[i, 0])), radius=5, color=(255, 0, 0), thickness=-1)
    return image_numpy


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib import animation

    PLOT_KEYPOINT_ON_VIDEO = False
    PLOT_KEYPOINT_ON_IMAGE = True
    SAVE_KEYPOINTS = False

    # Posenet Model Download URL: https://tfhub.dev/google/movenet/singlepose/lightning/4
    posenet = tf.saved_model.load("models/movenet_singlepose_lightning_4")
    posenet_model = posenet.signatures["serving_default"]

    if PLOT_KEYPOINT_ON_VIDEO:
        total_time_counter = 100
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

    elif PLOT_KEYPOINT_ON_IMAGE:
        image_path = "sample_image.jpeg"
        image = cv2.imread(image_path)
        image = plot_keypoint_on_image(image)
        plt.imshow(image)
        plt.savefig(image_path.split(".")[0] + "_with_confidence_gt_0.3.jpeg")

    elif SAVE_KEYPOINTS:
        image_path = "sample_image.jpeg"
        image = cv2.imread(image_path)
        keypoint_dict = return_keypoints_from_image(image)

        with open(image_path.split(".")[0] + "_keypoints.yaml", "w") as f:
            f.write("File Name: " + image_path + "\n")
            for key, value in keypoint_dict.items():
                f.write(key + ": " + str(list(value)) + "\n")