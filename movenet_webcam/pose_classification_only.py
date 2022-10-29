import csv
import cv2
import itertools
import numpy as np
import pandas as pd
import os
import sys
import tempfile
import tqdm
import queue
from queue import Queue
from threading import Thread
from typing import Iterator, Union, Tuple

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import time
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

pose_sample_rpi_path = os.path.join(
    os.getcwd(), "examples/lite/examples/pose_estimation/raspberry_pi"
)
sys.path.append(pose_sample_rpi_path)

# Load MoveNet Thunder model
import utils
from data import BodyPart
from ml import Movenet

movenet = Movenet("movenet_thunder")

# Define function to run pose estimation using MoveNet Thunder.
# You'll apply MoveNet's cropping algorithm and run inference multiple times on
# the input image to improve pose estimation accuracy.
def detect(input_tensor, inference_count=3):
    """Runs detection on an input image.

    Args:
      input_tensor: A [height, width, 3] Tensor of type tf.float32.
        Note that height and width can be anything since the image will be
        immediately resized according to the needs of the model within this
        function.
      inference_count: Number of times the model should run repeatly on the
        same input image to improve detection accuracy.

    Returns:
      A Person entity detected by the MoveNet.SinglePose.
    """
    image_height, image_width, channel = input_tensor.shape

    # Detect pose using the full input image
    movenet.detect(input_tensor.numpy(), reset_crop_region=True)

    # Repeatedly using previous detection result to identify the region of
    # interest and only croping that region to improve detection accuracy
    for _ in range(inference_count - 1):
        person = movenet.detect(input_tensor.numpy(), reset_crop_region=False)

    return person


# In[5]:

# ==================================================================================
# @title Functions to visualize the pose estimation results.


def draw_prediction_on_image(
    image, person, crop_region=None, close_figure=True, keep_input_size=False
):
    """Draws the keypoint predictions on image.

    Args:
      image: An numpy array with shape [height, width, channel] representing the
        pixel values of the input image.
      person: A person entity returned from the MoveNet.SinglePose model.
      close_figure: Whether to close the plt figure after the function returns.
      keep_input_size: Whether to keep the size of the input image.

    Returns:
      An numpy array with shape [out_height, out_width, channel] representing the
      image overlaid with keypoint predictions.
    """
    # Draw the detection result on top of the image.
    image_np = utils.visualize(image, [person])

    # Plot the image with detection results.
    height, width, channel = image.shape
    aspect_ratio = float(width) / height
    fig, ax = plt.subplots(figsize=(12 * aspect_ratio, 12))
    im = ax.imshow(image_np)
    plt.show()

    if close_figure:
        plt.close(fig)

    if not keep_input_size:
        image_np = utils.keep_aspect_ratio_resizer(image_np, (512, 512))

    return image_np


# ===========================================================
# -----------------------------------------------------------
# ===========================================================
is_skip_step_1 = False  # @param ["False", "True"] {type:"raw"}
is_skip_step_1 = True


print("\n")
print("Train a pose classification model")
print("\n")

# ## Part 2: Train a pose classification model that takes the landmark coordinates as input, and output the predicted labels.
#
# You'll build a TensorFlow model that takes the landmark coordinates and predicts the pose class that the person in the input image performs. The model consists of two submodels:
#
# * Submodel 1 calculates a pose embedding (a.k.a feature vector) from the detected landmark coordinates.
# * Submodel 2 feeds pose embedding through several `Dense` layer to predict the pose class.
#
# You'll then train the model based on the dataset that were preprocessed in part 1.

# ### (Optional) Download the preprocessed dataset if you didn't run part 1

# In[15]:

# Download the preprocessed CSV files which are the same as the output of step 1
if is_skip_step_1:
    # get_ipython().system('wget -O train_data.csv http://download.tensorflow.org/data/pose_classification/yoga_train_data.csv')
    # get_ipython().system('wget -O test_data.csv http://download.tensorflow.org/data/pose_classification/yoga_test_data.csv')

    csvs_out_train_path = "train_data.csv"
    csvs_out_test_path = "test_data.csv"
    is_skipped_step_1 = True


# ### Load the preprocessed CSVs into `TRAIN` and `TEST` datasets.

# In[16]:


def load_pose_landmarks(csv_path):
    """Loads a CSV created by MoveNetPreprocessor.

    Returns:
      X: Detected landmark coordinates and scores of shape (N, 17 * 3)
      y: Ground truth labels of shape (N, label_count)
      classes: The list of all class names found in the dataset
      dataframe: The CSV loaded as a Pandas dataframe features (X) and ground
        truth labels (y) to use later to train a pose classification model.
    """

    # Load the CSV file
    # print('load_pose_landmarks() - csv_path={}'.format(csv_path))
    dataframe = pd.read_csv(csv_path)
    df_to_process = dataframe.copy()

    # Drop the file_name columns as you don't need it during training.
    df_to_process.drop(columns=["file_name"], inplace=True)

    # Extract the list of class names
    classes = df_to_process.pop("class_name").unique()

    # Extract the labels
    y = df_to_process.pop("class_no")

    # Convert the input features and labels into the correct format for training.
    X = df_to_process.astype("float64")
    y = keras.utils.to_categorical(y)

    return X, y, classes, dataframe


# Load and split the original `TRAIN` dataset into `TRAIN` (85% of the data) and `VALIDATE` (the remaining 15%).

# In[17]:

# Load the train data
X, y, class_names, _ = load_pose_landmarks(csvs_out_train_path)

# Split training data (X, y) into (X_train, y_train) and (X_val, y_val)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)


# In[18]:

# Load the test data
X_test, y_test, _, df_test = load_pose_landmarks(csvs_out_test_path)


# ### Define functions to convert the pose landmarks to a pose embedding (a.k.a. feature vector) for pose classification
#
# Next, convert the landmark coordinates to a feature vector by:
# 1. Moving the pose center to the origin.
# 2. Scaling the pose so that the pose size becomes 1
# 3. Flattening these coordinates into a feature vector
#
# Then use this feature vector to train a neural-network based pose classifier.

# In[19]:


def get_center_point(landmarks, left_bodypart, right_bodypart):
    """Calculates the center point of the two given landmarks."""

    left = tf.gather(landmarks, left_bodypart.value, axis=1)
    right = tf.gather(landmarks, right_bodypart.value, axis=1)
    center = left * 0.5 + right * 0.5
    return center


def get_pose_size(landmarks, torso_size_multiplier=2.5):
    """Calculates pose size.

    It is the maximum of two values:
      * Torso size multiplied by `torso_size_multiplier`
      * Maximum distance from pose center to any pose landmark
    """
    # Hips center
    hips_center = get_center_point(landmarks, BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP)

    # Shoulders center
    shoulders_center = get_center_point(
        landmarks, BodyPart.LEFT_SHOULDER, BodyPart.RIGHT_SHOULDER
    )

    # Torso size as the minimum body size
    torso_size = tf.linalg.norm(shoulders_center - hips_center)

    # Pose center
    pose_center_new = get_center_point(landmarks, BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP)
    pose_center_new = tf.expand_dims(pose_center_new, axis=1)
    # Broadcast the pose center to the same size as the landmark vector to
    # perform substraction
    pose_center_new = tf.broadcast_to(
        pose_center_new, [tf.size(landmarks) // (17 * 2), 17, 2]
    )

    # Dist to pose center
    d = tf.gather(landmarks - pose_center_new, 0, axis=0, name="dist_to_pose_center")
    # Max dist to pose center
    max_dist = tf.reduce_max(tf.linalg.norm(d, axis=0))

    # Normalize scale
    pose_size = tf.maximum(torso_size * torso_size_multiplier, max_dist)

    return pose_size


def normalize_pose_landmarks(landmarks):
    """Normalizes the landmarks translation by moving the pose center to (0,0) and
    scaling it to a constant pose size.
    """
    # Move landmarks so that the pose center becomes (0,0)
    pose_center = get_center_point(landmarks, BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP)
    pose_center = tf.expand_dims(pose_center, axis=1)
    # Broadcast the pose center to the same size as the landmark vector to perform
    # substraction
    pose_center = tf.broadcast_to(pose_center, [tf.size(landmarks) // (17 * 2), 17, 2])
    landmarks = landmarks - pose_center

    # Scale the landmarks to a constant pose size
    pose_size = get_pose_size(landmarks)
    landmarks /= pose_size

    return landmarks


def landmarks_to_embedding(landmarks_and_scores):
    """Converts the input landmarks into a pose embedding."""
    # Reshape the flat input into a matrix with shape=(17, 3)
    reshaped_inputs = keras.layers.Reshape((17, 3))(landmarks_and_scores)

    # Normalize landmarks 2D
    landmarks = normalize_pose_landmarks(reshaped_inputs[:, :, :2])

    # Flatten the normalized landmark coordinates into a vector
    embedding = keras.layers.Flatten()(landmarks)

    return embedding


# ### Define a Keras model for pose classification
#
# Our Keras model takes the detected pose landmarks, then calculates the pose embedding and predicts the pose class.

# In[20]:

# Define the model
inputs = tf.keras.Input(shape=(51))
embedding = landmarks_to_embedding(inputs)

layer = keras.layers.Dense(128, activation=tf.nn.relu6)(embedding)
layer = keras.layers.Dropout(0.5)(layer)
layer = keras.layers.Dense(64, activation=tf.nn.relu6)(layer)
layer = keras.layers.Dropout(0.5)(layer)
outputs = keras.layers.Dense(len(class_names), activation="softmax")(layer)

model = keras.Model(inputs, outputs)
model.summary()


# In[21]:

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Add a checkpoint callback to store the checkpoint that has the highest
# validation accuracy.
checkpoint_path = "weights.best.hdf5"
checkpoint = keras.callbacks.ModelCheckpoint(
    checkpoint_path, monitor="val_accuracy", verbose=1, save_best_only=True, mode="max"
)
earlystopping = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=20)

# Start training
history = model.fit(
    X_train,
    y_train,
    epochs=200,
    batch_size=16,
    validation_data=(X_val, y_val),
    callbacks=[checkpoint, earlystopping],
)


# In[22]:

# Visualize the training history to see whether you're overfitting.
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["TRAIN", "VAL"], loc="lower right")
# plt.show()


# In[23]:

# Evaluate the model using the TEST dataset
loss, accuracy = model.evaluate(X_test, y_test)


# ### Draw the confusion matrix to better understand the model performance

# In[24]:


def plot_confusion_matrix(
    cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues
):
    """Plots the confusion matrix."""
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=55)
    plt.yticks(tick_marks, classes)
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    # plt.show()


# Classify pose in the TEST dataset using the trained model
y_pred = model.predict(X_test)

# Convert the prediction result to class name
y_pred_label = [class_names[i] for i in np.argmax(y_pred, axis=1)]
y_true_label = [class_names[i] for i in np.argmax(y_test, axis=1)]

# Plot the confusion matrix
cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
plot_confusion_matrix(
    cm, class_names, title="Confusion Matrix of Pose Classification Model"
)

# Print the classification report
print("\nClassification Report:\n", classification_report(y_true_label, y_pred_label))


# ### (Optional) Investigate incorrect predictions
#
# You can look at the poses from the `TEST` dataset that were incorrectly predicted to see whether the model accuracy can be improved.
#
# Note: This only works if you have run step 1 because you need the pose image files on your local machine to display them.

IMAGE_PER_ROW = 3
MAX_NO_OF_IMAGE_TO_PLOT = 30

# Extract the list of eincorrectly predicted poses
false_predict = [
    id_in_df
    for id_in_df in range(len(y_test))
    if y_pred_label[id_in_df] != y_true_label[id_in_df]
]
if len(false_predict) > MAX_NO_OF_IMAGE_TO_PLOT:
    false_predict = false_predict[:MAX_NO_OF_IMAGE_TO_PLOT]
images_out_test_folder = os.path.abspath("./yoga_cg/test")
# Plot the incorrectly predicted images
row_count = len(false_predict) // IMAGE_PER_ROW + 1
fig = plt.figure(figsize=(10 * IMAGE_PER_ROW, 10 * row_count))
for i, id_in_df in enumerate(false_predict):
    ax = fig.add_subplot(row_count, IMAGE_PER_ROW, i + 1)
    image_path = os.path.join(
        images_out_test_folder, df_test.iloc[id_in_df]["file_name"]
    )

    image = cv2.imread(image_path)
    plt.title(
        "Predict: %s; Actual: %s" % (y_pred_label[id_in_df], y_true_label[id_in_df])
    )
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.show()


class FrameSource(Iterator):
    def __init__(self, video_path: Union[str, int] = 0, flush=False):
        self.capture = cv2.VideoCapture(video_path)
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)

        self.block_buffer_reader = not flush
        self.counter = 0

        self.buffer = Queue(1)
        self.buffer_reader = Thread(target=self._read_buffer, daemon=True)

    def __iter__(self):
        self.buffer_reader.start()
        return self

    def __next__(self) -> Tuple[int, float, np.ndarray]:
        available, counter, frame = self.buffer.get()
        if not available:
            self.capture.release()
            raise StopIteration
        return self.counter, self.fps, frame

    def __del__(self):
        self.capture.release()

    def _read_buffer(self):
        while True:
            available, frame = self.capture.read()
            self.counter += 1
            try:
                self.buffer.put(
                    block=self.block_buffer_reader,
                    item=(available, self.counter, frame),
                )
            except queue.Full:
                self.buffer.get()
                self.buffer.put(item=(available, self.counter, frame))
            if not available:
                break


def draw_prediction(img, prediction: str):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 25)
    fontScale = 1
    fontColor = (55, 55, 240) # red
    thickness = 2
    lineType = 2

    cv2.putText(
        img,
        prediction,
        bottomLeftCornerOfText,
        font,
        fontScale,
        fontColor,
        thickness,
        lineType,
    )


if __name__ == "__main__":
    video_device_id = 1
    source = FrameSource(video_device_id)

    for counter, fps, frame in source:
        read_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        person = movenet.detect(read_frame, reset_crop_region=True)
        pose_landmarks = np.array(
            [
                [keypoint.coordinate.x, keypoint.coordinate.y, keypoint.score]
                for keypoint in person.keypoints
            ],
            dtype=np.float32,
        ).flatten()
        pose_landmarks = np.reshape(pose_landmarks, (1, pose_landmarks.shape[0]))

        prediction = model.predict(pose_landmarks)
        prediction_label = class_names[int(np.argmax(prediction))]
        print(prediction_label)

        image_np = utils.visualize(frame, [person])
        draw_prediction(image_np, prediction_label)

        cv2.imshow("capture", image_np)
        cv2.waitKey(1)

    cv2.destroyAllWindows()
