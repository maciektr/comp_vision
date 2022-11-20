import matplotlib.pyplot as plt
import cv2 as cv
from tensorflow.keras.models import model_from_json
import os

# Reading the model from JSON file
with open("transfer_learning.json", "r") as json_file:
    json_savedModel = json_file.read()
# load the model architecture
# from tensorflow.keras.models import model_from_json
model_j = model_from_json(json_savedModel)
model_j.summary()

model_j.load_weights("transfer_learning.h5")

model = model_j

# ## Evaluate model performance
#
# We can see the final accuracy based on the test data,
# but typically we'll want to explore performance metrics in a little more depth.
# Let's plot a confusion matrix to see how well the model is predicting each class.


# Tensorflow doesn't have a built-in confusion matrix metric, so we'll use SciKit-Learn
import numpy as np
from sklearn.metrics import confusion_matrix

# import matplotlib.pyplot as plt


print("Generating predictions from validation data...")
from tensorflow.keras.preprocessing.image import ImageDataGenerator

classnames = os.listdir("data")
classnames.sort()

data_folder = "data"
pretrained_size = (224, 224)
batch_size = 32

print("Getting Data...")
datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.3)

print("Preparing validation dataset...")
validation_generator = datagen.flow_from_directory(
    data_folder,
    target_size=pretrained_size,  # resize to match model expected input
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation",
)  # set as validation data

# Get the image and label arrays for the first batch of validation data
x_test = validation_generator[0][0]
y_test = validation_generator[0][1]

# Use the model to predict the class
class_probabilities = model.predict(x_test)

# The model returns a probability value for each class
# The one with the highest probability is the predicted class
predictions = np.argmax(class_probabilities, axis=1)

# The actual labels are hot encoded (e.g. [0 1 0], so get the one with the value 1
true_labels = np.argmax(y_test, axis=1)

# Plot the confusion matrix
cm = confusion_matrix(true_labels, predictions)
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.colorbar()
tick_marks = np.arange(len(classnames))
plt.xticks(tick_marks, classnames, rotation=85)
plt.yticks(tick_marks, classnames)
plt.xlabel("Predicted Shape")
plt.ylabel("Actual Shape")
plt.show()


import tensorflow as tf

equality = tf.math.equal(predictions, true_labels)
accuracy = tf.math.reduce_mean(tf.cast(equality, tf.float32))
print("Accuracy: ", accuracy.numpy())

# ## Use the trained model
#
# Now that we've trained the model, we can use it to predict the class of an image.

from tensorflow.keras import models
import numpy as np
from random import randint
import os

# Function to predict the class of an image
def predict_image(classifier, image):
    from tensorflow import convert_to_tensor

    # The model expects a batch of images as input, so we'll create an array of 1 image
    imgfeatures = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])

    # We need to format the input to match the training data
    # The generator loaded the values as floating point numbers
    # and normalized the pixel values, so...
    imgfeatures = imgfeatures.astype("float32")
    imgfeatures /= 255

    # Use the model to predict the image class
    class_probabilities = classifier.predict(imgfeatures)

    # Find the class predictions with the highest predicted probability
    index = int(np.argmax(class_probabilities, axis=1)[0])
    return index


# Function to create a random image (of a square, circle, or triangle)
def create_image(size, shape):
    from random import randint
    import numpy as np
    from PIL import Image, ImageDraw

    xy1 = randint(10, 40)
    xy2 = randint(60, 100)
    col = (randint(0, 200), randint(0, 200), randint(0, 200))

    img = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(img)

    if shape == "circle":
        draw.ellipse([(xy1, xy1), (xy2, xy2)], fill=col)
    elif shape == "triangle":
        draw.polygon([(xy1, xy1), (xy2, xy2), (xy2, xy1)], fill=col)
    else:  # square
        draw.rectangle([(xy1, xy1), (xy2, xy2)], fill=col)
    del draw

    return np.array(img)


# Create a random test image
# classnames = os.listdir(os.path.join('data', 'shapes'))
classnames = os.listdir("data")
classnames.sort()
shape = classnames[randint(0, len(classnames) - 1)]
print("Shape: " + shape)
img = create_image((224, 224), shape)
plt.axis("off")
plt.imshow(img)

# Use the classifier to predict the class
class_idx = predict_image(model, img)
print("Prediction: ", classnames[class_idx])
