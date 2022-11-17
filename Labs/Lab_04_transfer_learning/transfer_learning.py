# coding: utf-8

# # Transfer Learning
#


import tensorflow
from  tensorflow import keras
print('TensorFlow version:',tensorflow.__version__)
print('Keras version:',keras.__version__)

import matplotlib.pyplot as plt
import cv2 as cv

from tensorflow.keras.applications.vgg16 import VGG16

model = VGG16()

# summarize filter shapes
for layer in model.layers:
	# check for convolutional layer
	if 'conv' not in layer.name:
		continue
	# get filter weights
	filters, biases = layer.get_weights()
	print(layer.name, filters.shape)
print('=======================================')
print('\n')


from matplotlib import pyplot
# retrieve weights from the second hidden layer
filters, biases = model.layers[1].get_weights()
# normalize filter values to 0-1 so we can visualize them
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)
# plot first few filters
n_filters, ix = 6, 1
for i in range(n_filters):
	# get the filter
	f = filters[:, :, :, i]
	# plot each channel separately
	for j in range(3):
		# specify subplot and turn of axis
		ax = pyplot.subplot(n_filters, 3, ix)
		ax.set_xticks([])
		ax.set_yticks([])
		# plot filter channel in grayscale
		pyplot.imshow(f[:, :, j], cmap='gray')
		ix += 1
# show the figure
pyplot.show()


# plot feature map of first conv layer for given image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model
from numpy import expand_dims


# redefine model to output right after the first hidden layer
layer_number = 1
#layer_number = 14
model = Model(inputs=model.inputs, outputs=model.layers[layer_number].output)
model.summary()

# load the image with the required shape
img = load_img('bird.jpg', target_size=(224, 224))
# convert the image to an array
img = img_to_array(img)
# expand dimensions so that it represents a single 'sample'
img = expand_dims(img, axis=0)
# prepare the image (e.g. scale pixel values for the vgg)
img = preprocess_input(img)
# get feature map for first hidden layer

feature_maps = model.predict( img)
print('feature_maps.shape={}'.format(feature_maps.shape))

# plot all 64 maps in an 8x8 squares
square = 8
ix = 1
for _ in range(square):
	for _ in range(square):
		# specify subplot and turn of axis
		ax = pyplot.subplot(square, square, ix)
		ax.set_xticks([])
		ax.set_yticks([])
		# plot filter channel in grayscale
		pyplot.imshow(feature_maps[0, :, :, ix-1]) #, cmap='gray')
		ix += 1
# show the figure
pyplot.show()
plt.show()

del model

print('\n')
print('------------ Transfer learning -----------------\n')
print('\n')


# ## Prepare the base model

base_model = keras.applications.resnet.ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
#base_model = keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))


print(base_model.summary())

from tensorflow.keras.preprocessing.image import ImageDataGenerator

data_folder = 'data'
pretrained_size = (224,224)
batch_size = 32

print("Getting Data...")
datagen = ImageDataGenerator(rescale=1./255,
                             validation_split=0.3)

print("Preparing training dataset...")
train_generator = datagen.flow_from_directory(
    data_folder,
    target_size=pretrained_size, 		# resize to match model expected input
    batch_size=batch_size,
    class_mode='categorical',
    subset='training') 					# set as training data

print("Preparing validation dataset...")
validation_generator = datagen.flow_from_directory(
    data_folder,
    target_size=pretrained_size, 		# resize to match model expected input
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation') 				# set as validation data

classnames = list(train_generator.class_indices.keys())
print("class names: ", classnames)


# ## Create a prediction layer
#
# We downloaded the complete *resnet* model excluding its final prediction layer,
# so need to combine these layers with a fully-connected (*dense*) layer
# that takes the flattened outputs from the feature extraction layers
# and generates a prediction for each of our image classes.
#
# We also need to freeze the feature extraction layers to retain the trained weights.
# Then when we train the model using our images,
# only the final prediction layer will learn new weight and bias values - the pre-trained weights
# already learned for feature extraction will remain the same.


from tensorflow.keras import applications
from tensorflow.keras import Model
from tensorflow.keras.layers import Flatten, Dense

# Freeze the already-trained layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# Create prediction layer for classification of our images
x = base_model.output
x = Flatten()(x)
prediction_layer = Dense(len(classnames), activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=prediction_layer)

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])



# Now print the full model, which will include the layers of the base model plus the dense layer we added
print(model.summary())


# ## Train the Model
#
# With the layers of the CNN defined, we're ready to train it using our image data.
# The weights used in the feature extraction layers from the base resnet model will not be changed
# by training, only the final dense layer that maps the features to our shape classes will be trained.

# Train the model over 3 epochs
num_epochs = 3
history = model.fit(
    train_generator,
    steps_per_epoch = train_generator.samples // batch_size,
    validation_data = validation_generator,
    validation_steps = validation_generator.samples // batch_size,
    epochs = num_epochs)


# ## View the loss history
#
# We tracked average training and validation loss for each epoch.
# We can plot these to verify that the loss reduced over the training process and to detect *over-fitting* (which is indicated by a continued drop in training loss after validation loss has levelled out or started to increase).



from matplotlib import pyplot as plt

print('history.history.keys()={}'.format(history.history.keys() ))

epoch_nums = range(1,num_epochs+1)
training_loss = history.history["loss"]
validation_loss = history.history["val_loss"]
plt.plot(epoch_nums, training_loss)
plt.plot(epoch_nums, validation_loss)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['training', 'validation'], loc='upper right')
plt.show()

from tensorflow.keras.models import model_from_json
# serialize model to json
json_model = model.to_json()
#save the model architecture to JSON file
with open('transfer_learning.json', 'w') as json_file:
    json_file.write(json_model)
#saving the weights of the model
model.save_weights('transfer_learning.h5')

#Reading the model from JSON file
with open('transfer_learning.json', 'r') as json_file:
    json_savedModel= json_file.read()
#load the model architecture
#from tensorflow.keras.models import model_from_json
model_j = model_from_json(json_savedModel)
model_j.summary()

model_j.load_weights('transfer_learning.h5')

model = model_j

# ## Evaluate model performance
#
# We can see the final accuracy based on the test data,
# but typically we'll want to explore performance metrics in a little more depth.
# Let's plot a confusion matrix to see how well the model is predicting each class.


# Tensorflow doesn't have a built-in confusion matrix metric, so we'll use SciKit-Learn
import numpy as np
from sklearn.metrics import confusion_matrix
#import matplotlib.pyplot as plt


print("Generating predictions from validation data...")
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
    imgfeatures = imgfeatures.astype('float32')
    imgfeatures /= 255

    # Use the model to predict the image class
    class_probabilities = classifier.predict(imgfeatures)

    # Find the class predictions with the highest predicted probability
    index = int(np.argmax(class_probabilities, axis=1)[0])
    return index

# Function to create a random image (of a square, circle, or triangle)
def create_image (size, shape):
    from random import randint
    import numpy as np
    from PIL import Image, ImageDraw

    xy1 = randint(10,40)
    xy2 = randint(60,100)
    col = (randint(0,200), randint(0,200), randint(0,200))

    img = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(img)

    if shape == 'circle':
        draw.ellipse([(xy1,xy1), (xy2,xy2)], fill=col)
    elif shape == 'triangle':
        draw.polygon([(xy1,xy1), (xy2,xy2), (xy2,xy1)], fill=col)
    else: # square
        draw.rectangle([(xy1,xy1), (xy2,xy2)], fill=col)
    del draw

    return np.array(img)

# Create a random test image
# classnames = os.listdir(os.path.join('data', 'shapes'))
classnames = os.listdir('data')
classnames.sort()
img = create_image ((224,224), classnames[randint(0, len(classnames)-1)])
plt.axis('off')
plt.imshow(img)

# Use the classifier to predict the class
class_idx = predict_image(model, img)
print (classnames[class_idx])
