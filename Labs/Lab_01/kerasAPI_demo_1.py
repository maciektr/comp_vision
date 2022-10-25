
from keras.preprocessing.image import load_img
import warnings

# load the image
img = load_img('messi5.jpg')
# report details about the image
print(type(img))
print(img.format)
print(img.mode)
print(img.size)
# show the image
#img.show()