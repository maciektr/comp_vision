
# example of converting an image with the Keras API
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img

# load the image
img = load_img('messi5.jpg')
print("Orignal:" ,type(img))

# convert to numpy array
img_array = img_to_array(img)
print("NumPy array info:") 
print(type(img_array))    

print("type:",img_array.dtype)
print("shape:",img_array.shape)
# convert back to image

img_pil = array_to_img(img_array)
print("converting NumPy array:",type(img_pil))



# example of saving an image with the Keras API
from keras.preprocessing.image import load_img
from keras.preprocessing.image import save_img
# save the image with a new filename
save_img('messi5_keras.png', img_array)
# load the image to confirm it was saved correctly
img = load_img('messi5_keras.png')
print(type(img))
print(img.format)
print(img.mode)
print(img.size)
img.show()