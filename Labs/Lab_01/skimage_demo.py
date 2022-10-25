import matplotlib.pyplot as plt 

from skimage import data,filters

image = data.coins()
# ... or any other NumPy array!
plt.imshow(image, cmap='gray')
plt.show()

edges = filters.sobel(image)

plt.imshow(edges, cmap='gray')

plt.show()