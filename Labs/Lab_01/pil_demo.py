from PIL import Image, ImageFilter
#Read image
im = Image.open( 'messi5.jpg' )

print(im.format)
print(im.mode)
print(im.size)


#Display image
im.show()

from PIL import ImageEnhance

enh = ImageEnhance.Contrast(im)

enh.enhance(1.8).show("30% more contrast")