
# https://en.wikipedia.org/wiki/Python_Imaging_Library
# https://realpython.com/image-processing-with-the-python-pillow-library/

import glob
import os
from PIL import Image

'''
# create a thumbnail of an image
from PIL import Image
# load the image
image = Image.open('messi5.jpg')
# report the size of the image
print(image.size)
# create a thumbnail and *** preserve aspect ratio ***
image.thumbnail((100,100))
# report the size of the thumbnail
print(image.size)
'''


def main(max_px_size, images_folder):
    imgs = []
    imgs += sorted(glob.glob(os.path.join(images_folder, '*.jpg')))
    imgs += sorted(glob.glob(os.path.join(images_folder, '*.jpeg')))
    print('Image files to resize:')
    print('\n'.join(imgs))
    output_folder = '{}-{}px'.format(images_folder, max_px_size)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for img_path in imgs:
        try:
            resize(img_path, max_px_size, output_folder)
        except Exception as e:
            with open('error.txt', 'w') as f:
                f.write(str(e))

def resize(img_path, max_px_size, output_folder):
    with Image.open(img_path) as img:
        width_0, height_0 = img.size
        out_f_name = os.path.split(img_path)[-1]
        out_f_path = os.path.join(output_folder, out_f_name)

        if max((width_0, height_0)) <= max_px_size:
            print('writing {} to disk (no change from original)'.format(out_f_path))
            img.save(out_f_path)
            return

        if width_0 > height_0:
            wpercent = max_px_size / float(width_0)
            hsize = int(float(height_0) * float(wpercent))
            img = img.resize((max_px_size, hsize), Image.ANTIALIAS)
            print('writing {} to disk'.format(out_f_path))
            img.save(out_f_path)
            return
            
        if width_0 < height_0:
            hpercent = max_px_size / float(height_0)
            wsize = int(float(width_0) * float(hpercent))
            img = img.resize((wsize, max_px_size), Image.ANTIALIAS)
            print('writing {} to disk'.format(out_f_path))
            img.save(out_f_path)
            return

if __name__ == '__main__':
    max_px_size = 256
    images_folder = './'
    main(max_px_size, images_folder)