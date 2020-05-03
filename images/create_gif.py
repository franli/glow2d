"""
Create GIF of the training images in this folder.
"""
import os
import glob
import imageio

filenames = glob.glob('*.png')
filenames.sort(key=lambda x: int(x[6:-4]))  # e p o c h _ [XXX] .p n g

images = [imageio.imread(filename) for filename in filenames]

path = 'v.gif'  # use your own path


imageio.mimsave(path, images, duration=0.2)
os.system('gifsicle --scale 0.7 -O3 {} -o {} '.format(path, path))

print('Done.')