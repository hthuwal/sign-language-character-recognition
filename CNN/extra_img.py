from __future__ import print_function
from __future__ import division

# ~/.keras/keras.json

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

import os
import glob
import pickle
import random
import numpy as np
import tensorflow as tf

from tqdm import tqdm

from sklearn.preprocessing import OneHotEncoder
from skimage.io import imread, imsave
from scipy.misc import imresize

WIDTH, HEIGHT = 224, 224

def load_image(path):
    return imresize(imread(path), (HEIGHT, WIDTH))

def load_test(base):
    paths = glob.glob('{}*.png'.format(base))

    print('Reading images...')
    for i, path in tqdm(enumerate(paths), total=len(paths)):
        datagen = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.1,
                zoom_range=0.1,
                horizontal_flip=False,
                fill_mode='nearest')
        id = os.path.basename(path)
        img = load_image(path)

        x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
        x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

        # the .flow() command below generates batches of randomly transformed images
        # and saves the results to the `preview/` directory
        i = 0
	if (base == 'database/0/'):
            dir = 'train/0'
        elif(base == 'database/1/'):
            dir = 'train/1'
        elif(base == 'database/2/'):
            dir = 'train/2'
        elif(base == 'database/3/'):
            dir = 'train/3'
        elif(base == 'database/4/'):
            dir = 'train/4'
        elif(base == 'database/5/'):
            dir = 'train/5'
        elif(base == 'database/6/'):
            dir = 'train/6'
        elif(base == 'database/7/'):
            dir = 'train/7'
        elif(base == 'database/8/'):
            dir = 'train/8'
        elif(base == 'database/9/'):
            dir = 'train/9'
        elif(base == 'database/a/'):
            dir = 'train/10'
        elif(base == 'database/b/'):
            dir = 'train/11'
        elif(base == 'database/c/'):
            dir = 'train/12'
        elif(base == 'database/d/'):
            dir = 'train/13'
        elif(base == 'database/e/'):
            dir = 'train/14'
        elif(base == 'database/f/'):
            dir = 'train/15'
        elif(base == 'database/g/'):
            dir = 'train/16'
        elif(base == 'database/h/'):
            dir = 'train/17'
        elif(base == 'database/i/'):
            dir = 'train/18'
        elif(base == 'database/j/'):
            dir = 'train/19'
        elif(base == 'database/k/'):
            dir = 'train/20'
        elif(base == 'database/l/'):
            dir = 'train/21'
        elif(base == 'database/m/'):
            dir = 'train/22'
        elif(base == 'database/n/'):
            dir = 'train/23'
        elif(base == 'database/o/'):
            dir = 'train/24'
        elif(base == 'database/p/'):
            dir = 'train/25'
        elif(base == 'database/q/'):
            dir = 'train/26'
        elif(base == 'database/r/'):
            dir = 'train/27'
        elif(base == 'database/s/'):
            dir = 'train/28'
        elif(base == 'database/t/'):
            dir = 'train/29'
        elif(base == 'database/u/'):
            dir = 'train/30'
        elif(base == 'database/v/'):
            dir = 'train/31'
        elif(base == 'database/w/'):
            dir = 'train/32'
        elif(base == 'database/x/'):
            dir = 'train/33'
        elif(base == 'database/y/'):
            dir = 'train/34'
        elif(base == 'database/z/'):
            dir = 'train/35'


        # print (dir)
        for batch in datagen.flow(x, batch_size=1,
                                  save_to_dir=dir, save_prefix='gesture', save_format='jpg'):
            i += 1
            if i > 5:
                break  # otherwise the generator would loop indefinitely

load_test('database/0/')
load_test('database/1/')
load_test('database/2/')
load_test('database/3/')
load_test('database/4/')
load_test('database/5/')
load_test('database/6/')
load_test('database/7/')
load_test('database/8/')
load_test('database/9/')
load_test('database/a/')
load_test('database/b/')
load_test('database/c/')
load_test('database/d/')
load_test('database/e/')
load_test('database/f/')
load_test('database/g/')
load_test('database/h/')
load_test('database/i/')
load_test('database/j/')
load_test('database/k/')
load_test('database/l/')
load_test('database/m/')
load_test('database/n/')
load_test('database/o/')
load_test('database/p/')
load_test('database/q/')
load_test('database/r/')
load_test('database/s/')
load_test('database/t/')
load_test('database/u/')
load_test('database/v/')
load_test('database/w/')
load_test('database/x/')
load_test('database/y/')
load_test('database/z/')

