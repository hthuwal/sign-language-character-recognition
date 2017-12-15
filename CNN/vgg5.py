import numpy as np
import os
import glob
import cv2
import pickle
import datetime
import pandas as pd
import time
from shutil import copy2
import warnings
warnings.filterwarnings("ignore")
from numpy.random import permutation
np.random.seed(2016)
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.utils import np_utils
from keras.models import model_from_json
from sklearn.metrics import log_loss
from keras.models import load_model
import h5py


use_cache = 0


def get_im_cv2(path):
    img = cv2.imread(path)
    resized = cv2.resize(img, (224, 224))
    return resized



def normalize_image(img):
    img[:, :, 0] -= 103.939
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 123.68
    return img


def load_train():
    X_train = []
    y_train = []
    start_time = time.time()

    print('Read train images')
    for j in range(36):
        print('Load folder {}'.format(j))
        path = os.path.join('train', str(j), '*.jpg')
        files = glob.glob(path)
        for fl in files:
            # flbase = os.path.basename(fl)
            img = get_im_cv2(fl)
            X_train.append(img)
            y_train.append(j)

    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return X_train, y_train


def cache_data(data, path):
    if os.path.isdir(os.path.dirname(path)):
        file = open(path, 'wb')
        pickle.dump(data, file)
        file.close()
    else:
        print('Directory doesnt exists')


def restore_data(path):
    data = dict()
    if os.path.isfile(path):
        file = open(path, 'rb')
        data = pickle.load(file)
    return data



def read_and_normalize_train_data():
    cache_path = os.path.join('cache', 'train_r_' + str(224) + '_c_' + str(224) + '_t_' + str(3) + '.dat')
    if not os.path.isfile(cache_path) or use_cache == 0:
        train_data, train_target = load_train()
        # cache_data((train_data, train_target, train_id, driver_id, unique_drivers), cache_path)
    else:
        print('Restore train from cache!')
        (train_data, train_target, train_id, driver_id, unique_drivers) = restore_data(cache_path)

    print('Convert to numpy...')
    train_data = np.array(train_data, dtype=np.uint8)
    train_target = np.array(train_target, dtype=np.uint8)

    print('Reshape...')
    train_data = train_data.transpose((0, 3, 1, 2))

    print('Convert to float...')
    train_data = train_data.astype('float16')
    mean_pixel = [103.939, 116.779, 123.68]
    print('Substract 0...')
    train_data[:, 0, :, :] -= mean_pixel[0]
    print('Substract 1...')
    train_data[:, 1, :, :] -= mean_pixel[1]
    print('Substract 2...')
    train_data[:, 2, :, :] -= mean_pixel[2]

    train_target = np_utils.to_categorical(train_target, 36)

    # Shuffle
    perm = permutation(len(train_target))
    train_data = train_data[perm]
    train_target = train_target[perm]

    print('T rain shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')
    return train_data, train_target


def VGG_16():
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, 224, 224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))

    f = h5py.File('weights/vgg16_weights.h5')
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            # we don't look at the last (fully-connected) layers in the savefile
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
    f.close()
    print('Model loaded.')

    model.add(Dense(36, activation='softmax'))

    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='val_loss', value=0.00001, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current < self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True


def run_cross_validation_create_models(nfolds=10):
    # input image dimensions
    batch_size = 128
    nb_epoch = 4
    random_state = 51
    restore_from_last_checkpoint = 1

    train_data, train_target = read_and_normalize_train_data()

    model = load_model('save_vgg_model.h5')
    #model = VGG_16()

    X_train, Y_train = train_data, train_target

    print('Split train: ', len(X_train), len(Y_train))

    X_test = X_train[11085:]
    Y_test = Y_train[11085:]

    X_train = X_train[:11085]
    Y_train = Y_train[:11085]

    callbacks = [
        EarlyStoppingByLossVal(monitor='val_loss', value=0.00001, verbose=1),
        EarlyStopping(monitor='val_loss', patience=5, verbose=0)
    ]
    
    guess = model.predict_classes(X_test,batch_size,1)

    f = open("table.txt","wb")
    for i in range(0,len(guess)):
    	print Y_test[i],
    	print "-->",
    	x = reversecategorical(Y_test[i],36)
    	print x,
    	print "-->",
    	print guess[i]
    	print >>f, Y_test[i],
    	print >>f, " --> "+ str(x) + " --> " + str(guess[i])+"\n"
    
    #model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
    #      shuffle=True, verbose=1, validation_split = 0.1)

    #model.save('save_vgg_model.h5')
    # score = model.evaluate(X_train, Y_train, show_accuracy=True, verbose=0)
    # print('Score log_loss: ', score)

def reversecategorical(a,nb):
	for i in range(0,nb):
		if(a[i]==1):
			return i;
	return -1;

if __name__ == '__main__':
    num_folds = 10
    if not os.path.isdir("subm"):
        os.mkdir("subm")
    if not os.path.isdir("cache"):
        os.mkdir("cache")
    if not os.path.isfile("weights/vgg16_weights.h5"):
        print('Please put VGG16 pretrained weights in weights/vgg16_weights.h5')
        exit()
    run_cross_validation_create_models(num_folds)

