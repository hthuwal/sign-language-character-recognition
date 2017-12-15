""" 
Python module for use with David Lowe's SIFT code available at:
http://www.cs.ubc.ca/~lowe/keypoints/
adapted from the matlab code examples.
Jan Erik Solem, 2009-01-30
http://www.janeriksolem.net/2009/02/sift-python-implementation.html
"""
from os.path import exists, isdir, basename, join, splitext, dirname
from os import makedirs
import os
import subprocess
from PIL import Image
from numpy import *
import numpy
import cPickle
import pylab
from os.path import exists
MAXSIZE = 1024
VERBOSE = True
WRITE_VERBOSE = False  # no verbose reading atm


def process_image(imagename, resultname='temp.sift'):
    """ process an image and save the results in a .key ascii file"""
    print "working on ", imagename
    # if dense == False:
    if imagename[-3:] != 'pgm':
        # create a pgm file, image is resized, if it is too big.
        # sift returns an error if more than 8000 features are found
        size = (MAXSIZE, MAXSIZE)
        im = Image.open(imagename).convert('L')
        im.thumbnail(size, Image.ANTIALIAS)
        im.save('tmp.pgm')
        imagename = 'tmp.pgm'

    # check if linux or windows
    if os.name == "posix":
        cmmd = "./sift < " + imagename + " > " + resultname
    else:
        cmmd = "siftWin32 < " + imagename + " > " + resultname

    # run extraction command
    returnvalue = subprocess.call(cmmd, shell=True)
    if returnvalue == 127:
        # removing empty resultfile created by output redirection
        os.remove(resultname)
        raise IOError("SIFT executable not found")
    if returnvalue == 2:
        # removing empty resultfile created by output redirection
        os.remove(resultname)
        raise IOError("image " + imagename + " not found")
    if os.path.getsize(resultname) == 0:
        raise IOError("extracting SIFT features failed " + resultname)


def read_features_from_file(filename='temp.sift', dense=False):
    """ read feature properties and return in matrix form"""

    if exists(filename) != False | os.path.getsize(filename) == 0:
        raise IOError("wrong file path or file empty: " + filename)
    if dense == True:
        with open(filename, 'rb') as f:
            locs, descriptors = cPickle.load(f)
    else:
        f = open(filename, 'r')
        header = f.readline().split()

        num = int(header[0])  # the number of features
        featlength = int(header[1])  # the length of the descriptor
        if featlength != 128:  # should be 128 in this case
            raise RuntimeError(
                'Keypoint descriptor length invalid (should be 128).')

        locs = zeros((num, 4))
        descriptors = zeros((num, featlength))

        # parse the .key file
        e = f.read().split()  # split the rest into individual elements
        pos = 0
        for point in range(num):
            # row, col, scale, orientation of each feature
            for i in range(4):
                locs[point, i] = float(e[pos + i])
            pos += 4

            # the descriptor values of each feature
            for i in range(featlength):
                descriptors[point, i] = int(e[pos + i])
            # print descriptors[point]
            pos += 128

            # normalize each input vector to unit length
            descriptors[point] = descriptors[point] / \
                linalg.norm(descriptors[point])  # doubt!!
            # print descriptors[point]

        f.close()

    return locs, descriptors
