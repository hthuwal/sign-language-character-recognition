import libsvm
import os
import argparse
from cPickle import load
from learn import extractSift, computeHistograms, writeHistogramsToFile
import pickle
from cPickle import dump
import numpy as np

EXTENSIONS = [".jpg", ".bmp", ".png", ".pgm", ".tif", ".tiff"]
HISTOGRAMS_FILE = '../testdata.svm'
CODEBOOK_FILE = '../databasecodebook.file'
MODEL_FILE = '../databasetrainingdata.svm.model'


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='classify images with a visual bag of words model')
    parser.add_argument('-c', help='path to the codebook file',
                        required=False, default=CODEBOOK_FILE)
    parser.add_argument('-m', help='path to the model  file',
                        required=False, default=MODEL_FILE)
    parser.add_argument('input_images', help='images to classify', nargs='+')
    args = parser.parse_args()
    return args


def findkey(a, x):
    for i in a:
        if (a[i] == x):
            return i

print "---------------------"
print "## extract Sift features"
all_files = []
all_files_labels = {}
all_features = {}

args = parse_arguments()
model_file = args.m
codebook_file = args.c
farr = []
val = []
predict = []
fol = args.input_images
fl = os.listdir(fol[0])
fl.sort()
dct = pickle.load(open("dict.p", "rb"))

print "---------------------"
print "## loading codebook from " + codebook_file

with open(codebook_file, 'rb') as f:
    codebook = load(f)

nclusters = codebook.shape[0]

for f in fl:

    fnames = [fol[0]+'/'+f]
    if(os.path.isfile(fnames[0])):
        farr.append(f)
        all_features = extractSift(fnames)
        for i in fnames:
            all_files_labels[i] = 0  # label is unknown

        print "---------------------"

        print "## computing visual word histograms"

        all_word_histgrams = {}
        for imagefname in all_features:
            word_histgram = computeHistograms(
                codebook, all_features[imagefname], imagefname)
            all_word_histgrams[imagefname] = word_histgram

        print "---------------------"
        print "## write the histograms to file to pass it to the svm"
        writeHistogramsToFile(nclusters,
                              all_files_labels,
                              fnames,
                              all_word_histgrams,
                              HISTOGRAMS_FILE)

        print "---------------------"
        print "## test data with svm"
        hold = libsvm.test(HISTOGRAMS_FILE, model_file)
        vl = hold[0]
        val.append(vl)
        predict.append(findkey(dct, vl))
        # print hold

f = open("table.txt", "wb")
string = ""
for i in range(0, len(farr)):
    print farr[i]+"--->"+predict[i]
    string = string + farr[i]+"--->"+predict[i]+"\n"

f.write(string)
