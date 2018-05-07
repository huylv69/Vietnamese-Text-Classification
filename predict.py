"""
================
Phân loại văn bản Tiếng Việt
================

Chương trình phân loại văn bản Tiếng Việt.

"""

print(__doc__)
import numpy as np
from random import randint
import os
import json
import settings
import cPickle as pickle
from fileProcess import FileReader, FileStore 
from preProcessData import FeatureExtraction ,NLP
from pyvi import ViTokenizer
from sklearn.svm import LinearSVC
from gensim import corpora, matutils
from sklearn.metrics import classification_report
from argparse import ArgumentParser
import os.path

def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return open(arg, 'r')   # return an open file handle

def readInput():
    parser = ArgumentParser(description="ikjMatrix multiplication")
    parser.add_argument("-i", dest="filename", required=True,
                        help="File need predict", metavar="FILE",
                        type=lambda x: is_valid_file(parser, x))
    args = parser.parse_args()
    return args.filename.read()

if __name__ == '__main__':
    #  Read input data
    data = readInput()
    classifier = pickle.load(open('trained_model/logistic_model.pk','rb'))
    
    # bow
    # dense = FeatureExtraction(None).get_dense(text=data.decode('utf-16le'))
    # features = [] 
    # features.append(dense)
    
    # tf-idf
    vectorizer = pickle.load(open(settings.VECTOR_EMBEDDING,'rb'))
    data_features = []
    data_features.append(' '.join(NLP(text=data.decode('utf-16le')).get_words_feature()))
    features = vectorizer.transform(data_features)
    
    # predict result 
    print classifier.predict(features)