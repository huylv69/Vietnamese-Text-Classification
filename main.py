# -*- coding: utf-8 -*-
import numpy as np
from random import randint
import os
import json
import settings
import cPickle as pickle
from fileProcess import FileReader, FileStore 
from preProcessData import FeatureExtraction
from pyvi import ViTokenizer
from sklearn.svm import LinearSVC
from gensim import corpora, matutils
from sklearn.metrics import classification_report

class Classifier(object):
    def __init__(self, features_train = None, labels_train = None, features_test = None, labels_test = None,  estimator = LinearSVC(random_state=0)):
        self.features_train = features_train
        self.features_test = features_test
        self.labels_train = labels_train
        self.labels_test = labels_test
        self.estimator = estimator

    def training(self):
        self.estimator.fit(self.features_train, self.labels_train)
        self.__training_result()

    def save_model(self, filePath):
        FileStore(filePath=filePath).save_pickle(obj=est)

    def __training_result(self):
        y_true, y_pred = self.labels_test, self.estimator.predict(self.features_test)
        # print ('y_pred' , y_pred , 'y_true' , y_true)
        print(classification_report(y_true, y_pred))

def get_feature_dict(value_features,value_labels):
    return {
            "features":value_features,
            "labels":value_labels
        }

if __name__ == '__main__':
      
    # Load data after preprocess 
    train_loader = FileReader(filePath=settings.DATA_TRAIN_JSON)
    test_loader = FileReader(filePath=settings.DATA_TEST_JSON)
    data_train = train_loader.read_json()
    data_test = test_loader.read_json()

    # Feature Extraction
    features_train, labels_train = FeatureExtraction(data=data_train).get_data_and_label()
    features_test, labels_test = FeatureExtraction(data=data_test).get_data_and_label()
    
    # Save feature extraction 
    features_train_dict = get_feature_dict(value_features=features_train,value_labels=labels_train)
    features_test_dict = get_feature_dict(value_features=features_test,value_labels=labels_test)
    FileStore(filePath=settings.FEATURES_TRAIN).save_pickle(obj=features_train_dict)
    FileStore(filePath=settings.FEATURES_TEST).save_pickle(obj=features_test_dict)

    # Read feature extraction 
    features_train_loader = pickle.load(open(settings.FEATURES_TRAIN,'rb'))
    features_test_loader = pickle.load(open(settings.FEATURES_TEST,'rb'))
    features_train, labels_train = FeatureExtraction(data=features_train_loader).read_feature()
    features_test, labels_test = FeatureExtraction(data=features_test_loader).read_feature()

    # Classifier
    est = Classifier(features_train=features_train, features_test=features_test, labels_train=labels_train, labels_test=labels_test)
    est.training()

    # save Model
    est.save_model(filePath='trained_model/linear_svc_model.pk')