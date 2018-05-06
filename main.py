# -*- coding: utf-8 -*-
import os
import json
import settings
from datetime import datetime
from fileProcess import FileReader, FileStore 
from preProcessData import FeatureExtraction

import cPickle as pickle
from gensim import corpora, matutils
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

class Classifier(object):
    def __init__(self, features_train = None, labels_train = None, features_test = None, labels_test = None, estimator=None):
        self.features_train = features_train
        self.features_test = features_test  
        self.labels_train = labels_train
        self.labels_test = labels_test
        self.estimator = estimator

    def training(self):
        print 'Tranning... ',  str(datetime.now())
        self.estimator.fit(self.features_train, self.labels_train)
        self.__training_result()
        print 'Tranning Done! ',  str(datetime.now())

    def save_model(self, filePath):
        print 'Saving Model... ',  str(datetime.now())
        FileStore(filePath=filePath).save_pickle(obj=self.estimator)
        print 'Save Model Done! ',  str(datetime.now())

    def __training_result(self):
        y_true, y_pred = self.labels_test, self.estimator.predict(self.features_test)
        print 'Accurancy: ',self.estimator.score(self.features_test,self.labels_test)
        print(classification_report(y_true, y_pred))

if __name__ == '__main__':
    # Read feature extraction 
    print 'Reading Feature Extraction... ',  str(datetime.now())
    features_test_loader = pickle.load(open(settings.FEATURES_TEST,'rb'))
    features_train_loader = pickle.load(open(settings.FEATURES_TRAIN,'rb'))
    features_train, labels_train = FeatureExtraction(data=features_train_loader).read_feature()
    features_test, labels_test = FeatureExtraction(data=features_test_loader).read_feature()
    print 'Read Feature Extraction Done! ',  str(datetime.now())

    # # KNeighbors Classifier
    # print 'Training by KNeighbors Classifier ...'
    # estKNeighbors = Classifier(features_train=features_train, features_test=features_test, labels_train=labels_train, labels_test=labels_test,estimator=KNeighborsClassifier(n_neighbors=3))
    # estKNeighbors.training()
    # estKNeighbors.save_model(filePath='trained_model/knn_model_tfidf.pk') # save Model
    # print 'Training by KNeighbors Classifier Done !'
    
    # SVM Classifier 
    print 'Training by SVM Classifier ...'
    estSVM = Classifier(features_train=features_train, features_test=features_test, labels_train=labels_train, labels_test=labels_test,estimator= LinearSVC())
    estSVM.training()
    estSVM.save_model(filePath='trained_model/svm_model.pk') # save Model
    print 'Training by SVM Classifier Done !'

    # # RandomForest Classifier
    # print 'Training by RandomForest Classifier ...'
    # estRandomForest = Classifier(features_train=features_train, features_test=features_test, labels_train=labels_train, labels_test=labels_test,estimator=RandomForestClassifier())
    # estRandomForest.training()
    # estRandomForest.save_model(filePath='trained_model/random_forest_model_tfidf.pk') # save Model
    # print 'Training by RandomForest Classifier Done ! '

    # Logistic_Classifier        
    print 'Training by Logistic_Classifier ...'
    estLogistic = Classifier(features_train=features_train, features_test=features_test, labels_train=labels_train, labels_test=labels_test,estimator=LogisticRegression())
    estLogistic.training()
    estLogistic.save_model(filePath='trained_model/logistic_model.pk') # save Model
    print 'Training by Logistic_Classifier Done !'
