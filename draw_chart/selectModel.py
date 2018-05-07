# -*- coding: utf-8 -*-

# import vectorize_data as vd
import settings
import pickle as pickle
from preProcessData import FeatureExtraction
import numpy as np
# X_train, y_train, X_test, y_test = vd.tf_Idf('./dataS/train/pre_train.txt', './dataS/test/pre_test.txt')
# X_train, y_train, X_test, y_test = vd.Bow('./data/train/pre_train.txt', './data/test/pre_test.txt')
features_test_loader = pickle.load(open(settings.FEATURES_TEST,'rb'))
features_train_loader = pickle.load(open(settings.FEATURES_TRAIN,'rb'))
features_train, labels_train = FeatureExtraction(data=features_train_loader).read_feature()
features_test, labels_test = FeatureExtraction(data=features_test_loader).read_feature()

X_train=features_train
y_train=labels_train
X_test=features_test
y_test=labels_test

def SVM():

    from sklearn.svm import LinearSVC
    from sklearn.model_selection import GridSearchCV
    svc = LinearSVC()
    max_iter = [1,10,20,50,100]
    penalty = ['l2']
    C= [0.1,1,10,20]
    param_grid = {'penalty':penalty,'max_iter': max_iter,'C': C}

    clf = GridSearchCV(svc, param_grid, refit=True)

    clf.fit(X_train, y_train)

    best_score = clf.best_score_
    best_param = clf.best_params_
    score = clf.score(X_test, y_test)

    scores = [x[1] for x in clf.grid_scores_]
    print(scores)
    scores = np.array(scores).reshape(len(C), len(max_iter)*len(penalty))
    np.save('drawChart',scores)

    return best_score, best_param, score

print ('SVM: ', SVM())

# [0.8810391303059925, 0.90257412838058, 0.90257412838058, 0.90257412838058, 0.90257412838058, 0.8783435528303564, 0.9171776415178174, 0.917088776326313, 0.91702953286531, 0.91702953286531, 0.8891851061939039, 0.9183032672768743, 0.9182144020853699, 0.9179774282413579, 0.9180366717023608, 0.8776326312983205, 0.9180662934328624, 0.9175627240143369, 0.9183328890073759, 0.9179478065108564]
# ('SVM: ', (0.9183328890073759, {'penalty': 'l2', 'C': 20, 'max_iter': 50}, 0.9204137136958291))