# -*- coding: utf-8 -*-

import vectorize_data as vd
import settings
import pickle as pickle
from preProcessData import FeatureExtraction

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

def logictic():

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    lr = LogisticRegression()

    param_grid = {'penalty': ['l1', 'l2'], 'max_iter': [20, 50, 70, 100, 200], 'C': [0.1, 10, 20, 30]}
  
    clf = GridSearchCV(lr, param_grid, refit=True, cv=5)
    clf.fit(X_train, y_train)
   
   
    best_score = clf.best_score_
    best_param = clf.best_params_
    score = clf.score(X_test, y_test)

    C = [0.1, 10, 20, 30]
    max_iter = [20, 50, 70, 100, 200]
    penalty = ['l1', 'l2']

    scores = [x[1] for x in clf.grid_scores_]
    scores = np.array(scores).reshape(len(C), len(max_iter)*len(penalty))
    np.save('drawChart',scores)


    return best_score, best_param, score

print ('logictic: ', logictic())
# tfidf: logictic:  (0.67009803921568623, {'penalty': 'l2', 'C': 10, 'max_iter': 100}, 0.69411764705882351)
# Bow: logictic:  (0.66004901960784312, {'penalty': 'l2', 'C': 10, 'max_iter': 100}, 0.68627450980392157)

# logictic:  (0.68112745098039218, {'penalty': 'l2', 'C': 30, 'max_iter': 20}, 0.69019607843137254)


# Nearest Neighbors
def nearest_Neighbors():

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import GridSearchCV
    knn = KNeighborsClassifier()

    param_grid = {'n_neighbors': [3, 5, 7, 9]}
    clf = GridSearchCV(knn, param_grid, refit=True)

    clf.fit(X_train, y_train)
    best_score = clf.best_score_
    best_param = clf.best_params_
    score = clf.score(X_test, y_test)

    return best_score, best_param, score

# # print 'nearest_Neighbors: ', nearest_Neighbors()
# # tfidf: nearest_Neighbors:  (0.61078431372549025, {'n_neighbors': 9}, 0.61078431372549025)
# # Bow: nearest_Neighbors:  (0.47647058823529409, {'n_neighbors': 3}, 0.51372549019607838)


# # Decision Tree
# def decision_Tree():

#     from sklearn.tree import DecisionTreeClassifier
#     from sklearn.model_selection import GridSearchCV
#     dt = DecisionTreeClassifier()

#     param_grid = {'max_depth': [5, 10, 20, 30]}
#     clf = GridSearchCV(dt, param_grid, refit=True)

#     clf.fit(X_train, y_train)
#     best_score = clf.best_score_
#     best_param = clf.best_params_
#     score = clf.score(X_test, y_test)

#     return best_score, best_param, score

# # print 'decision_Tree: ', decision_Tree()
# # tfidf: decision_Tree:  (0.51593137254901966, {'max_depth': 30}, 0.52549019607843139)
# # Bow: decision_Tree:  (0.52941176470588236, {'max_depth': 20}, 0.53921568627450978)


# # scikit_learn: SVM
# def SVM():

#     from sklearn.svm import SVC
#     from sklearn.model_selection import GridSearchCV
#     svc = SVC()

#     param_grid = [
#         {'C': [0.1, 1, 10], 'kernel': ['linear']},
#         {'C': [0.1, 1, 10], 'degree': [2, 3, 4], 'kernel': ['poly']},
#         {'C': [0.1, 1, 10], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
#     ]
#     clf = GridSearchCV(svc, param_grid, refit=True)

#     clf.fit(X_train, y_train)
#     best_score = clf.best_score_
#     best_param = clf.best_params_
#     score = clf.score(X_test, y_test)

#     return best_score, best_param, score

# # print 'SVM: ', SVM()
# # tfidf: linear SVM:  (0.66249999999999998, {'kernel': 'linear', 'C': 10}, 0.68823529411764706)
# # tfidf: poly SVM: (0.6169117647058824, {'kernel': 'poly', 'C': 0.1, 'degree': 4}, 0.6441176470588236)
# # tfidf: rbf SVM:  (0.4257, {'kernel: 'rbf', 'C': 0.1, 'gama': 0.001}, 0.41


# # Naive Bayes
# def naive_Bayes():

#     from sklearn.naive_bayes import GaussianNB
#     clf = GaussianNB()

#     clf.fit(X_train, y_train)

#     score = clf.score(X_test, y_test)

#     return score

# # print 'naive_Bayes: ', naive_Bayes()
# # tfidf: naive_Bayes:  0.530392156863
# # Bow: naive_Bayes:  0.547058823529


# # Random Forest
# def random_Forest():

#     from sklearn.ensemble import RandomForestClassifier
#     from sklearn.model_selection import GridSearchCV
#     rfc = RandomForestClassifier()

#     param_grid = {'max_depth': [5, 10, 35], 'n_estimators': [5, 10, 20], 'max_features': [10, 20]}
#     clf = GridSearchCV(rfc, param_grid, refit=True)

#     clf.fit(X_train, y_train)
#     best_score = clf.best_score_
#     best_param = clf.best_params_
#     score = clf.score(X_test, y_test)

#     return best_score, best_param, score

# # print 'random_Forest: ', random_Forest()
# # tfidf: random_Forest:  (0.45196078431372549, {'max_features': 20, 'n_estimators': 20, 'max_depth': 35}, 0.4696078431372549)
# # Bow: random_Forest:  (0.44485294117647056, {'max_features': 20, 'n_estimators': 20, 'max_depth': 35}, 0.44509803921568625)


# # Neural Net
# def neural_Net():

#     from sklearn.neural_network import MLPClassifier
#     from sklearn.model_selection import GridSearchCV
#     mlp = MLPClassifier()

#     param_grid = {'alpha': [1, 0.1, 0.01], 'hidden_layer_sizes': [(20,), (100,), (100, 20)]}
#     clf = GridSearchCV(mlp, param_grid, refit=True)

#     clf.fit(X_train, y_train)
#     best_score = clf.best_score_
#     best_param = clf.best_params_
#     score = clf.score(X_test, y_test)

#     return best_score, best_param, score
# # print 'neural_Net: ',neural_Net() #69% #layer: default
# # tfidf: neural_Net:  (0.6693627450980392, {'alpha': 0.01, 'hidden_layer_sizes': (22,)}, 0.6872549019607843)
# # tfidf: neural_Net:  (0.6686274509803921, {'alpha': 0.1, 'hidden_layer_sizes': (100, 20)}, 0.6882352941176471)