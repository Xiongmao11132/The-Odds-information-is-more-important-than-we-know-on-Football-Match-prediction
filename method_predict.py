# -*- coding:utf-8 -*-
# coding:unicode_escape
from sklearn.ensemble import RandomForestClassifier
from IPython.display import Image
from sklearn import tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
import numpy as np
import warnings
import random
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# DecisionTree
'''
datasets = pd.read_csv( 'data.csv', encoding='gbk',engine='python', error_bad_lines=False)
Train_X = datasets.iloc[0:47266, 2:22].values # 2-16 means Except Odds, 17-22 means Only Odds, 2-22 means All Data
Train_y = datasets.iloc[0:47266, 23:24].values # 23-24 means results
Train_X, X_Test, Train_y, Y_Test = train_test_split(Train_X, Train_y, test_size=0.4, random_state=1)
clf_tree=DecisionTreeClassifier(criterion='gini', max_depth=6)
clf_tree.fit(Train_X, Train_y)
y_predict = clf_tree.predict(X_Test)
print('Tree Acc: ',clf_tree.score(X_Test, Y_Test))
print('Tree Recall: ',recall_score(Y_Test, y_predict, average='macro'))
print('F1 score',f1_score(Y_Test, y_predict, average='macro'))
'''
# GDBT
'''
datasets = pd.read_csv('data.csv', encoding='gbk',engine='python', error_bad_lines=False)
Train_X = datasets.iloc[0:47266, 17:22].values # 2-16 means Except Odds, 17-22 means Only Odds, 2-22 means All Data
Train_y = datasets.iloc[0:47266, 23:24].values # 23-24 means results
Train_X, X_Test, Train_y, Y_Test = train_test_split(Train_X, Train_y, test_size=0.4, random_state=1)
clf_1 = GradientBoostingClassifier(n_estimators=200, max_depth=6, random_state=0)
clf_1.fit(Train_X, Train_y)
y_predict = clf_1.predict(X_Test)
print('GDBT Acc: ', precision_score(Y_Test, y_predict, average='macro'))
print('GDBT Recall: ', recall_score(Y_Test, y_predict, average='macro'))
print('F1 Score: ', f1_score(Y_Test, y_predict, average='macro'))
'''
# Xgboost
'''
datasets = pd.read_csv('data.csv', encoding='gbk',engine='python', error_bad_lines=False)
Train_X = datasets.iloc[0:47266, 2:17].values # 2-16 means Except Odds, 17-22 means Only Odds, 2-22 means All Data
Train_y = datasets.iloc[0:47266, 23:24].values # 23-24 means results
Train_X, X_Test, Train_y, Y_Test = train_test_split(Train_X, Train_y, test_size=0.4, random_state=1)
XgBoost = xgb.XGBClassifier(n_estimators=10, max_depth=15, random_state=0)
XgBoost.fit(Train_X, Train_y)
y_predict = XgBoost.predict(X_Test)
print('XgBoost Acc: ', precision_score(Y_Test, y_predict, average='macro'))
print('XgBoost Recall: ', recall_score(Y_Test, y_predict, average='macro'))
print('F1 Score: ', f1_score(Y_Test, y_predict, average='macro'))
'''
# KNN
'''
datasets = pd.read_csv('data.csv', encoding='gbk',engine='python', error_bad_lines=False)
X_Train = datasets.iloc[0:47263, 2:22].values # 2-16 means Except Odds, 17-22 means Only Odds, 2-22 means All Data
Y_Train = datasets.iloc[0:47263, 23:24].values # 23-24 means results
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X_Train, Y_Train, test_size=0.4, random_state=1)
from sklearn.neighbors import KNeighborsClassifier as KNN
knc = KNN(n_neighbors=15)
knc.fit(X_Train, Y_Train)
y_predict = knc.predict(X_Test)
print('KNN Acc: ', knc.score(X_Test, Y_Test))
print('KNN Recall: ', recall_score(Y_Test, y_predict, average='macro'))
print('F1 Score: ', f1_score(Y_Test, y_predict, average='macro'))
'''
# RF
'''
datasets = pd.read_csv('data.csv', encoding='gbk',engine='python', error_bad_lines=False)
X_Train = datasets.iloc[0:47263, 17:22].values # 2-16 means Except Odds, 17-22 means Only Odds, 2-22 means All Data
Y_Train = datasets.iloc[0:47263, 23:24].values # 23-24 means results
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X_Train, Y_Train, test_size=0.4, random_state=1)
clf_tree_1 = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=0, class_weight=None)
clf_tree_1.fit(X_Train, Y_Train)
test_predictions = clf_tree_1.predict(X_Test)
print('RF Acc: ',clf_tree_1.score(X_Test, Y_Test))
print('RF Recall: ',recall_score(Y_Test, test_predictions, average='macro'))
print('F1 Score: ',f1_score(Y_Test, test_predictions, average='macro'))
'''
# LogisticRegression
'''
model = LogisticRegression()
datasets = pd.read_csv('data.csv', encoding='gbk',engine='python', error_bad_lines=False)
Train_X = datasets.iloc[0:47263, 17:22].values # 2-16 means Except Odds, 17-22 means Only Odds, 2-22 means All Data
Train_y = datasets.iloc[0:47263, 23:24].values # 23-24 means results
model.fit(Train_X, Train_y)
X_Train, X_Test, Y_Train, Y_Test = train_test_split(Train_X, Train_y, test_size=0.4, random_state=1)
y_predict = model.predict(X_Test)
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False)
print('LR Acc: ',model.score(X_Test, Y_Test))
print('LR Recall: ',recall_score(Y_Test, y_predict, average='macro'))
print('F1 Score: ',f1_score(Y_Test, y_predict, average='macro'))
'''