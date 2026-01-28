#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 12:30:14 2020

@author: ghostman
"""
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

df = pd.read_csv('zoo.csv')

from sklearn.model_selection import train_test_split

all_x = df.iloc[:,1:17]
all_y = df.iloc[:,17]

x_train, x_test, y_train, y_test = train_test_split(all_x,all_y)

clf = svm.SVC()
clf.fit(x_train,y_train)
testing = clf.predict(x_test[10:15])

# model = KNeighborsClassifier(n_neighbors=3)

# model.fit(x_train,y_train)
# score_ = model.score(x_test,y_test)

#predicted_ = model.predict(x_test[10:15])

check_y_test = y_test[10:15]








