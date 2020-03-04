
"""
Created on Wed Mar  4 07:58:20 2020

@author: je28rodr
"""
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles, load_digits, load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

comp = np.loadtxt('Z:\\CSIS320\\project1\\training.csv',delimiter=',',skiprows=1)
X = comp[:,0:7]
y = comp[:,7]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
