#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 20:24:01 2017

@author: harrisonhocker
"""

import os
import tarfile
from six.moves import urllib
import numpy as np

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
    
import pandas as pd

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)
    
housing = load_housing_data()

import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
plt.show()

def split_train_data(data, test_ratio):
    np.random.seed(42)
    shuffled_indicies = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indicies = shuffled_indicies[:test_set_size]
    train_indicies = shuffled_indicies[test_set_size:]
    return data.iloc[train_indicies], data.iloc[test_indicies]
    
train_data, test_data = split_train_data(housing, 0.2)

from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(housing, test_size=0.2, random_state=42)

a = np.ceil(housing["median_income"] / 1.5)
b = np.ceil(housing["median_income"] / 1.5)
difference_locations = np.where(a != b)


housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] <= 5, 5.0, inplace=True)

from sklearn.model_selection import StratifiedShuffleSplit
split.split(housing, housing["income_cat"])
