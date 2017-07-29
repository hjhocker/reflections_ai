#!/usr/bin/env python

from sklearn.neighbors import KNeighborsClassifier
import requests
import json

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.ensemble import IsolationForest

def extractData(data):
    return [data['sepalLength'], data['sepalWidth'], data['petalLength'], data['petalWidth']]

data = json.loads(requests.get("http://harrisonhocker.com/api/data/iris").text)

trainingData = list(map(extractData, data))

target = list(map((lambda x: x['species']), data))

# Binarize the output
y = label_binarize(target, classes=['SETOSA', 'VERSICOLOR', 'VIRGINICA'])
n_classes = y.shape[1]

trainingData = np.c_[trainingData]
X_train, X_test, y_train, y_test = train_test_split(trainingData, y, test_size=.5,
                                                    random_state=0)

random_state = np.random.RandomState(9)
classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                 random_state=random_state))

#classifier = OneVsRestClassifier(IsolationForest())

y_score = classifier.fit(X_train, y_train).decision_function(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

plt.figure()
lw = 2
plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()