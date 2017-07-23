#!/usr/bin/env python

from sklearn.neighbors import KNeighborsClassifier
import requests
import json

def extractData(data):
    return [data['sepalLength'], data['sepalWidth'], data['petalLength'], data['petalWidth']]

data = json.loads(requests.get("http://harrisonhocker.com/api/data/iris").text)

trainingData = list(map(extractData, data))

target = list(map((lambda x: x['species']), data))

# Create and fit a nearest-neighbor classifier
knn = KNeighborsClassifier()
knn.fit(trainingData, target) 

sample = [6.1, 2.9 , 4.7, 1.4]
predict = []
predict.append(sample)

print knn.predict(predict)