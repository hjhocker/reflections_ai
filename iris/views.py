# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render
from django.http import HttpResponse

import json
import requests

def get_iris_data(request):
    iris_data = requests.get("http://harrisonhocker.com/api/data/iris").text;
    return HttpResponse(iris_data, content_type="application/json")

def get_object(request):
    data = {}
    data['purpose'] = 'ai'
    return HttpResponse(json.dumps(data), content_type="application/json")
