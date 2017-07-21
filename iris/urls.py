from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'data', views.get_iris_data, name='get_iris_data'),
    url(r'object', views.get_object, name='get_object')
]
