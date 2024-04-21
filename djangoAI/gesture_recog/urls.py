from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("get_prediction/", views.get_prediction, name="get_prediction"),
]