from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('train_model', views.train_model, name='train'),
    path('time_dataset', views.time_function, name='time')
]
