from django.urls import path
from . import views

urlpatterns = [
    # control center
    path('', views.index, name='index'),
    # collecting
    path('collect_new_cards', views.collect, name='collect new cards'),
]