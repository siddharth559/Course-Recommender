from django.urls import path
from . import views

urlpatterns = [
    path('', views.run_page, name='model')
]