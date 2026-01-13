from django.urls import path
from . import views

app_name = 'classifier'

urlpatterns = [
    path('', views.index, name='index'),
    path('predict/', views.predict, name='predict'),
]
