from django.contrib import admin
from django.urls import include, path
from rest_framework import permissions

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('chatbot.urls')),  # Your app's API
]
