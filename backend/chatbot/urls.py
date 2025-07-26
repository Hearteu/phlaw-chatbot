from django.urls import path

from .views import ChatView

urlpatterns = [
    path("api/chat/", ChatView.as_view(), name="api-chat"),
]
