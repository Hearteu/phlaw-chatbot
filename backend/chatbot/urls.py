from django.urls import path

from .views import (AdminMemoryStatsView, AdminReloadLLMView, ChatView,
                    ClearCacheView)

urlpatterns = [
    path("chat/", ChatView.as_view(), name="api-chat"),
    path("admin/reload-llm/", AdminReloadLLMView.as_view(), name="api-admin-reload-llm"),
    path("admin/memory-stats/", AdminMemoryStatsView.as_view(), name="api-admin-memory-stats"),
    path("admin/clear-cache/", ClearCacheView.as_view(), name="api-admin-clear-cache"),
]