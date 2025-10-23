from django.urls import path

from .views import (AdminMemoryStatsView, AdminReloadLLMView, ChatView,
                    ClearCacheView, RatingMetricsView, RatingView,
                    StreamingChatView)

urlpatterns = [
    path("chat/", ChatView.as_view(), name="api-chat"),
    path("chat/stream/", StreamingChatView.as_view(), name="api-chat-stream"),
    path("rating/", RatingView.as_view(), name="api-rating"),
    path("rating/metrics/", RatingMetricsView.as_view(), name="api-rating-metrics"),
    path("admin/reload-llm/", AdminReloadLLMView.as_view(), name="api-admin-reload-llm"),
    path("admin/memory-stats/", AdminMemoryStatsView.as_view(), name="api-admin-memory-stats"),
    path("admin/clear-cache/", ClearCacheView.as_view(), name="api-admin-clear-cache"),
]