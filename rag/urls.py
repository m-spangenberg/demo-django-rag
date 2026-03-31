from django.urls import path

from .views import ChatView, DocumentDetailView, HomeView, document_health

app_name = "rag"

urlpatterns = [
    path("", HomeView.as_view(), name="home"),
    path("documents/<int:pk>/", DocumentDetailView.as_view(), name="document-detail"),
    path("chat/", ChatView.as_view(), name="chat"),
    path("health/", document_health, name="health"),
]
