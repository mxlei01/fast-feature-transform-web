"""The fare app's URLs."""

from django.urls import path

from fare import views

app_name = "fare"

urlpatterns = [
    path("", views.index, name="index"),
    path("api/predict", views.api_predict, name="api_predict"),
    path("api/pipeline", views.api_pipeline, name="api_pipeline"),
    path("healthz", views.healthz, name="healthz"),
]
