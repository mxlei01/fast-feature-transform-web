"""The site's URLs: everything lives in the fare app."""

from django.urls import include, path

urlpatterns = [path("", include("fare.urls"))]
