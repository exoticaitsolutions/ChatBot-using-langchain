# chatapp/urls.py
from django.urls import path
from .views import chat_view, upload_pdf


urlpatterns = [
    path('', upload_pdf, name='upload_pdf'),
    path('chat/', chat_view, name='chat'),
]

