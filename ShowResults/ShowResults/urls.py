"""ShowResults URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.conf.urls import url
from . import settings
from django.conf.urls.static import static

from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('hello/', views.hello),
    path('runnoob/', views.runnoob),
    path('section3_list/', views.section3_example),
    path('section4_list/', views.section4_example),
    path('section3_predict/', views.section3_predict),
    url(r'^search-form/$', views.search_form),
    url(r'^search/$', views.search),
    url(r'^search-post/$', views.search_post),
    url(r'^section3/$', views.section3_view),
    url(r'^section4/$', views.section4_view),
    url(r'^receive_upload/$', views.receive_view),
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)