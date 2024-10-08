from django.urls import path

from . import views

urlpatterns = [
    path('', views.Class_Simulation_Page.as_view(), name='Template'),
    path('<int:User_ID>/', views.Function_detail, name = 'userid'),
    path('<int:User_ID>/Set_Col_1/', views.Function_Set_Col_1, name = 'Test'),
    path('Download', views.Function_Download_Simulation, name = 'Template'),
]