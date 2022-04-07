from django.urls import path
from . import views

app_name = "qa_system"

urlpatterns = [
    #ex: /qa/
    path("",views.form, name="form"), 
    path("algorithm/",views.algorithm, name="algorithm"), 

    # #ex: /polls/
    # path("", views.IndexView.as_view(), name="index"), # views.index
    # #ex: /polls/5/
    # path("<int:pk>/detail/", views.DetailView.as_view(), name="detail"), # <int:question_id> views.detail
    # #ex: /polls/5/results
    # path("<int:pk>/results/", views.ResultView.as_view(), name="results"), # <int:question_id> views.results

    # #ex: /polls/
    # path("", views.index, name="index"), 
    # # ex: /polls/5/
    # path("<int:question_id>/", views.detail, name="detail"), 
    # #ex: /polls/5/results
    # path("<int:question_id>/results/", views.results, name="results"), 
    
    
    # #ex: /polls/5/vote
    # path("<int:question_id>/vote/", views.vote, name="vote"),         
]