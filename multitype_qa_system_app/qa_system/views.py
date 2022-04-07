from django.shortcuts import render, get_object_or_404
from django.http import HttpResponse, HttpResponseRedirect
# from .models import Question, Choice
from django.urls import reverse
# from django.views import generic
# from django.utils import timezone

from transformers import BertTokenizer, BertForQuestionAnswering
#from transformers import AutoTokenizer, AutoModel
# from transformers import *
from transformers import pipeline
import scispacy
import spacy
from spacy import displacy

def qa_system(information, question):
    #It works
    modelname = 'deepset/bert-base-cased-squad2'
    tokenizer = BertTokenizer.from_pretrained(modelname)
    model = BertForQuestionAnswering.from_pretrained(modelname)
    qa = pipeline('question-answering', model=model, tokenizer=tokenizer)    

    nlp1 = spacy.load("en_core_web_sm")
    nlp2 = spacy.load("en_ner_bc5cdr_md")
    med7 = spacy.load("en_core_med7_lg")  

    doc1 = nlp1(information)
    doc2 = nlp2(information)
    doc3 = med7(information)
    
    ans = qa({'question': question, 'context': information}) #ans['answer']

    docs = [doc1, doc2, doc3]
    answers=[]
    for d in docs:
        for entity in d.ents:
            if entity.text == ans['answer']:
                answers.append(entity)
                # print(entity.label_)
                
    answers_filter = list(set(answers))
    if len(answers_filter) > 1:
        answer_unique = answers_filter[0]
    elif len(answers_filter) == 0:
        answer_unique = 'None'
    elif len(answers_filter) == 1:
        answer_unique = answers_filter

    text = answer_unique.text
    label = answer_unique.label_

    # create distinct colours for labels
    # col_dict = {}
    # seven_colours = ['#e6194B', '#3cb44b', '#ffe119', '#ffd8b1', '#f58231', '#f032e6', '#42d4f4']
    # for label, colour in zip(med7.pipe_labels['ner'], seven_colours):
    #     col_dict[label] = colour
    # options = {'ents': med7.pipe_labels['ner'], 'colors':col_dict}  
    # displacy.render(doc1, style='ent')
    # displacy.render(doc2, style='ent')
    # spacy.displacy.render(doc3, style='ent', jupyter=True, options=options)
    return {
        'full_answer': ans,
        'answer': text+', '+label,
        'docs': docs
    }
        

def form(request):
    return render(request, "qa_system/form.html")

def algorithm(request):
    try:
        information = request.POST["textfield"]
        question = request.POST["questionfield"] 
    except(KeyError):
        return render(request, "qa_system/form.html", {"error_message": "It doesn't work"})
    else:
        if not information and not question:
            return render(request, "qa_system/form.html", {"error_message": "Fill medical record and question"})
        elif not question:
            return render(request, "qa_system/form.html", {"error_message": "Fill question",'information_cache':information})
        elif not information: 
            return render(request, "qa_system/form.html", {"error_message": "Fill medical record",'question_cache':question})
        else:
            #algoritmo
            full_information = qa_system(information, question) #Mitigar posibles errores
            
            if full_information:
                answer = full_information['answer']
                if answer:
                    full_answer = full_information['full_answer']
                    docs = full_information['docs']     
                    return render(request, "qa_system/form.html", {"answer": answer, 'information_cache':information, 'question_cache':question}) 
                else:
                    return render(request, "qa_system/form.html", {"error_message": "It has not answer"})                       
            else:
                return render(request, "qa_system/form.html", {"error_message": "It has not answer"}) 

