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

def answer_unique_function(docs, ans):
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
    
    return answer_unique

def render_text(docs, models):  
    # create distinct colours for labels
    col_dict = {}
    seven_colours = ['#e6194B', '#3cb44b', '#ffe119', '#ffd8b1', '#f58231', '#f032e6', '#42d4f4']
    for label, colour in zip(models[2].pipe_labels['ner'], seven_colours):
        col_dict[label] = colour
    options = {'ents': models[2].pipe_labels['ner'], 'colors':col_dict}  

    html1 = displacy.render(docs[0], style='ent', page=True)
    html2 = displacy.render(docs[1], style='ent', page=True)
    html3 = displacy.render(docs[2], style='ent', options=options, page=True)
    html = [html1, html2, html3]

    return html

def qa_system(information, question):
    #It works
    modelname = 'deepset/bert-base-cased-squad2'
    tokenizer = BertTokenizer.from_pretrained(modelname)
    model = BertForQuestionAnswering.from_pretrained(modelname)
    qa = pipeline('question-answering', model=model, tokenizer=tokenizer)    

    nlp1_title = "en_core_web_sm"
    nlp2_title = "en_ner_bc5cdr_md"
    med7_title = "en_core_med7_lg"
    models_names = [nlp1_title, nlp2_title, med7_title]

    nlp1 = spacy.load(nlp1_title)
    nlp2 = spacy.load(nlp2_title)
    med7 = spacy.load(med7_title) 
    models = [nlp1, nlp2, med7]

    doc1 = nlp1(information)
    doc2 = nlp2(information)
    doc3 = med7(information)
    docs = [doc1, doc2, doc3]
    
    entities1 = []
    if doc1:
        for en in doc1.ents:
            entity = en.text+', '+en.label_
            entities1.append(entity)
    
    entities2 = []
    if doc2:
        for en in doc2.ents:
            entity = en.text+', '+en.label_
            entities2.append(entity)

    entities3 = []
    if doc3:
        for en in doc3.ents:
            entity = en.text+', '+en.label_
            entities3.append(entity)  
    
    entities = [entities1, entities2, entities3]              

    ans = qa({'question': question, 'context': information}) #ans['answer']

    answer_unique = answer_unique_function(docs, ans)

    text = answer_unique.text
    label = answer_unique.label_

    html = render_text(docs, models)
    
    file = open("./qa_system/templates/qa_system/displacy1.html","w")
    file.write(html[0])
    file.close()

    file = open("./qa_system/templates/qa_system/displacy2.html","w")
    file.write(html[1])
    file.close()

    file = open("./qa_system/templates/qa_system/displacy3.html","w")
    file.write(html[2])
    file.close()  

    return {
        'full_answer': ans,
        'answer': text+' ['+label+']',
        'entities': entities,
        'models_names': models_names,
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
            #algorithm
            full_information = qa_system(information, question) #Mitigar posibles errores
            
            if full_information:
                answer = full_information['answer']
                if answer:
                    full_answer = full_information['full_answer']
                    entities1 = full_information['entities'][0] 
                    entities2 = full_information['entities'][1] 
                    entities3 = full_information['entities'][2] 
                    model1_name = full_information['models_names'][0]  
                    model2_name = full_information['models_names'][1]  
                    model3_name = full_information['models_names'][2]   
                    return render(request, "qa_system/form.html", {
                        "answer": answer, 
                        'information_cache':information, 
                        'question_cache':question, 
                        'entities1': entities1,
                        'entities2': entities2,
                        'entities3': entities3,
                        'model1_name':model1_name,
                        'model2_name':model2_name,
                        'model3_name':model3_name,
                        'score': full_answer['score'],
                        'start_point': full_answer['start'],
                        'end_point': full_answer['end'],
                        'contextp1':information[:full_answer['start']],
                        'contextp2':information[full_answer['end']:],
                        }) 
                else:
                    return render(request, "qa_system/form.html", {"error_message": "It has not answer"})                       
            else:
                return render(request, "qa_system/form.html", {"error_message": "It has not answer"}) 

def displacy1_view(request):
    return render(request, "qa_system/displacy1.html") 
def displacy2_view(request):
    return render(request, "qa_system/displacy2.html")     
def displacy3_view(request):
    return render(request, "qa_system/displacy3.html")     
    