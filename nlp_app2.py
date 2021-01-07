#Necessary imports
import streamlit as st
import torch
import re
from transformers import pipeline
import spacy
from spacy import displacy
import en_core_web_sm
sum2 = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", tokenizer="sshleifer/distilbart-cnn-12-6",framework="pt")
#sentiment = pipeline("sentiment-analysis", framework="pt")
#nlp = en_core_web_sm.load()
#device = 0 if torch.cuda.is_available() else -1

#Headings for Web Application
st.title("Adam's Natural Language Processing Application Prototype")
#st.subheader("Adam's Natural Language Processing Application Prototype")
st.subheader("Which natural language processing task would you like to try?")

option = st.selectbox('Please select from the list',('','Text Summarization', 'Sentiment Analysis', 'Named Entity Recognition'))
#def entRecognizer(entDict, typeEnt):
#    entList = [ent for ent in entDict if entDict[ent] == typeEnt]
#    return entList

#min_lengthy = st.slider('Minimum summary length (words)', min_value=10, max_value=120, value=30, step=10)

#Textbox for text user is entering

#if uploaded_file is not None:
#    text2 = uploaded_file
#if text is None:
#    text = text2

#Sentiment Analysis
if option == 'Text Summarization':
    max_lengthy = st.slider('Maximum summary length (words)', min_value=30, max_value=150, value=60, step=10)
    num_beamer = st.slider('Speed vs quality of summary (1 is fastest)', min_value=1, max_value=8, value=4, step=1)
    text = st.text_area('Enter Text Below:', height=300) #text is stored in this variable
    #uploaded_file = st.file_uploader("Choose a file", type=['txt'])
    submit = st.button('Generate')
    if submit:
        st.subheader("Summary")
        st.write("This may take a minute...")
        #@st.cache
        #sum2 = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", tokenizer="sshleifer/distilbart-cnn-12-6",framework="pt")#, device=device)
        summWords = sum2(text,max_length=max_lengthy, min_length=25, num_beams=num_beamer, do_sample=True, early_stopping=True)
        text2 = re.sub(r'\s([?.!"](?:\s|$))', r'\1', summWords[0]["summary_text"])
        st.write(text2)    #Creating graph for sentiment across each sentence in the text inputted

if option == 'Sentiment Analysis':   
    text = st.text_area('Enter Text Below:', height=200) #text is stored in this variable
    #uploaded_file = st.file_uploader("Choose a file", type=['txt'])
    submit = st.button('Generate')
    if submit:
        st.subheader("Sentiment")
        sentiment = pipeline("sentiment-analysis", framework="pt")#, device=0)
        result = sentiment(text)
        sent = result[0]['label']
        cert = result[0]['score']
        st.write(f'Text Sentiment: {sent} | Certainty Score: {(cert*100):.2f}')

#Named Entity Recognition
if option == 'Named Entity Recognition':
    text = st.text_area('Enter Text Below:', height=300) #text is stored in this variable
    #uploaded_file = st.file_uploader("Choose a file", type=['txt'])
    submit = st.button('Generate')
    nlp = en_core_web_sm.load()
    #Getting Entity and type of Entity
    if submit:    
        #entities = []
        #entityLabels = []
        doc = nlp(text)
        #for ent in doc.ents:
        #    entities.append(ent.text)
         #   entityLabels.append(ent.label_)
        #entDict = dict(zip(entities, entityLabels)) #Creating dictionary with entity and entity types

    #Using function to create lists of entities of each type
        #entOrg = entRecognizer(entDict, "ORG")
        #entPerson = entRecognizer(entDict, "PERSON")
        #entDate = entRecognizer(entDict, "DATE")
        #entGPE = entRecognizer(entDict, "GPE")
        options = {"ents": ["ORG", "GPE", "PERSON", "LOC", "DATE"]}
        HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""
        html = displacy.render(doc, style="ent", options=options)
        html = html.replace("\n", " ")
        st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)

    #Displaying entities of each type
        #st.write("People: " + str(entPerson))
        #st.write("Places: " + str(entGPE))
        #st.write("Organizations: " + str(entOrg))
        #st.write("Dates: " + str(entDate))
        #st.write(displacy.render(doc, style='ent'))
