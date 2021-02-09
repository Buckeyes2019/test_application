import streamlit as st
import torch
#import tensorflow
from transformers import pipeline
import spacy
from spacy import displacy
#import en_core_web_sm
st.set_page_config(page_title="NLP Prototype")

st.title("Natural Language Processing Prototype")
st.write("_This web application is intended for educational use, please do not upload any classified, proprietary, or sensitive information._")
st.subheader("__Which natural language processing task would you like to try?__")
st.write("- __Sentiment Analysis:__ Identifying whether a piece of text has a positive or negative sentiment.")
st.write("- __Named Entity Recognition:__ Identifying all geopolitical entities, organizations, people, locations, or dates in a body of text.")
st.write("- __Text Summarization:__ Condensing larger bodies of text into smaller bodies of text.")

option = st.selectbox('Please select from the list',('','Sentiment Analysis','Named Entity Recognition','Text Summarization'))

#@st.cache(allow_output_mutation=True, show_spinner=False)
#def Loading_Model_1():
#   sum2 = pipeline("summarization", model="facebook/bart-large-cnn", tokenizer="facebook/bart-large-cnn",framework="pt")
#    return sum2

#@st.cache(allow_output_mutation=True, show_spinner=False)
#def Loading_Model_2():
#    sentiment = pipeline("sentiment-analysis", framework="pt")
#    return sentiment

#@st.cache(allow_output_mutation=True, show_spinner=False)
#def Loading_Model_3():
#    nlp = spacy.load('en_core_web_sm')
#    return nlp

#@st.cache(allow_output_mutation=True)
def entRecognizer(entDict, typeEnt):
    entList = [ent for ent in entDict if entDict[ent] == typeEnt]
    return entList

#with st.spinner(text="Please wait for the three models to load. This should take approximately 60 seconds."):
#    sum2 = Loading_Model_1()
#    sentiment = Loading_Model_2()
#    nlp = Loading_Model_3()

if option == 'Text Summarization':
    max_lengthy = st.slider('Maximum summary length (words)', min_value=30, max_value=150, value=60, step=10)
    num_beamer = st.slider('Speed vs quality of summary (1 is fastest)', min_value=1, max_value=8, value=4, step=1)
    text = st.text_area('Enter Text Below (maximum 900 words):', height=300) 
    #uploaded_file = st.file_uploader("Choose a file", type=['txt'])
    submit = st.button('Generate')  
    if submit:
        st.subheader("Summary:")
        st.write("This may take a moment...")
        sum2 = pipeline("summarization", model="sshleifer/distill-pegasus-xsum-12-12", tokenizer="sshleifer/distill-pegasus-xsum-12-12",framework="pt")
        summWords = sum2(text, max_length=max_lengthy, min_length=25, num_beams=num_beamer, do_sample=True, early_stopping=False, repetition_penalty=1.2, length_penalty=1.2)
        text2 =summWords[0]["summary_text"] #re.sub(r'\s([?.!"](?:\s|$))', r'\1', )
        st.write(text2)

if option == 'Sentiment Analysis':
    text = st.text_area('Enter Text Below:', height=200)
    #uploaded_file = st.file_uploader("Choose a file", type=['txt'])
    submit = st.button('Generate')
    if submit:
        st.subheader("Sentiment:")
        sentiment = pipeline("sentiment-analysis", framework="pt")
        result = sentiment(text)
        sent = result[0]['label']
        cert = result[0]['score']
        st.write(f'Text Sentiment: {sent} | Certainty Score: {(cert*100):.2f}')

if option == 'Named Entity Recognition':
    text = st.text_area('Enter Text Below:', height=300)
    #uploaded_file = st.file_uploader("Choose a file", type=['txt'])
    submit = st.button('Generate')
    if submit:    
        entities = []
        entityLabels = []
        nlp = spacy.load('en_core_web_sm')
        doc = nlp(text)
        for ent in doc.ents:
            entities.append(ent.text)
            entityLabels.append(ent.label_)
        entDict = dict(zip(entities, entityLabels)) 
        entOrg = entRecognizer(entDict, "ORG")
        entPerson = entRecognizer(entDict, "PERSON")
        entDate = entRecognizer(entDict, "DATE")
        entGPE = entRecognizer(entDict, "GPE")
        entLoc = entRecognizer(entDict, "LOC")
        options = {"ents": ["ORG", "GPE", "PERSON", "LOC", "DATE"]}
        HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""
        
        st.subheader("List of Named Entities:")
        st.write("Geopolitical Entities (GPE): " + str(entGPE))
        st.write("People (PERSON): " + str(entPerson))
        st.write("Organizations (ORG): " + str(entOrg))
        st.write("Dates (DATE): " + str(entDate))
        st.write("Locations (LOC): " + str(entLoc))
        st.subheader("Original Text with Entities Highlighted")
        html = displacy.render(doc, style="ent", options=options)
        html = html.replace("\n", " ")
        st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)
