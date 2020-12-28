#Necessary imports
import streamlit as st
from transformers import pipeline

#Headings for Web Application
st.title("Adam's Text Summarization Application Prototype")
#st.subheader("Adam's Text Summarization Application Prototype")

#min_lengthy = st.slider('Minimum summary length (words)', min_value=10, max_value=120, value=30, step=10)
max_lengthy = st.slider('Maximum summary length (words)', min_value=30, max_value=150, value=60, step=10)
num_beamer = st.slider('Speed vs quality of summary (1 is fastest)', min_value=1, max_value=8, value=4, step=1)

#Textbox for text user is entering
text = st.text_area('Enter Text Below:', height=300) #text is stored in this variable
#uploaded_file = st.file_uploader("Choose a file", type=['txt'])
submit = st.button('Generate')
#if uploaded_file is not None:
#    text2 = uploaded_file
#if text is None:
#    text = text2
if submit:
    st.subheader("Summary")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", tokenizer="facebook/bart-large-cnn",framework="pt")#, device=0)
    summWords = summarizer(text,max_length=max_lengthy, min_length=30, num_beams=num_beamer, do_sample=True, early_stopping=True)
    st.write(summWords[0]["summary_text"])
