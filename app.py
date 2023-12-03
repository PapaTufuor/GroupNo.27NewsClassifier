import streamlit as st
import pandas as pd
import pickle
import joblib
import requests
from bs4 import BeautifulSoup
import requests
from keras.preprocessing.sequence import pad_sequences





def url_scrape(url):
    try:
        response=requests.get(url)
        soup=BeautifulSoup(response.text, 'html.parser')

        text_content = ''
        for paragraph in soup.find_all('p'):
            text_content += paragraph.get_text() + ''
        
        return text_content.strip()
    
    except Exception as e:
        st.error("Error occurred while scraping the URL.")
        return None
    
    



model =joblib.load('C:\\Users\\hp\\Desktop\\Year 3 Sem 2\\Introduction to AI\\Final Project\\model_lstm.pkl')#put file directory to the pickled model here

tokenizer=joblib.load('C:\\Users\\hp\\Desktop\\Year 3 Sem 2\\Introduction to AI\\Final Project\\tokenizer_keras.pkl')


MAX_SEQUENCE_LENGTH= max([len(sequence) for sequence in tokenizer.texts_to_sequences(texts)])



def predict_fake_news(text):
    tokens=tokenizer.texts_to_sequences([text])
    padded_tokens= pad_sequences(tokens, maxlen=MAX_SEQUENCE_LENGTH)
    prediction=model.predict(padded_tokens)
    return prediction [0][0]
    

st.title("Fake News Detector")

option =st.radio("Choose input type: ", ("URL", "Upload Document", "Paste Text"))

if option == "URL":
    url_input = st.text_input("Enter the URL: ")


    if st.button("Check"):
        if url_input:
            extracted_text= url_scrape(url_input)
            if extracted_text:
                prediction= predict_fake_news(extracted_text)
                
                if prediction >= 0.5:
                    st.write("Prediction: Real News")
                else:
                    st.write("Prediction: Fake News")

    else:
        st.warning("Please enter a valid URL.")



elif option == "Upload Document":
    uploaded_file=st.file_uploader("Upload a file", type=['txt', 'pdf', 'docx'])

    if uploaded_file is not None:
        text=uploaded_file.read().decode("utf-8")

        if st.button("Check"):
            if text:
                prediction=predict_fake_news(text)
                if prediction >= 0.5:
                    st.write("Prediction: Real News")
                else:
                    st.write("Prediction: Fake News")
            else:
                st.warning("No content found in the uploaded document")


elif option == "Paste Text":
    entered_text=st.text_area("Paste your text here: ", height=200)

    if st.button("Check"):
        if entered_text:
            prediction= predict_fake_news(entered_text)
            if prediction >= 0.5:
                    st.write("Prediction: Real News")
            else:
                    st.write("Prediction: Fake News")
        else:
            st.warning("Please enter some text.")


        





