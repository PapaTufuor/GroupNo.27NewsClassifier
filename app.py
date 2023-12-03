import streamlit as st
import pandas as pd
import pickle
import joblib
import requests
from bs4 import BeautifulSoup
import requests
from keras.preprocessing.sequence import pad_sequences

# Set page title and favicon
st.set_page_config(page_title="Fake News Detector", page_icon="🔍")

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

tokenizer =joblib.load('C:\\Users\\joelk\\Downloads\\tokenizer_keras.pkl')#put file directory to the pickled model here
model=joblib.load('C:\\Users\\joelk\\Downloads\\model_lstm.pkl')

MAX_SEQUENCE_LENGTH= 300

def predict_fake_news(text):
    tokens=tokenizer.texts_to_sequences([text])
    padded_tokens= pad_sequences(tokens, maxlen=MAX_SEQUENCE_LENGTH)
    prediction=model.predict(padded_tokens)
    return prediction [0][0]

# Add a header and an image to the app
st.image("data:image/png;base64,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", use_column_width=True)
st.title("Fake News Detector")
st.write("This app uses NLP and Machine Learning to predict whether a piece of news is real or fake.")

option =st.radio("Choose input type: ", ("URL", "Upload Document", "Paste Text"))

if option == "URL":
    url_input = st.text_input("Enter the URL: ")
    if st.button("Check"):
        if url_input:
            extracted_text= url_scrape(url_input)
            if extracted_text:
                prediction= predict_fake_news(extracted_text)
                confidence = prediction if prediction >= 0.5 else 1 - prediction
                if prediction >= 0.5:
                    st.success("Prediction: Real News")
                else:
                    st.error("Prediction: Fake News")
                st.write(f'Confidence: {confidence*100:.2f}%')
        else:
            st.warning("Please enter a valid URL.")

elif option == "Upload Document":
    uploaded_file=st.file_uploader("Upload a file", type=['txt', 'pdf', 'docx'])
    if uploaded_file is not None:
        text=uploaded_file.read().decode("utf-8")
        if st.button("Check"):
            if text:
                prediction=predict_fake_news(text)
                confidence = prediction if prediction >= 0.5 else 1 - prediction
                if prediction >= 0.5:
                    st.success("Prediction: Real News")
                else:
                    st.error("Prediction: Fake News")
                st.write(f'Confidence: {confidence*100:.2f}%')
            else:
                st.warning("No content found in the uploaded document")

elif option == "Paste Text":
    entered_text=st.text_area("Paste your text here: ", height=200)
    if st.button("Check"):
        if entered_text:
            prediction= predict_fake_news(entered_text)
            confidence = prediction if prediction >= 0.5 else 1 - prediction
            if prediction >= 0.5:
                st.success("Prediction: Real News")
            else:
                st.error("Prediction: Fake News")
            st.write(f'Confidence: {confidence*100:.2f}%')
        else:
            st.warning("Please enter some text.")

