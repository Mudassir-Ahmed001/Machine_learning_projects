import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import pickle


#Title
st.title('Spam Mail Detector')

#loading Model
model = pickle.load(open('/content/drive/MyDrive/Colab Notebooks/Spam_mail_detection/Spam_mail_detection.sav', 'rb'))
vectorizer = pickle.load(open('/content/drive/MyDrive/Colab Notebooks/Spam_mail_detection/vectorizer.sav', 'rb'))

#Input
Mail = st.text_input(
    'Enter the Mail:- ',
    ""
)

def prediction(Mail):
  #feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
  #x_train_features = feature_extraction.fit_transform(x_train) 
  input_data_extraction = vectorizer.transform([Mail])
  prediction = model.predict(input_data_extraction)
  if prediction[0] == 1:
    return "Ham Mail"
  else:
    return "Spam Mail"

  
if st.button('Predict'):
  result = prediction(Mail)
  st.success(result)
