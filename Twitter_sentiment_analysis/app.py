import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import pickle


#Title
st.title('Twitter Sentiment Analysis')

#loading Model
model = pickle.load(open('/content/drive/MyDrive/Colab Notebooks/Twitter_sentiment_analysis/trained_model.sav', 'rb'))
vectorizor = pickle.load(open('/content/drive/MyDrive/Colab Notebooks/Twitter_sentiment_analysis/Vectorizor.sav', 'rb'))

#Input
tweet = st.text_input(
    'Enter the Tweet:- ',
    ""
)

def prediction(tweet):
  vectorised_tweet = vectorizor.transform([tweet])
  prediction = model.predict(vectorised_tweet)

  if prediction[0] == 1:
    return "Positive Tweet"
  else:
    return "Negative Tweet"


if st.button('Predict'):
  result = prediction(tweet)
  if result == "Positive Tweet":
    st.success("Positive Tweet")
  else:
    st.error("Negative Tweet")
