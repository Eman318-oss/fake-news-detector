# app.py
import streamlit as st
import joblib
import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Downloads (first time only)
nltk.download('stopwords')
nltk.download('wordnet')

# Load models and vectorizer
nb_model = joblib.load("nb_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Preprocessing tools
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [stemmer.stem(lemmatizer.lemmatize(w)) for w in words]
    return " ".join(words)

# Streamlit UI
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detection Web App")
st.write("Enter a news article below to predict whether it's **Fake** or **Real**.")

# Input field
news_input = st.text_area("Paste your news article here:", height=250)

# Model selection
model_choice = st.radio("Choose a model to test with:", ["Naive Bayes", "Random Forest"])

if st.button("Predict"):
    if news_input.strip() == "":
        st.warning("Please enter some text before predicting.")
    else:
        cleaned = clean_text(news_input)
        vectorized = vectorizer.transform([cleaned]).toarray()

        if model_choice == "Naive Bayes":
            prediction = nb_model.predict(vectorized)[0]
        else:
            rf_model = joblib.load("rf_model.pkl")
            prediction = rf_model.predict(vectorized)[0]

        result = "🟢 Real News" if prediction == 0 else "🔴 Fake News"
        st.success(f"Prediction: **{result}**")
