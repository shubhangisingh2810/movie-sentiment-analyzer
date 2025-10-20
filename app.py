import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langdetect import detect
from joblib import load
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from googletrans import Translator
import re

# Load model and vectorizer
model = load("model/sentiment_model.pkl")
vectorizer = load("model/tfidf.pkl")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
translator = Translator()

# Preprocess text
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

# Set page config
st.set_page_config(page_title="🎬 Movie Sentiment Analyzer", layout="wide")

# Custom CSS for modern UI
# 🌟 Modern 3D-style UI with gradient & animation
st.markdown("""
    <style>
        body {
            background: linear-gradient(to right, #0f172a, #1e293b);
            color: #fff;
            font-family: 'Segoe UI', sans-serif;
        }

        .main > div:first-child h1 {
            font-size: 2.8rem;
            background: linear-gradient(to right, #38bdf8, #6366f1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 1rem;
        }

        textarea, .stTextInput>div>div>input, .stFileUploader > div {
            background-color: #1e293b;
            color: white;
            border-radius: 10px;
            border: 1px solid #475569;
            font-size: 1rem;
            padding: 0.75rem;
        }

        .stDataFrame {
            background-color: #1e293b !important;
            color: #e2e8f0;
        }

        button[kind="primary"] {
            background: linear-gradient(to right, #6366f1, #38bdf8);
            border: none;
            color: white;
            border-radius: 12px;
            padding: 0.6rem 1.2rem;
            font-weight: bold;
            font-size: 1rem;
            transition: all 0.3s ease;
        }

        button[kind="primary"]:hover {
            transform: scale(1.05);
            background: linear-gradient(to right, #2563eb, #06b6d4);
        }

        .sidebar .sidebar-content {
            background-color: #0f172a;
            color: white;
        }

        .stPlotlyChart, .stAltairChart, .stPyplot {
            background-color: transparent;
            padding-top: 1rem;
        }
    </style>
""", unsafe_allow_html=True)


# Sidebar
st.sidebar.title("ℹ️ Project Info")
st.sidebar.markdown("""
This is a **Movie Review Sentiment Analyzer**.

🔍 Features:
- Analyze individual or batch reviews
- Translates non-English to English
- Beautiful sentiment bar chart
""")

# Header
st.title("🎬 Movie Sentiment Analyzer")
st.subheader("Enter a movie review below or upload a .txt file with multiple reviews.")

# Single review input
review_input = st.text_area("✍️ Enter a single movie review here", height=120)

# .txt file upload
uploaded_file = st.file_uploader("📄 Or upload a .txt file (one review per line)", type="txt")

# Analyze function with translation
def analyze_reviews(reviews):
    sentiments = []
    languages = []
    translations = []

    for review in reviews:
        try:
            lang = detect(review)
        except:
            lang = "unknown"

        languages.append(lang)

        if lang != "en":
            try:
                translated = translator.translate(review, src=lang, dest='en').text
            except:
                translated = "Translation Failed"
        else:
            translated = review

        translations.append(translated)

        clean = preprocess_text(translated)
        vector = vectorizer.transform([clean])
        pred = model.predict(vector)[0]
        sentiments.append("Positive" if pred == 1 else "Negative")

    return pd.DataFrame({
        "Original Review": reviews,
        "Detected Language": languages,
        "Translated Review": translations,
        "Sentiment": sentiments
    })

# Analyze button
if st.button("🔍 Analyze Reviews"):
    if review_input:
        df = analyze_reviews([review_input])
        st.dataframe(df)
    elif uploaded_file:
        content = uploaded_file.read().decode("utf-8").splitlines()
        if len(content) > 200:
            st.warning("⚠️ Please upload a file with up to 200 reviews.")
        else:
            df = analyze_reviews(content)
            st.dataframe(df)

            st.subheader("📊 Sentiment Distribution")
            fig, ax = plt.subplots()
            sns.countplot(x="Sentiment", data=df, palette="coolwarm")
            st.pyplot(fig)
    else:
        st.warning("⚠️ Please enter a review or upload a file first.")
