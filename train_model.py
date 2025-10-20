# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from joblib import dump
from preprocess import clean_text
import os

def train_and_save_model():
    df = pd.read_csv("data/IMDB Dataset.csv")
    df['clean_review'] = df['review'].apply(clean_text)
    df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

    X = df['clean_review']
    y = df['label']

    vectorizer = TfidfVectorizer(max_features=5000)
    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

    model = MultinomialNB()
    model.fit(X_train, y_train)

    # ✅ Ensure model folder exists
    os.makedirs("model", exist_ok=True)

    # ✅ Save using joblib
    dump(model, 'model/sentiment_model.joblib')
    dump(vectorizer, 'model/tfidf_vectorizer.joblib')

    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Model trained. Accuracy: {acc*100:.2f}%")

if __name__ == '__main__':
    train_and_save_model()
