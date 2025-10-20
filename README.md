# 🎬 Movie Sentiment Analyzer

[](https://www.python.org/)
[](https://streamlit.io/)
[](https://opensource.org/licenses/MIT)

## 🌟 Overview

The **Movie Sentiment Analyzer** is a web application built using **Streamlit** that utilizes a pre-trained machine learning model (Logistic Regression with TF-IDF vectorization) to classify movie reviews as either **Positive** or **Negative**. A key feature is its ability to automatically **detect and translate** non-English reviews into English before performing sentiment analysis, thanks to the `langdetect` and `googletrans` libraries.

It supports both single review analysis and batch processing via file upload.

## ✨ Key Features

  * **Single Review Analysis:** Quickly get the sentiment for a typed-in review.
  * **Batch Analysis:** Upload a `.txt` file (one review per line) for large-scale analysis (up to 200 reviews).
  * **Automatic Translation:** Uses `langdetect` and `googletrans` to translate reviews from any language to English before prediction.
  * **Interactive Visualization:** Displays a count plot of sentiment distribution for batch analysis.
  * **Modern UI:** Features a sleek, custom-styled Streamlit interface.

## ⚙️ Installation

### Prerequisites

You need **Python 3.8+** and **`pip`** installed on your system.

### 1\. Clone the Repository

```bash
git clone https://github.com/your-username/movie-sentiment-analyzer.git
cd movie-sentiment-analyzer
```

### 2\. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 3\. Install Dependencies

Install all necessary Python packages. You should create a `requirements.txt` file from your project's dependencies:

*(Assuming you have created a `requirements.txt` containing `streamlit`, `pandas`, `joblib`, `nltk`, `langdetect`, `googletrans`, `matplotlib`, `seaborn`)*

```bash
pip install -r requirements.txt
```

### 4\. Download NLTK Data

You need to manually download the `stopwords` data for NLTK's preprocessing step:

```bash
python
>>> import nltk
>>> nltk.download('stopwords')
>>> exit()
```

### 5\. Add Model Files

Ensure you have your pre-trained model and vectorizer files in the correct location:

```
movie-sentiment-analyzer/
├── app.py
├── model/
│   ├── sentiment_model.pkl   <-- Your trained model
│   └── tfidf.pkl             <-- Your fitted vectorizer
└── ...
```

## ▶️ How to Run the App

With your virtual environment activated and dependencies installed, run the Streamlit app from your project root directory:

```bash
streamlit run app.py
```

The application will open automatically in your web browser, typically at `http://localhost:8501`.

## 💻 Project Structure

```
.
├── app.py                  # Main Streamlit application script
├── model/                  # Directory for ML model and vectorizer files
│   ├── sentiment_model.pkl # The saved ML model (e.g., Logistic Regression)
│   └── tfidf.pkl           # The saved TF-IDF Vectorizer
├── requirements.txt        # List of Python dependencies
├── .gitignore              # Files/folders ignored by Git (e.g., venv)
└── README.md               # This file
```

## 🛠️ Technologies Used

  * **Python:** The core programming language.
  * **Streamlit:** For creating the interactive web application.
  * **Scikit-learn / Joblib:** For model training, saving, and loading.
  * **NLTK:** For text preprocessing (stopword removal, lemmatization).
  * **`langdetect`:** For detecting the language of the input review.
  * **`googletrans`:** For translating non-English reviews to English.
  * **Pandas, Matplotlib, Seaborn:** For data handling and visualization.

## 🤝 Contribution

Feel free to fork this repository, submit pull requests, or open issues for bugs and feature suggestions.

## 📜 License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.
