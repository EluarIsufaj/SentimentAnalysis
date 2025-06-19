# 🧠 Twitter Sentiment Analysis with Logistic Regression

This project uses machine learning (Logistic Regression) to classify the sentiment of tweets as either **positive** or **negative**. It involves data preprocessing, text vectorization (TF-IDF with n-grams), and model training using the [Sentiment140 dataset](https://www.kaggle.com/datasets/kazanova/sentiment140).

---

## 📦 Features

- Preprocessing of tweet text (lowercasing, mention removal, etc.)
- TF-IDF vectorization with n-gram support
- Model training using Logistic Regression
- Real-time sentiment prediction with a separate Python script
- Accuracy: ~77% on 320,000 test tweets

---

## 🧪 How to Use

### 🔧 Requirements
- Python 3.10+
- `pandas`, `scikit-learn`, `joblib`

### 📂 Dataset

> 🔗 [Sentiment140 Dataset on Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140)

Download the CSV file and place it in your project directory as `tweets.csv`.

### 🚀 Running the Project

1. **Train the model**  
   Run the training script to clean the data, vectorize, and save the model:
   ```bash
   python sentimentAnalysisModel.py




💾 Saved Model & Vectorizer
To avoid re-training the model every time you want to make a prediction, the trained Logistic Regression model and TF-IDF vectorizer are saved using joblib. This allows fast, real-time sentiment predictions without repeating the training process.

logistic_model.joblib: the trained classification model

tfidf_vectorizer.joblib: the fitted TF-IDF transformer

These are loaded directly in predictSentiment.py to process and classify new text instantly.
