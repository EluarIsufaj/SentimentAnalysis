import joblib
from sentimentAnalysisModel import clean_tweet


#Here we load the model and vectorizer
model = joblib.load('logistic_model.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')

def predict_sentiment(text):
    cleaned = clean_tweet(text)
    features = vectorizer.transform([cleaned])
    pred = model.predict(features)[0]
    label_map = {0: "Negative", 4: "Positive"}
    return label_map.get(pred, "Unknown")


text = input("Enter your text:\n")

print(predict_sentiment(text))
