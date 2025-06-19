import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


def clean_tweet(text):
    text = text.lower()
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'http\S+|www.\S+', '', text)  # Remove URLs
    text = re.sub(r'#', '', text)  # Remove hashtag symbol
    text = re.sub(r'[^a-z\s]', '', text)  # Remove non-letter characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text


if __name__ == "__main__":
    column_names = ['target', 'ids', 'date', 'flag', 'user', 'text']
    df = pd.read_csv("tweets.csv", encoding='latin-1', names=column_names)
    df = df[['target', 'text']]
    print("Read the data")




    df['clean_text'] = df['text'].apply(clean_tweet)


    X = df['clean_text']      # Cleaned tweet texts
    y = df['target']          # Sentiment labels (0 or 4)

    #We split into train and test sets (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #Here we create a TF-IDF vectorizer and transform data
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,3), stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)


    print("Beginning to train")

    model = LogisticRegression(max_iter=300)
    model.fit(X_train_tfidf, y_train)


    y_pred = model.predict(X_test_tfidf)

    # Let us see how well our model performed
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))


