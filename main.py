import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import torch as t



# 1. Download DataSet
df = pd.read_csv("spam_ham_dataset.csv", encoding="latin-1")[["label", "text"]]
df.columns = ["label", "text"]

import re

# Mapping spam as 1 and ham as 0
df["label"] = df["label"].map({"spam": 1, "ham": 0})

# Drop duplicates based on 'text' column
df.drop_duplicates(subset="text", inplace=True)


# 2. Data Preprocessing
def temizle(metin):
    # Lowercase, remove punctuation and spaces
    metin = metin.str.lower()  # Convert to lowercase
    metin = metin.apply(lambda x: re.sub(r'[^\w\s]', '', x))  # Remove punctuation
    metin = metin.apply(lambda x: re.sub('\n', '', x))  # Remove spaces
    metin = metin.apply(lambda x: re.sub('\r', '', x))  # Remove spaces
    return metin


# Apply cleaning function
df["text"] = temizle(df["text"])

# Check the dataframe
#print(df.head())

# 3. Feature Extraction
vectorizer = CountVectorizer(stop_words=stopwords.words('english'))
X = vectorizer.fit_transform(df["text"])
y = df["label"]

# 4. Training Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

# 5. Testing Model
y_pred = model.predict(X_test)
print("Accuracy Score:", accuracy_score(y_test, y_pred))


# 6. Örnek Mesajı Test Etme
ornek_mesaj = ["Congragulations! You won money!"]
ornek_mesaj_v = vectorizer.transform(ornek_mesaj)
tahmin = model.predict(ornek_mesaj_v)
print("Prediction:", "Spam" if tahmin[0] else "Ham")
