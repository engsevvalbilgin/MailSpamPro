# Spam Email Classification using Naive Bayes

This project demonstrates how to build a **Spam Email Classifier** using the **Naive Bayes algorithm** with **text preprocessing**, **feature extraction**, and **model training**. The dataset used in this project is from the Kaggle repository, which contains labeled spam and ham emails.

You can download the dataset from Kaggle using the following link:
[Spam Mails Dataset on Kaggle](https://www.kaggle.com/datasets/venky73/spam-mails-dataset/data)

## Prerequisites

Before running the script, you need to install the following Python libraries:
- pandas
- nltk
- scikit-learn

You can install the required libraries using `pip`:

```bash
pip install pandas nltk scikit-learn
```

## Steps Involved

### 1. **Download DataSet**
The dataset is loaded from a CSV file (`spam_ham_dataset.csv`), which contains labeled email texts (spam or ham). The dataset is read using the `pandas` library, and the relevant columns (`label` and `text`) are selected. 

```python
df = pd.read_csv("spam_ham_dataset.csv", encoding="latin-1")[["label", "text"]]
df.columns = ["label", "text"]
```

### 2. **Label Mapping**
The labels `spam` and `ham` are mapped to binary values:
- `spam` is mapped to `1`
- `ham` is mapped to `0`

```python
df["label"] = df["label"].map({"spam": 1, "ham": 0})
```

### 3. **Data Preprocessing**
The text data is preprocessed using the function `temizle`:
- Convert all text to lowercase.
- Remove punctuation and spaces using regular expressions.

```python
def temizle(metin):
    metin = metin.str.lower()  # Convert to lowercase
    metin = metin.apply(lambda x: re.sub(r'[^\w\s]', '', x))  # Remove punctuation
    metin = metin.apply(lambda x: re.sub('\n', '', x))  # Remove newlines
    metin = metin.apply(lambda x: re.sub('\r', '', x))  # Remove carriage returns
    return metin
```

The cleaning function is applied to the `text` column to ensure the text is ready for vectorization.

### 4. **Feature Extraction**
The text data is converted into a numerical format using the **CountVectorizer** from **scikit-learn**. The `stopwords` from the **nltk** library are used to exclude common words (like "the", "is", etc.) that don’t provide useful information for classification.

```python
vectorizer = CountVectorizer(stop_words=stopwords.words('english'))
X = vectorizer.fit_transform(df["text"])
y = df["label"]
```

### 5. **Model Training**
The dataset is split into training and testing sets using **train_test_split**. The Naive Bayes model (**MultinomialNB**) is then trained on the training data.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)
```

### 6. **Model Testing**
After training the model, the performance is evaluated on the test data using **accuracy score**. The accuracy score tells you how well the model performs in classifying unseen emails.

```python
y_pred = model.predict(X_test)
print("Accuracy Score:", accuracy_score(y_test, y_pred))
```

### 7. **Testing with New Examples**
Finally, the model is tested with new example messages. The input message is transformed using the same vectorizer, and a prediction is made.

```python
ornek_mesaj = ["BIG DISCOUNT"]
ornek_mesaj_v = vectorizer.transform(ornek_mesaj)
tahmin = model.predict(ornek_mesaj_v)
print("Prediction:", "Spam" if tahmin[0] else "Ham")
```

### Sample Output

```
Accuracy Score: 0.978978978978979
Prediction: Spam
```

## Conclusion

This project demonstrates the power of **Naive Bayes classification** for **text data**. By using a clean and well-preprocessed dataset, you can effectively classify spam and ham emails. The model achieves high accuracy on the given dataset and can be extended for further applications like **spam filtering** for email systems.
