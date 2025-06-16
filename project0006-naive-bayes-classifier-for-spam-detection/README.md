### Description:

Naive Bayes is a probabilistic classifier based on Bayes’ Theorem with an assumption of feature independence. It's especially effective for spam detection, thanks to its ability to handle high-dimensional text data. In this project, we’ll build a spam filter using a bag-of-words model and train a Naive Bayes classifier with scikit-learn.

This model uses the Multinomial Naive Bayes algorithm with a bag-of-words representation, making it suitable for large-scale spam filtering tasks.

## Spam Detection with Naive Bayes - Code Explanation

This script demonstrates a simple spam detection system using machine learning techniques in Python. It utilizes the Naive Bayes algorithm to classify email messages as either "Spam" or "Ham" (non-spam).

### Libraries Used

```python
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
```

* **train\_test\_split**: Splits data into training and testing subsets.
* **CountVectorizer**: Converts text into a numerical format using bag-of-words.
* **MultinomialNB**: Implements the Naive Bayes classifier for multinomial models.
* **classification\_report**: Evaluates the performance of the classifier.

### Sample Dataset

```python
data = [
    ("Congratulations, you've won a $1000 Walmart gift card. Click to claim now!", 1),
    ("Reminder: Your appointment is scheduled for 10 AM tomorrow.", 0),
    ("URGENT: You have been selected for a prize. Act fast!", 1),
    ("Hi, can we reschedule our meeting?", 0),
    ("Earn money from home without doing anything!", 1),
    ("Your Amazon order has been shipped.", 0),
    ("Claim your free vacation now by clicking this link!", 1),
    ("Lunch at 1 PM?", 0),
    ("You are pre-approved for a $5000 loan. No credit check!", 1),
    ("Can you send me the report by EOD?", 0)
]
```

This dataset contains labeled email messages, where `1` indicates spam and `0` indicates ham.

### Data Preparation

```python
texts, labels = zip(*data)
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.3, random_state=42)
```

* Splits messages and labels.
* Further splits them into training and testing sets with 70%-30% split.

### Text Vectorization

```python
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
```

* **CountVectorizer** converts text into a sparse matrix of token counts.
* `fit_transform()` learns vocabulary and transforms training data.
* `transform()` applies the same vocabulary to the test set.

### Model Training and Prediction

```python
classifier = MultinomialNB()
classifier.fit(X_train_vec, y_train)
y_pred = classifier.predict(X_test_vec)
```

* Trains the Naive Bayes classifier on the vectorized training data.
* Makes predictions on the vectorized test data.

### Evaluation

```python
print("Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))
```

* Prints precision, recall, f1-score, and support for both classes.

### Custom Email Prediction

```python
custom_email = ["Free entry in a contest to win $10000 cash!"]
custom_vec = vectorizer.transform(custom_email)
prediction = classifier.predict(custom_vec)

print("\nCustom Email Prediction:")
print(f"'{custom_email[0]}' is classified as:", "Spam" if prediction[0] == 1 else "Ham")
```

* Tests the model on a new, unseen message.
* Classifies the email and prints the result.

### Summary

This code serves as a basic implementation of text classification for spam detection. It covers:

* Dataset preparation
* Text vectorization using CountVectorizer
* Model training using Naive Bayes
* Performance evaluation
* Real-world usage with custom input


