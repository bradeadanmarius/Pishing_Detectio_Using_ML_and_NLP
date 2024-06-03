# -*- coding: utf-8 -*-
"""
Streamlined Training Script for Phishing Email Detection
"""

import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from joblib import dump

# Setup the connection to the database
engine = create_engine('sqlite:///D:/Year 3 School/CS6P05 Project - Year 2023-24/Phishing detection using ML and NLP Dan-Marius Bradea codebase/phishing_emails.db')
data = pd.read_sql("SELECT Text, Class FROM phishing_emails", engine)

# Feature selection using email text and class labels
X = data['Text']  # Features - Email texts
y = data['Class'].astype(int)  # Target - Class labels

# Splitting the data into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializing the TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the training data to a document-term matrix
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Initializing and training the Logistic Regression model
logreg = LogisticRegression()
logreg.fit(X_train_tfidf, y_train)

# Save the updated Logistic Regression model and TF-IDF vectorizer, replacing the old files
dump(logreg, 'D:/Year 3 School/CS6P05 Project - Year 2023-24/Phishing detection using ML and NLP Dan-Marius Bradea codebase/logreg_model.pkl')
dump(tfidf_vectorizer, 'D:/Year 3 School/CS6P05 Project - Year 2023-24/Phishing detection using ML and NLP Dan-Marius Bradea codebase/tfidf_vectorizer.pkl')

# Output the number of training and test samples
print("Training set size:", X_train.shape[0])
print("Test set size:", X_test.shape[0])
print("Model and vectorizer updated and saved.")
