# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 15:21:54 2024

@author: brade
"""

import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json

generate_outputs = False

# Loading the dataset from SQLite database created during the cleaning and transforming step
engine = create_engine('sqlite:///D:/Year 3 School/CS6P05 Project - Year 2023-24/Phishing detection using ML and NLP Dan-Marius Bradea codebase/phishing_emails.db')
data = pd.read_sql("SELECT Text, Class FROM phishing_emails", engine)

# Feature Selection
X = data['Text']  # Features - Email texts
y = data['Class'].astype(int)  # Target - Class labels

# Data Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# I will print the shapes of the training and testing data to verify
#print("Training set shape:", X_train.shape, y_train.shape)
#print("Testing set shape:", X_test.shape, y_test.shape)


# Initializing the TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the training data to a document-term matrix
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# Transform the test data to the same document-term matrix
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Note: X_train_tfidf and X_test_tfidf are now in a format suitable for machine learning models
# Print the shape of the TF-IDF matrices
print("Shape of TF-IDF training data:", X_train_tfidf.shape)
print("Shape of TF-IDF testing data:", X_test_tfidf.shape)

# Print some of the features (words) learned by the vectorizer
#feature_names = tfidf_vectorizer.get_feature_names_out()
#print("Some feature names:", feature_names[:20])  # Print the first 20 features


# I'm initializing the Logistic Regression model
logreg = LogisticRegression()

# Now, I'm training the model using the TF-IDF vectorized training data
logreg.fit(X_train_tfidf, y_train)

# After training, I use the model to make predictions on the test set
y_pred = logreg.predict(X_test_tfidf)

# Finally, I evaluate the model's performance using the classification report
#print(classification_report(y_test, y_pred))


if generate_outputs:

# Classification report as Json file.
    # Generate and print classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    print(classification_report(y_test, y_pred))



    # Generating the "metrics_dict" that will contain all the metrics data generated after chosing and training.
    # y_test and y_pred have already been defined from your previous classification task
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    
    # Save the classification report dictionary as a JSON file
    report_json_path = 'D:/Year 3 School/CS6P05 Project - Year 2023-24/Phishing detection using ML and NLP Dan-Marius Bradea codebase/classification_report.json'
    with open(report_json_path, 'w') as json_file:
        json.dump(report_dict, json_file, indent=4)

    print(f"Classification report saved as JSON to {report_json_path}")


# Density plot visualization
    # Generating a difrent wisualization for the data in the report as a Density plot from the metrix dictionary created earlier.
    df_metrics = pd.DataFrame({
            'Precision': [report_dict['0']['precision'], report_dict['1']['precision']],
            'Recall': [report_dict['0']['recall'], report_dict['1']['recall']],
            'F1-Score': [report_dict['0']['f1-score'], report_dict['1']['f1-score']]
            })

    plt.figure(figsize=(10, 6))

    # Plotting the density plot for Precision, Recall, and F1-Score
    sns.kdeplot(data=df_metrics, fill=True)

    plt.title('Density Plot of Precision, Recall, and F1-Score')
    plt.xlabel('Score')
    plt.ylabel('Density')

    # Save the plot
    density_plot_path = 'D:/Year 3 School/CS6P05 Project - Year 2023-24/Phishing detection using ML and NLP Dan-Marius Bradea codebase/metrics_density_plot.png'
    plt.savefig(density_plot_path)

    print(f"Density plot saved to {density_plot_path}")

# Confussion matrix visualization
    # Printing after generating the confussion matrix visual representation of the report data.
    # Generating confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Plotting confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", xticklabels=['Non-Phishing', 'Phishing'], yticklabels=['Non-Phishing', 'Phishing'])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')

    # Save the plot to a specific location
    plot_save_path = 'D:/Year 3 School/CS6P05 Project - Year 2023-24/Phishing detection using ML and NLP Dan-Marius Bradea codebase/confusion_matrix.png'
    plt.savefig(plot_save_path)

    # Notify where the plot is saved
    print(f"Plot saved to {plot_save_path}")



