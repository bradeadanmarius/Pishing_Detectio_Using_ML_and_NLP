# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 11:13:56 2024

@author: brade
"""

import pandas as pd
from sqlalchemy import create_engine

# Define the path to your CSV file and the path for your SQLite database
csv_file_path = 'D:/Year 3 School/CS6P05 Project - Year 2023-24/Phishing detection using ML and NLP Dan-Marius Bradea codebase/fraud_email_.csv'
database_path = 'sqlite:///D:/Year 3 School/CS6P05 Project - Year 2023-24/Phishing detection using ML and NLP Dan-Marius Bradea codebase/phishing_emails.db'

# Load the dataset
data = pd.read_csv(csv_file_path)

# Clean the data
# Remove unnamed columns that contain mostly NaN values
data_cleaned = data.loc[:, ~data.columns.str.contains('^Unnamed')]

# Convert 'Class' to numeric, setting errors='coerce' to turn invalid parsing into NaN
data_cleaned['Class'] = pd.to_numeric(data_cleaned['Class'], errors='coerce')

# Drop rows where 'Class' is NaN after the conversion attempt
data_cleaned = data_cleaned.dropna(subset=['Class'])

# Now we safely convert 'Class' to integer, as all values should be numeric
data_cleaned['Class'] = data_cleaned['Class'].astype(int)

# Create a database engine
engine = create_engine(database_path)

# Import the cleaned data into the database
# 'phishing_emails' is the name of the table where we will store our dataset.
# If the table doesn't exist, it will be created automatically.
data_cleaned.to_sql('phishing_emails', con=engine, if_exists='replace', index=False)

print("Data imported successfully into the database.")
