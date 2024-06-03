# -*- coding: utf-8 -*-
"""
Updated Script for Merging New Emails into Existing Training Data
"""

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

# Setup the connection to the database
engine = create_engine('sqlite:///D:/Year 3 School/CS6P05 Project - Year 2023-24/Project/phishing_emails.db')
connection = engine.connect()

try:
    # SQL query to merge new emails into the training data, avoiding duplicates
    merge_query = text("""
    INSERT INTO phishing_emails (Text, Class)
    SELECT email_text, is_phishing FROM analyzed_emails
    WHERE NOT EXISTS (
        SELECT 1 FROM phishing_emails WHERE phishing_emails.Text = analyzed_emails.email_text
    );
    """)

    # Execute the merge query
    result = connection.execute(merge_query)
    print(f"{result.rowcount} rows were added to the phishing_emails table.")

    # Commit the transaction if necessary (depends on DB API)
    connection.commit()

except SQLAlchemyError as e:
    print(f"An error occurred: {e}")
finally:
    # Close the connection
    connection.close()

# After running this script, check to see if the rows have been added successfully
print("Data merging operation completed.")
