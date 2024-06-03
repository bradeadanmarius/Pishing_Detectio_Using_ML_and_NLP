# README for Phishing Detection Using Natural Language Processing (NLP)

This README document accompanies the submission of the software development project artifact for the Phishing Detection system using Natural Language Processing (NLP). Below you will find comprehensive guidance on the contents of the project submission, file descriptions, and instructions for installing, executing, and testing the software.



# Project Contents Overview:

The project submission is provided as a ZIP file containing all necessary computer programs, data files, and documentation required for installation, execution, and testing of the phishing email detection software. Below is a detailed overview of the contents:

1. Python Scripts:**
   - "merging_new_emails_into_existing_table.py": Integrates newly collected email data into the existing database table for ongoing model training.
   - "Script to clean CSV file.py": Cleans and preprocesses raw email data from CSV files for further analysis.
   - "stream_line_script_model_training.py": Contains streamlined code for training the phishing detection model.
   - "Main python script.py": The primary script that includes the full functionality from data loading to model application.
   - "improved_visual_aspect_of_application.py": Enhancements to the GUI aspects of the application for better user experience.

2. Data Files:
   - "phishing_emails.db": The SQLite database containing processed email data used for model training.
   - "fraud_email_.csv": Raw dataset of emails used for initial data processing and model training.
   - "classification_report.json": Contains a JSON formatted report of model performance metrics such as precision, recall, and F1-score.

3. Model and Vectorizer Files:
   - "logreg_model.pkl": Serialized Logistic Regression model trained to detect phishing emails.
   - "tfidf_vectorizer.pkl": Serialized TF-IDF vectorizer used for converting email texts into numeric features.

4. Dependency List:
   - "package_list.txt": A list of Python packages and their versions required to execute the scripts successfully.

5. Documentation and Miscellaneous:
   - "README.txt": This document explaining the project setup and execution.

# Installation and Setup Instructions:

Pre-requisites:
Ensure Anaconda is installed on your machine, which will be used to manage the Python environment and dependencies. The project was developed using Spyder, an IDE provided by Anaconda.

Installation Steps:
1. Extract the ZIP File:
   - After downloading, extract the contents of the ZIP file into a directory of your choice.

2. Environment Setup:
   - Open Anaconda Prompt and navigate to the project directory.
   - Create a new virtual environment:
     ```
     conda create --name phishing_detection python=3.8
     ```
   - Activate the environment:
     ```
     conda activate phishing_detection
     ```
   - Install required packages:
     ```
     pip install -r package_list.txt
     ```

Configuring Paths:
   - You will need to modify the paths in the scripts (notably in database and model loading lines) to match the locations where you have stored the files on your system.

# Running the Software:

1. Starting the Application:

- Open Spyder or your preferred Python IDE from the Anaconda environment where you installed the dependencies.
   - Navigate to the folder where you extracted the project files.
   - Run `Main python script.py` to launch the phishing detection system. This script integrates all components and provides the primary user interface.

Testing the Software:
   - Use the `Script to clean CSV file.py` to preprocess new raw data.
   - Run `stream_line_script_model_training.py` to retrain the model with any new or updated data.
   - Test the model's effectiveness using `merging_new_emails_into_existing_table.py` to merge new data and observe the model's performance on updated datasets.

Using the GUI:
   - Execute `improved_visual_aspect_of_application.py` to interact with the enhanced graphical user interface, which allows for easy input of data and displays the phishing detection results.

# Technical Documentation:

Each file and folder in the ZIP package serves specific functions:
- Python Scripts handle everything from data preprocessing, model training, to user interface operations.
- Data Files include the database and raw data essential for training and validation.
- Model and Vectorizer Files are crucial for the immediate deployment of the system, allowing for rapid setup and execution without the need for re-training on initial launch.
- Package List ensures compatibility and proper environment setup for running the software.

# Additional Notes:

This README aims to provide clear and concise instructions to facilitate the use and evaluation of the phishing detection system but can be expanded as required by project specifics or supervisor recommendations.

For any further clarification or issues with setup and execution, refer to the contact details provided in the project submission or consult the support documentation included in the ZIP file.



This README structure provides a comprehensive guide for users to understand the purpose of each component, install and run the application, and perform necessary tests to evaluate the system's functionality. It adheres to the submission guidelines by detailing each requirement and ensuring the user has all the information needed for a successful deployment and use of the phishing detection system.