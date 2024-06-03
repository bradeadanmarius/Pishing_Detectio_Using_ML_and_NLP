import tkinter as tk
from tkinter import messagebox, Toplevel
from PIL import ImageGrab
import pytesseract
from joblib import load
import pandas as pd
from sqlalchemy import create_engine

# Configuration for pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\brade\anaconda3\envs\phishing_email_detection_py39\Library\bin\tesseract.exe'
pytesseract.pytesseract.tesseract_data_dir = r'C:\Users\brade\anaconda3\envs\phishing_email_detection_py39\share\tessdata'

# Load the trained machine learning model and the TF-IDF vectorizer
model = load('D:/Year 3 School/CS6P05 Project - Year 2023-24/Phishing detection using ML and NLP Dan-Marius Bradea codebase/logreg_model.pkl')
tfidf_vectorizer = load('D:/Year 3 School/CS6P05 Project - Year 2023-24/Phishing detection using ML and NLP Dan-Marius Bradea codebase/tfidf_vectorizer.pkl')

# Database connection
engine = create_engine('sqlite:///D:/Year 3 School/CS6P05 Project - Year 2023-24/Phishing detection using ML and NLP Dan-Marius Bradea codebase/phishing_emails.db')

# Determines the color based on phishing likelihood
def determine_color(likelihood):
    if likelihood < 25:
        return '#58D68D'  # Green
    elif 25 <= likelihood < 45:
        return '#F7DC6F'  # Yellow-Green
    elif 45 <= likelihood < 60:
        return '#F8C471'  # Yellow-Orange
    elif 60 <= likelihood < 80:
        return '#F0B27A'  # Orange
    else:
        return '#E74C3C'  # Red
    
# Function to check for phishing in the screenshot
def check_phishing():
    messagebox.showinfo("Processing", "Analyzing the screen for phishing... Please wait.")
    try:
        screenshot = ImageGrab.grab()
        text = pytesseract.image_to_string(screenshot)
        if not text.strip():
            raise ValueError("No text found on the screen. Please try again.")
        
        tfidf_features = tfidf_vectorizer.transform([text])
        prediction_proba = model.predict_proba(tfidf_features)
        likelihood = round(prediction_proba[0][1] * 100, 2)

        # Determine classification based on likelihood
        is_phishing = 1 if likelihood > 59 else 0

        # Save the result to the database
        df = pd.DataFrame({
            'email_text': [text],
            'phishing_likelihood': [likelihood],
            'is_phishing': [is_phishing]
        })
        df.to_sql('analyzed_emails', engine, if_exists='append', index=False)

        # Create a popup window for the result
        result_window = Toplevel()
        result_window.title("Phishing Detection Result")
        result_window.geometry("+750+50")  # Position top-right corner
        result_window.configure(bg='white')
        result_window.attributes('-topmost', 'true')  # Always on top
        
        # Style the result label
        result_label = tk.Label(result_window, text=f"Phishing Likelihood: {likelihood}%", font=("Arial", 14), bg='white', fg=determine_color(likelihood))
        result_label.pack(pady=10, padx=10)

        # OK button to close the popup
        ok_button = tk.Button(result_window, text="OK", command=result_window.destroy, bg='#4CAF50', fg='white', relief='flat', font=("Arial", 12), borderwidth=0)
        ok_button.pack(pady=10)

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

# Main application window
root = tk.Tk()
root.title("Phishing Email Detection")
root.geometry("250x150+1000+50")  # Smaller size and positioned top-right corner
root.configure(bg='white')
root.attributes('-topmost', 'true')  # Always on top

# Rounded button
button = tk.Button(root, text="Check Screen for Phishing", command=check_phishing, bg='#4CAF50', fg='white', relief='flat', font=("Arial", 12), borderwidth=0)
button.pack(expand=True, padx=20, pady=20)

root.mainloop()

