from flask import Flask, request, render_template
import joblib  # For loading the model
import os
import sys

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from init import clean_text

app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load("model.pkl")  # Save your model and vectorizer
vectorizer = joblib.load("vectorizer.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    email_content = request.form['email_content']
    cleaned_content = clean_text(email_content)
    print(f"Cleaned content: {cleaned_content}")  # Debugging line
    vectorized_content = vectorizer.transform([cleaned_content])
    prediction = model.predict(vectorized_content)
    return 'Phishing' if prediction[0] == 1 else 'Legitimate'

if __name__ == '__main__':
    app.run(debug=True)
