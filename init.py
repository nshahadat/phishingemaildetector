import pandas as pd
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load your dataset
data = pd.read_csv("./dataset/kaggle_datasheet.csv")
# print(data.head())

# Function to clean text
def clean_text(text):
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    return text.lower()  # Convert to lowercase

# Apply cleaning function
data['cleaned_content'] = data['body'].apply(clean_text)
# print("Cleaned content sample:\n", data['cleaned_content'].head())

# Check for any missing values
if data['cleaned_content'].isnull().any():
    print("Warning: There are missing values in cleaned_content.")


# Split features and labels
X = data['cleaned_content']
y = data['label']  # 1 for phishing, 0 for legitimate

# Check class distribution
# print("Class distribution:\n", y.value_counts())

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# print("Training set shape:", X_train.shape)
# print("Testing set shape:", X_test.shape)

# Vectorization
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Print shapes of vectorized data
# print("Shape of vectorized training data:", X_train_vec.shape)
# print("Shape of vectorized testing data:", X_test_vec.shape)

# Initialize and train the model
model = RandomForestClassifier(class_weight={0: 1.26, 1: 0.78})  # Adjust weights as needed
model.fit(X_train_vec, y_train)

# Make predictions
y_pred = model.predict(X_test_vec)

# Print classification report
# print(classification_report(y_test, y_pred))

# Train the model
model = RandomForestClassifier()
model.fit(X_train_vec, y_train)

# Save the model and vectorizer
joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

# print("Model and vectorizer saved successfully.")

# print(data['label'].value_counts())


