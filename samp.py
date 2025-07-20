import pickle
import re

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

# Load model and vectorizer
with open('chatbot_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Test prediction
test_text = "I am feeling stressed"
processed_text = preprocess(test_text)
X_test = vectorizer.transform([processed_text])

predicted_tag = model.predict(X_test)[0]

print(f"Input text: {test_text}")
print(f"Processed text: {processed_text}")
print(f"Predicted tag: {predicted_tag}")
