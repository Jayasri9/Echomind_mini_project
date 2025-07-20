# chatbot_app.py

from flask import Flask, request, render_template, jsonify  # ✅ Combine all flask imports here
import json
import random
import re
import pickle
from textblob import TextBlob
from googletrans import Translator

# Load data and model
with open('intents.json') as file:
    intents = json.load(file)

model = pickle.load(open('chatbot_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Initialize app and translator
app = Flask(__name__)
translator = Translator()

def correct_spelling(text):
    return str(TextBlob(text))

def preprocess(text):
    text = correct_spelling(text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

def predict_class(text):
    text = preprocess(text)
    X = vectorizer.transform([text])
    return model.predict(X)[0]

def fallback_match(text):
    text = preprocess(text)
    tokens = set(text.split())
    best_score = 0.0
    best_tag = None

    for intent in intents['intents']:
        for pattern in intent['patterns']:
            pattern_tokens = set(preprocess(pattern).split())
            score = len(tokens & pattern_tokens) / max(len(pattern_tokens), 1)
            if score > best_score:
                best_score = score
                best_tag = intent['tag']

    return best_tag if best_score > 0.2 else None

def get_response(text):
    try:
        detected_lang = translator.detect(text).lang
        print(f"[DEBUG] Detected language: {detected_lang}")
        translated = translator.translate(text, dest='en').text
        print(f"[DEBUG] Translated input to English: {translated}")   
    except Exception as e:
        print(f"[ERROR] Translation failed: {e}")
        detected_lang = 'en'
        translated = text
    try:
        tag = predict_class(translated)
        print(f"[DEBUG] Predicted tag: {tag}")
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        tag = None
    response = None
    if tag:
        for intent in intents['intents']:
            if intent['tag'] == tag:
                response = random.choice(intent['responses'])
                print(f"[DEBUG] Response from predicted tag: {response}")
                break
    if not response:
        fallback = fallback_match(translated)
        print(f"[DEBUG] Fallback tag: {fallback}")
        if fallback:
            for intent in intents['intents']:
                if intent['tag'] == fallback:
                    response = random.choice(intent['responses'])
                    print(f"[DEBUG] Response from fallback: {response}")
                    break
    if not response:
        response = "I'm not sure I understand. Could you rephrase that?"
    try:
        if detected_lang != 'en':
            response = translator.translate(response, dest=detected_lang).text
            print(f"[DEBUG] Translated response back to '{detected_lang}': {response}")
    except Exception as e:
        print(f"[ERROR] Back-translation failed: {e}")
    return response
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chatbot_response():
    msg = request.form["msg"]
    reply = get_response(msg)
    return jsonify({"response": reply})  
if __name__ == "__main__":
    app.run(debug=True)