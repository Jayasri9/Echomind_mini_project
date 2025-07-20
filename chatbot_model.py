
import json
import pickle
import re
import nltk
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
import numpy as np

# Download NLTK dependencies
try:
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('stopwords')
except:
    pass

lemmatizer = WordNetLemmatizer()

def correct_spelling(text):
    try:
        return str(TextBlob(text).correct())
    except:
        return text

def preprocess(text):
    text = correct_spelling(text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = ' '.join(text.split())  # Remove extra whitespace
    return text.strip()

# Load your existing intents
with open("intents.json", "r", encoding="utf-8") as file:
    intents = json.load(file)

# Prepare training data
corpus = []
labels = []

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        processed = preprocess(pattern)
        if processed:  # Only add non-empty patterns
            corpus.append(processed)
            labels.append(intent["tag"])

print(f"Total training samples: {len(corpus)}")
print(f"Unique labels: {len(set(labels))}")

# Print label distribution to see if any classes are underrepresented
from collections import Counter
label_counts = Counter(labels)
print("\nLabel distribution:")
for label, count in label_counts.most_common():
    print(f"  {label}: {count}")

# Improved TF-IDF configuration - more generalized
vectorizer = TfidfVectorizer(
    max_features=1500,
    ngram_range=(1, 2),  # Unigrams and bigrams
    min_df=1,
    max_df=0.85,
    lowercase=True,
    token_pattern=r'\b[a-zA-Z]{2,}\b'  # Only words with 2+ letters
)

X = vectorizer.fit_transform(corpus)
print(f"Feature matrix shape: {X.shape}")

# Try different alpha values to find the best one
alphas = [0.01, 0.1, 0.5, 1.0]
best_alpha = 1.0
best_score = 0

print("\nTesting different alpha values:")
for alpha in alphas:
    model_test = MultinomialNB(alpha=alpha)
    scores = cross_val_score(model_test, X, labels, cv=3, scoring='accuracy')
    avg_score = scores.mean()
    print(f"Alpha {alpha}: {avg_score:.3f} (+/- {scores.std() * 2:.3f})")
    if avg_score > best_score:
        best_score = avg_score
        best_alpha = alpha

print(f"\nBest alpha: {best_alpha} with score: {best_score:.3f}")

# Train final model with best alpha
model = MultinomialNB(alpha=best_alpha)
model.fit(X, labels)

# Save model and vectorizer
with open("chatbot_model.pkl", "wb") as mf:
    pickle.dump(model, mf)

with open("vectorizer.pkl", "wb") as vf:
    pickle.dump(vectorizer, vf)

print("✅ Model & vectorizer saved successfully!")

# Test some cases to see prediction scores
test_cases = [
    "i have future anxiety",
    "i'm anxious about my future", 
    "hello",
    "feeling stressed",
    "thanks"
]

print("\n--- Prediction Analysis ---")
for test in test_cases:
    processed = preprocess(test)
    X_test = vectorizer.transform([processed])
    prediction = model.predict(X_test)[0]
    probabilities = model.predict_proba(X_test)[0]
    
    # Get top 3 predictions with scores
    top_indices = np.argsort(probabilities)[-3:][::-1]
    print(f"\nInput: '{test}'")
    for i, idx in enumerate(top_indices):
        class_name = model.classes_[idx]
        score = probabilities[idx]
        marker = " ← PREDICTED" if i == 0 else ""
        print(f"  {i+1}. {class_name}: {score:.3f}{marker}")