from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import os

# custom dataset
texts = [
    "I want a refund for my order",
    "I was charged twice",
    "My payment failed but money deducted",
    
    "My internet is not working",
    "App is crashing on startup",
    "System is very slow",
    
    "I want to buy your product",
    "Tell me pricing details",
    "Do you offer discounts",
    
    "Win money now click here",
    "You are selected for prize",
    "Free gift claim now"
]

labels = [
    "billing","billing","billing",
    "technical","technical","technical",
    "sales","sales","sales",
    "spam","spam","spam"
]

# vectorize
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# model
model = LogisticRegression()
model.fit(X, labels)

# save
os.makedirs("model", exist_ok=True)
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Support Ticket Model Ready")

