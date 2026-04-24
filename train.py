from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

texts = [
    # billing
    "I want a refund for my order",
    "I was charged twice for my subscription",
    "Payment deducted but service not activated",
    "Refund has not been processed yet",
    "Wrong billing amount charged",
    "Invoice shows incorrect price",

    # technical
    "My app keeps crashing on startup",
    "Internet connection is not working",
    "Unable to login to my account",
    "Website is loading very slowly",
    "There is a bug in your software",
    "Server error when I try to open dashboard",

    # sales
    "I want to purchase your product",
    "Can you share pricing details?",
    "Do you offer any discounts?",
    "I am interested in your service",
    "What is the cost of your plan?",
    "Tell me more about your features",

    # spam
    "Win money now click this link",
    "You have been selected for a free gift",
    "Claim your prize now",
    "Earn cash fast from home",
    "Limited time offer click here",
    "Congratulations you won a reward"
]

labels = [
    "billing","billing","billing","billing","billing","billing",
    "technical","technical","technical","technical","technical","technical",
    "sales","sales","sales","sales","sales","sales",
    "spam","spam","spam","spam","spam","spam"
]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

model = LogisticRegression(max_iter=1000)
model.fit(X, labels)

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Model trained and saved")



