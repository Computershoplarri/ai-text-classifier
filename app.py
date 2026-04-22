import os
import pickle

base_dir = os.path.dirname(__file__)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))



