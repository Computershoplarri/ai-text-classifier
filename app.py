import os
import pickle

base_dir = os.path.dirname(__file__)

model = pickle.load(open(os.path.join(base_dir, "model", "model.pkl"), "rb"))
vectorizer = pickle.load(open(os.path.join(base_dir, "model", "vectorizer.pkl"), "rb"))


