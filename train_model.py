import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle

# Load datasets
fake = pd.read_csv("Fake_cleaned.csv", encoding='ISO-8859-1')
true = pd.read_csv("True_cleaned.csv", encoding='ISO-8859-1')


# Add labels
fake['label'] = 0  # Fake
true['label'] = 1  # Real

# Combine data
data = pd.concat([fake, true])
data = data.sample(frac=1).reset_index(drop=True)  # Shuffle

# Split
X = data['text']
y = data['label']

# Vectorize
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_vec = vectorizer.fit_transform(X)

# Train model
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_vec, y)

# Save model and vectorizer
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("✅ Model trained and saved successfully!")
