import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Sample dataset (replace with IMDb, Twitter, or your own)
data = {
    "text": [
        "I love this product", "This is amazing", "I hate it", "Worst experience ever", "Not bad, pretty good"
    ],
    "label": [1, 1, 0, 0, 1]   # 1 = Positive, 0 = Negative
}

df = pd.DataFrame(data)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

# Vectorizer
vectorizer = TfidfVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Save vectorizer and model
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
pickle.dump(model, open("model.pkl", "wb"))

print("✅ Model trained & saved successfully!")
