import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Load data
data = pd.read_csv("F:/Ai interview failure Analyser/data/interview_data.csv")

# Combine text features
data["combined_text"] = data["resume_text"] + " " + data["jd_text"] + " " + data["self_reflection"]

# Vectorization
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(data["combined_text"])
y = data["failure_reason"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)

# Save model
pickle.dump(model, open("F:/Ai interview failure Analyser/model/failure_model.pkl", "wb"))
pickle.dump(vectorizer, open("F:/Ai interview failure Analyser/model/vectorizer.pkl", "wb"))
