import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------- LOAD DATA ----------------
data = pd.read_csv("F:/Ai interview failure Analyser/data/interview_data.csv")

# Filter classes (same as during training)
min_samples = 2
class_counts = data["failure_reason"].value_counts()
valid_classes = class_counts[class_counts >= min_samples].index.tolist()
filtered_data = data[data["failure_reason"].isin(valid_classes)]

print(f"Filtered data shape: {filtered_data.shape}")
filtered_data = data[data["failure_reason"].isin(valid_classes)].copy()

# Prepare text
filtered_data["combined_text"] = (
    filtered_data["resume_text"] + " " +
    filtered_data["jd_text"] + " " +
    filtered_data["self_reflection"]
)

X_text = filtered_data["combined_text"]
y = filtered_data["failure_reason"]

# ---------------- LOAD SAVED VECTORIZER ----------------
try:
    with open("F:/Ai interview failure Analyser/model/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    print(f"✓ Loaded vectorizer with {vectorizer.vocabulary_.__len__()} features")
except FileNotFoundError:
    print("✗ Vectorizer not found. Training new one...")
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=1000,
        min_df=2,
        max_df=0.95
    )
    vectorizer.fit(X_text)

# Transform text using the SAME vectorizer
X = vectorizer.transform(X_text)  # Use transform, NOT fit_transform
print(f"Transformed data shape: {X.shape}")

# ---------------- SPLIT DATA ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

# ---------------- LOAD MODEL ----------------
try:
    with open("F:/Ai interview failure Analyser/model/failure_model.pkl", "rb") as f:
        model = pickle.load(f)
    print(f"✓ Model loaded: {model.__class__.__name__}")
    print(f"Model expects {model.n_features_in_} features")
    print(f"We have {X.shape[1]} features")
    
    # Check if dimensions match
    if model.n_features_in_ != X.shape[1]:
        print(f"⚠ WARNING: Feature mismatch! Model expects {model.n_features_in_}, we have {X.shape[1]}")
        # If mismatch, retrain model
        print("Retraining model with current data...")
        model.fit(X_train, y_train)
except FileNotFoundError:
    print("✗ Model not found. Training new model...")
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

# ---------------- PREDICT ----------------
y_pred = model.predict(X_test)

# ---------------- METRICS ----------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

print("\n" + "="*50)
print("MODEL PERFORMANCE")
print("="*50)
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

# ---------------- CLASSIFICATION REPORT ----------------
print("\n" + "="*50)
print("CLASSIFICATION REPORT")
print("="*50)
print(classification_report(y_test, y_pred, zero_division=0))

# ---------------- VISUALIZATIONS ----------------
# 1. Performance Metrics Bar Chart
plt.figure(figsize=(10, 6))
metrics = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1
}

bars = plt.bar(metrics.keys(), metrics.values(), color=['blue', 'green', 'orange', 'red'])
plt.ylim(0, 1.0)
plt.title("Model Performance Metrics", fontsize=14, fontweight='bold')
plt.ylabel("Score", fontsize=12)
plt.grid(axis='y', alpha=0.3)

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.01,
        f"{height:.3f}",
        ha='center',
        va='bottom',
        fontsize=11,
        fontweight='bold'
    )

plt.tight_layout()
plt.savefig("F:/Ai interview failure Analyser/Saved Files/performance_metrics.png", dpi=300)
plt.show()

# 2. Confusion Matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)
disp.plot(cmap="Blues", values_format="d", ax=plt.gca())
plt.title("Confusion Matrix", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("F:/Ai interview failure Analyser/Saved Files/confusion_matrix.png", dpi=300)
plt.show()

# 3. Class Distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

y_train.value_counts().plot(kind='bar', ax=axes[0], color='skyblue')
axes[0].set_title("Training Set Class Distribution", fontsize=12)
axes[0].set_xlabel("Failure Reason")
axes[0].set_ylabel("Count")
axes[0].tick_params(axis='x', rotation=45)

y_test.value_counts().plot(kind='bar', ax=axes[1], color='lightcoral')
axes[1].set_title("Test Set Class Distribution", fontsize=12)
axes[1].set_xlabel("Failure Reason")
axes[1].set_ylabel("Count")
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig("F:/Ai interview failure Analyser/Saved Files/class_distribution.png", dpi=300)
plt.show()