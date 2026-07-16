import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load Dataset
df = pd.read_csv("dataset/Training.csv")

if "Unnamed: 133" in df.columns:
    df = df.drop(columns=["Unnamed: 133"])

print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

# Features
X = df.drop("prognosis", axis=1)

# Target
y = df["prognosis"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

# Train Model
model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:")
print(accuracy)

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Save Model
joblib.dump(model, "model/disease_model.pkl")

print("\nModel Saved Successfully!")