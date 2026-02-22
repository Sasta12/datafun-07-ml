# Machine Learning Workflow Demo
# Author: Your Name
# Date: 2026-02-22
"""
This script demonstrates a typical machine learning workflow using Python, pandas, seaborn, matplotlib, and scikit-learn. We use the classic Iris dataset for classification.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Set plot style
sns.set(style="whitegrid")

# Load the Iris dataset
df = sns.load_dataset("iris")
print("First 5 rows of the dataset:")
print(df.head())

# Check for missing values and basic info
print("\nDataFrame info:")
df.info()
print("\nMissing values:")
print(df.isnull().sum())

# Summary statistics
print("\nSummary statistics:")
print(df.describe())

# Visualize distributions of numerical features
df.hist(figsize=(10, 8))
plt.tight_layout()
plt.show()

# Value counts for categorical feature
print("\nSpecies value counts:")
print(df["species"].value_counts())

# Encode the target variable
df["species_encoded"] = df["species"].astype("category").cat.codes

# Features and target
X = df.drop(["species", "species_encoded"], axis=1)
y = df["species_encoded"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Show the first 5 rows of scaled features
print("\nFirst 5 rows of scaled features:")
print(pd.DataFrame(X_scaled, columns=X.columns).head())

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")

# Train a logistic regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {acc:.2f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=df["species"].unique(),
    yticklabels=df["species"].unique(),
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Classification report
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=df["species"].unique()))

# Predict on a few test samples
sample_idx = [0, 1, 2]
sample_X = X_test[sample_idx]
sample_true = y_test.iloc[sample_idx]
sample_pred = model.predict(sample_X)
print("\nTrue labels:", sample_true.values)
print("Predicted labels:", sample_pred)
