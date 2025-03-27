#!/usr/bin/env python
# coding: utf-8

# In[21]:


import os
import pandas as pd
os.chdir(r"C:\Users\pc\Desktop\har-ml")  # Raw string
print(os.getcwd())  # Confirm the path change


# In[22]:


print(os.listdir("data"))


# In[32]:


import os
import pandas as pd
file_path = r"C:\Users\pc\Desktop\har-ml\data\UCI_preprocessed.csv"
uci_df = pd.read_csv(file_path)


if os.path.exists(file_path):
    print("File found. Loading dataset...")
    uci_df = pd.read_csv(file_path)
    print("Dataset loaded successfully.")
else:
    print("Error: File not found at", file_path)


# In[33]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Encode labels
label_encoder = LabelEncoder()
uci_df['Activity'] = label_encoder.fit_transform(uci_df['Activity'])

# Prepare data
X = uci_df.iloc[:, :-1]  # Features
y = uci_df['Activity']   # Target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Logistic Regression
lr = LogisticRegression(max_iter=2000)
lr.fit(X_train, y_train)

# Evaluate
y_pred = lr.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


# In[6]:


from sklearn.ensemble import RandomForestClassifier

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluate
y_pred_rf = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))


# In[35]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

# Define parameter grid with fewer values
param_dist = {
    'n_estimators': np.arange(50, 200, 50),
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Use Randomized Search instead of Grid Search
random_search = RandomizedSearchCV(RandomForestClassifier(), param_dist, n_iter=10, cv=3, scoring='accuracy', n_jobs=-1, random_state=42)

# Make sure X_train and y_train are defined before fitting
random_search.fit(X_train, y_train)

# Best model
best_model = random_search.best_estimator_
best_accuracy = random_search.best_score_

print("Best Model Accuracy:", best_accuracy)


# In[37]:


import os  

# Ensure the models directory exists
os.makedirs("../models", exist_ok=True)


# In[38]:


import os
print(os.listdir("models"))


# In[40]:


import joblib

# Save the best model
joblib.dump(best_model, "best_model.pkl")

# Save the scaler and label encoder (ensure these are defined before saving)
joblib.dump(scaler, "../models/scaler.pkl")
joblib.dump(label_encoder, "../models/label_encoder.pkl")

print(f"Best Model Saved! Accuracy: {best_accuracy}")

