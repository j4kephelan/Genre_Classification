#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier


import xgboost as xgb
from sklearn.multioutput import MultiOutputClassifier


# In[5]:


# Read in data as DataFrame
df = pd.read_csv('/Users/veronica/Downloads/features_3_sec.csv').drop(columns=['filename', 'length'])

# Extract feature names and store features in X
features = list(df.columns)[:-1]
X = df[features]

# Encode and store labels in y, extract label names
encoder = OneHotEncoder()
y = encoder.fit_transform(df[['label']]).toarray()
labels = encoder.get_feature_names_out(['label'])

# Add encoded labels back into df
encoded_df = pd.DataFrame(y, columns=labels)
df = pd.concat([df, encoded_df], axis=1)
df = df.drop(columns=['label'], inplace=False)

# Partition data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize feature data, fitting to the training data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[6]:


plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=False, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Map')
plt.show()


# In[7]:


xgb_base = xgb.XGBClassifier(objective='binary:logistic')
multilabel_model = MultiOutputClassifier(xgb_base)

multilabel_model.fit(X_train, y_train)
y_pred = multilabel_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[10]:


from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as IMBPipeline

# Apply PCA to reduce dimensions to 95% variance retained
pca = PCA(n_components=0.95)

# Use KD-Tree algorithm for KNN with hyperparameters handled by grid search
knn = KNeighborsClassifier(algorithm='kd_tree', n_jobs=-1)

# Create an imbalanced-learn pipeline with SMOTE, PCA, and KNN
pipeline = IMBPipeline([
    ('smote', SMOTE(random_state=42)),  # Apply SMOTE for oversampling
    ('pca', pca),
    ('knn', knn)
])

param_grid = {
    'knn__n_neighbors': [3, 5, 7],  # Reduced range for n_neighbors
    'knn__weights': ['uniform', 'distance'],
    'knn__metric': ['euclidean', 'manhattan']
}

# Perform cross-validation with GridSearchCV using the pipeline
grid_search = GridSearchCV(pipeline, param_grid, cv=5, verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Evaluate the best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Print classification report and handle zero_division
print(classification_report(y_test, y_pred, zero_division=1))

# Get and display the best parameters
best_params = grid_search.best_params_
print("Best parameters:", best_params)


# In[11]:


from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Reconfigure the pipeline with the best parameters
optimized_pipeline = IMBPipeline([
    ('smote', SMOTE(random_state=42)),  # Handle class imbalance
    ('pca', PCA(n_components=0.95)),    # Dimensionality reduction
    ('knn', KNeighborsClassifier(
        n_neighbors=3, 
        weights='distance', 
        metric='euclidean', 
        algorithm='kd_tree', 
        n_jobs=-1))                     # Optimized KNN with best parameters
])

# Train the model with the entire training dataset
optimized_pipeline.fit(X_train, y_train)

# Predict the test dataset
y_pred = optimized_pipeline.predict(X_test)

# Evaluation metrics
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=1))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()


# In[ ]:




