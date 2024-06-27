import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable TensorFlow using oneDNN to avoid floating-point errors

import pandas as pd
import numpy as np
import random
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate
import joblib

# Set random seeds for reproducibility
def set_random_seed(seed_value=42):
    np.random.seed(seed_value)
    random.seed(seed_value)

set_random_seed()

# Load the dataset
df = pd.read_excel('D:/documenti/uni/data_analysis/tesi/opzione 1/datasets/final1_bool.xlsx')

# Drop the date columns not needed
df = df.drop(columns=['Date Target', 'Date'])

# Target columns to predict individually
target_columns = [
    'inizio guerra usa Target', 'inizio guerra algeria Target', 'inizio guerra nigeria Target',
    'inizio guerra libia Target', 'inizio guerra arabia Target', 'inizio guerra quatar Target',
    'inizio guerra iraq Target', 'inizio guerra azerbaijan Target', 'inizio guerra russia Target',
    'inizio guerra civile usa Target', 'inizio guerra civile algeria Target', 'inizio guerra civile nigeria Target',
    'inizio guerra civile libia Target', 'inizio guerra civile arabia Target', 'inizio guerra civile quatar Target',
    'inizio guerra civile iraq Target', 'inizio guerra civile azerbaijan Target', 'inizio guerra civile russia Target',
    'avvisaglie guerra usa Target', 'avvisaglie guerra algeria Target', 'avvisaglie guerra nigeria Target',
    'avvisaglie guerra libia Target', 'avvisaglie guerra arabia Target', 'avvisaglie guerra quatar Target',
    'avvisaglie guerra iraq Target', 'avvisaglie guerra azerbaijan Target', 'avvisaglie guerra russia Target',
    'sanzioni verso usa Target', 'sanzioni verso algeria Target', 'sanzioni verso nigeria Target',
    'sanzioni verso libia Target', 'sanzioni verso arabia Target', 'sanzioni verso quatar Target',
    'sanzioni verso iraq Target', 'sanzioni verso azerbaijan Target', 'sanzioni verso russia Target',
    'crisi economica Target', 'pandemia Target', 'lockdown Target', 'stato di emergenza Target'
]

# Prepare the feature set
X = df.drop(columns=target_columns)

# Split the dataset into training and testing sets
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# Scale the features
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Placeholder for the predictions
predictions = pd.DataFrame()

# Create a directory to save the models
model_directory = 'D:/documenti/uni/data_analysis/tesi/opzione 1/code/bool models'
if not os.path.exists(model_directory):
    os.makedirs(model_directory)

# Function to build and train the logistic regression model for a specific target variable
def train_and_predict_logistic_regression(target):
    print(f"Training logistic regression model for {target}...")

    y = df[target]
    y_train, y_test = train_test_split(y, test_size=0.2, random_state=42)

    # Check if both classes (0 and 1) are present in the training data
    unique_classes = np.unique(y_train)
    if len(unique_classes) < 2:
        print(f"Skipping {target} because only one class present in the data.")
        predictions[target] = y_test.reset_index(drop=True)
        predictions[f"{target}_Predicted"] = 0
        return

    # Build the logistic regression model
    model = LogisticRegression(max_iter=1000)

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate the model on the test data
    accuracy = model.score(X_test, y_test)
    print(f"Accuracy on test data for {target}: {accuracy}")

    # Make predictions on the test data
    y_pred = model.predict(X_test)

    # Add the predictions to the DataFrame
    predictions[target] = y_test.reset_index(drop=True)
    predictions[f"{target}_Predicted"] = y_pred

    # Save the model
    model_filename = os.path.join(model_directory, f'model_{target}.joblib')
    joblib.dump(model, model_filename)

# Train and predict for each target variable using logistic regression
for target in target_columns:
    train_and_predict_logistic_regression(target)

# Calculate accuracy only for columns with predictions
for col in predictions.columns:
    if col.endswith('_Predicted'):
        target_col = col.replace('_Predicted', '')
        accuracy = np.mean(predictions[target_col] == predictions[col])
        print(f"Accuracy for {target_col}: {accuracy}")

# Print the predicted values for columns that have predictions
cols_to_print = [col for col in predictions.columns if '_Predicted' in col]
if cols_to_print:
    print(tabulate(predictions[cols_to_print + [col for col in target_columns]], headers='keys', tablefmt='fancy_grid', showindex=False))

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.show()

# Plot confusion matrices for each target variable that contains both 0 and 1 in the test set
for col in target_columns:
    if col in predictions.columns and f"{col}_Predicted" in predictions.columns:
        y_true = predictions[col]
        y_pred = predictions[f"{col}_Predicted"]
        if len(np.unique(y_true)) == 2:  # Check if both classes are present
            plot_confusion_matrix(y_true, y_pred, title=f'Confusion Matrix for {col}')
