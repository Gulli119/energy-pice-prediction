import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Eliminate TensorFlow floating point errors due to TensorFlow using oneDNN

import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import matplotlib.pyplot as plt
from tabulate import tabulate


# Set random seeds for reproducibility
def set_random_seed(seed_value=42):
    np.random.seed(seed_value)
    random.seed(seed_value)
    tf.random.set_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)


set_random_seed()

# Load the dataset
df = pd.read_excel('D:/documenti/uni/data_analysis/tesi/opzione 1/datasets/final1.xlsx')

# Drop the date columns not needed
df = df.drop(columns=['Date Target', 'Date'])

# Target columns to predict individually
target_columns = [
    'Gas Price Target', 'Petrol Price Target', 'Coal Price Target', 'estrazione di petrolio greggio Target',
    'estrazione di gas naturale Target'
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


# Function to build and train the model for a specific target variable
def train_and_predict(target):
    print(f"Training model for {target}...")

    y = df[target]
    y_train, y_test = train_test_split(y, test_size=0.2, random_state=42)

    # Build the neural network
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)  # Output layer for regression
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

    # Evaluate the model on the test data
    loss, mae = model.evaluate(X_test, y_test, verbose=1)
    print(f"Mean Absolute Error on test data for {target}: {mae}")

    # Make predictions on the test data
    y_pred = model.predict(X_test).flatten()

    # Add the predictions to the DataFrame
    predictions[target] = y_test.reset_index(drop=True)
    predictions[f"{target}_Predicted"] = y_pred

    # Plot training history
    plot_history(history, target)

    # Save the model
    model_directory = 'D:/documenti/uni/data_analysis/tesi/opzione 1/code/num models/'
    model_filename = os.path.join(model_directory, f'model_{target}.keras')
    model.save(model_filename)


# Plot training history for a specific target variable
def plot_history(history, target):
    plt.figure(figsize=(12, 6))

    # Plot training & validation MAE values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    plt.title(f'Model MAE for {target}')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'Model Loss for {target}')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')

    plt.tight_layout()
    plt.show()


# Train and predict for each target variable
for target in target_columns:
    train_and_predict(target)

# Calculate differences
for col in target_columns:
    predictions[f"{col}_Difference"] = predictions[f"{col}_Predicted"] - predictions[col]

# Print the predicted values
print(tabulate(predictions[[f"{col}_Predicted" for col in target_columns] + [col for col in target_columns] + [
    f"{col}_Difference" for col in target_columns]], headers='keys', tablefmt='fancy_grid', showindex=False))

# Plot the differences
for col in target_columns:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(predictions.index, predictions[f"{col}_Difference"], color='g', align='center', label=f'{col} Difference')
    ax.set_title(f'Difference between Predicted and Actual {col}')
    ax.set_xlabel('Index or Time')
    ax.set_ylabel('Difference Value')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

# Plot the comparison
for col in target_columns:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(predictions.index, predictions[f"{col}_Predicted"], label=f'Predicted {col}', color='blue', marker='o')
    ax.plot(predictions.index, predictions[col], label=f'Actual {col}', color='red', linestyle='--', marker='x')
    ax.set_title(f'Comparison of Predicted and Actual {col} Values')
    ax.set_xlabel('Index or Time')
    ax.set_ylabel(f'{col} Value')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()
