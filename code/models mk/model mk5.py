# best performing for now

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # eliminate tensotflow floating point errors due to TensorFlow using oneDNN
                                          # (oneAPI Deep Neural Network Library) custom operations

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from keras import layers
import keras_tuner as kt
import matplotlib.pyplot as plt



# Preprocess the Data
df = pd.read_csv('D:/documenti/uni/data_analysis/tesi/opzione 1/datasets/final.csv')

# Drop the date columns not needed
df = df.drop(columns=['Date', 'Reference Date'])

# Separate the features and the target variable
X = df.drop(columns=['PUN'])
y = df['PUN']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# build neural network

# Define the neural network architecture
model = Sequential([
    Input(shape=(X_train.shape[1],)),  # Use Input layer to specify the input shape
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1)  # Output layer for regression
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)


# Evaluate the model on the test data
loss, mae = model.evaluate(X_test, y_test, verbose=1)
print(f"Mean Absolute Error on test data: {mae}")


# Make predictions on the test data
y_pred = model.predict(X_test)

# Compare the predictions with the actual values
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred.flatten()})
print(results.head())


# Plot training history
def plot_history(history):
    plt.figure(figsize=(12, 6))

    # Plot training & validation MAE values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    plt.title('Model MAE')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')

    plt.tight_layout()
    plt.show()

plot_history(history)




# Example of making predictions on new data

new_data = np.array([[351.92, 98.62, 388, 91.3, 98.7, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 1, 0, 0]])

# print(X_train.shape[1])
# Ensure new_data has the same number of features as X_train
assert new_data.shape[1] == X_train.shape[1], "Number of features in new_data does not match X_train"

# Predict PUN value for the new data
new_data_scaled = scaler.transform(new_data)
predicted_pun = model.predict(new_data_scaled)
print(f"Predicted PUN value: {predicted_pun[0][0]}")