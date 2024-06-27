import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # eliminate tensotflow floating point errors due to TensorFlow using oneDNN
                                          # (oneAPI Deep Neural Network Library) custom operations
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
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



df = pd.read_csv('D:/documenti/uni/data_analysis/tesi/opzione 1/datasets/final.csv')

# Drop the date columns not needed
df = df.drop(columns = ['Date', 'Reference Date'])

# Apply logarithmic transformation to 'PUN'
df['PUN Target'] = np.log1p(df['PUN Target'])

# Separate the features and the target variable
X = df.drop(columns = ['PUN Target'])
y = df['PUN Target']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# build neural network
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

# Evaluate the model on the test data
loss, mae = model.evaluate(X_test, y_test, verbose=1)
print(f"Mean Absolute Error (MAE) on test data: {mae}")

# Make predictions on the test data
y_pred = model.predict(X_test)

# Compute the Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE) on test data: {mse}")

# Compute the R-squared (R²) score
r2 = r2_score(y_test, y_pred)
print(f"R-squared (R²) on test data: {r2}")

# Compare the predictions with the actual values
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred.flatten()})
results['Actual'] = np.expm1(results['Actual'])  # Inverse transformation
results['Predicted'] = np.expm1(results['Predicted'])  # Inverse transformation
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

# Save the model
model.save('model_mk_6.keras')



# Example of making predictions on new data
new_data = df.drop(columns=['PUN Target'])

# Transform new data using the scaler fitted on X_train
new_data_scaled = scaler.transform(new_data)

# Predict PUN values for the new data
predicted_pun = model.predict(new_data_scaled)

# Create a DataFrame to store the predicted PUN values
predicted_df = pd.DataFrame(predicted_pun, columns=['Predicted PUN'])
predicted_df['PUN Target'] = df['PUN Target'].values  # Add the transformed 'PUN' values
predicted_df['Actual PUN'] = np.expm1(predicted_df['PUN Target']) / 1000  # Inverse transformation
predicted_df['Predicted PUN'] = np.expm1(predicted_df['Predicted PUN']) /1000  # Inverse transformation
predicted_df['Difference'] = predicted_df['Predicted PUN'] - predicted_df['Actual PUN']

print(predicted_df['Difference'].mean() * 1000)
# Print the predicted values
#print(tabulate(predicted_df[['Predicted PUN', 'Actual PUN', 'Difference']], headers='keys', tablefmt='fancy_grid', showindex=False))


# eample of testing with todays data (23/06/2024)

new_data = [94.88, 81.83, 76.99, 105.50, 89.5, 84.2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]

new_data = np.array(new_data).reshape(1, -1)

new_data_scaled = scaler.transform(new_data)

predicted_pun = model.predict(new_data_scaled)

print("predicted value: ", np.expm1(predicted_pun) / 1000)
print("actual value: ", 102.3 / 1000)
print("difference: ", (np.expm1(predicted_pun) - 102.3) / 1000)





# Create figure and axis objects
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the difference
ax.bar(predicted_df.index, predicted_df['Difference'], color='g', align='center', label='Difference')

# Set labels and title
ax.set_title('Difference between Predicted and Actual PUN')
ax.set_xlabel('Index or Time')
ax.set_ylabel('Difference Value')
ax.legend()
ax.grid(True)
plt.tight_layout()

# Show plot
plt.show()


# Create figure and axis objects for the PUN comparison plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot Predicted PUN and PUN Target
ax.plot(predicted_df.index, predicted_df['Predicted PUN'], label='Predicted PUN', color='blue', marker='o')
ax.plot(predicted_df.index, predicted_df['Actual PUN'], label='Actual PUN', color='red', linestyle='--', marker='x')

# Set labels and title for the PUN comparison plot
ax.set_title('Comparison of Predicted and Actual PUN Values')
ax.set_xlabel('Index or Time')
ax.set_ylabel('PUN Value')
ax.legend()
ax.grid(True)
plt.tight_layout()

# Show PUN comparison plot
plt.show()