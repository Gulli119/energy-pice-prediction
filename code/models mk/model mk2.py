import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # eliminate TensorFlow floating point errors due to oneDNN

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from keras import layers
import keras_tuner as kt
import matplotlib.pyplot as plt
from tabulate import tabulate

# Define the function to build the model
def build_model(hp):
    model = tf.keras.Sequential()  # Change here: use tf.keras instead of keras

    # Tune the number of layers.
    for i in range(hp.Int('num_layers', 1, 4)):
        model.add(layers.Dense(
            units=hp.Int('units_' + str(i),
                         min_value=32,
                         max_value=512,
                         step=32),
            activation='relu'))

    model.add(layers.Dense(1))  # Output layer

    # Tune the learning rate for the optimizer.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(  # Change here: use tf.keras.optimizers.Adam instead of keras.optimizers.Adam
            hp.Float('learning_rate', 1e-4, 1e-2, sampling='LOG')),
        loss='mse',
        metrics=['mae'])

    return model


# Preprocess the Data
df = pd.read_csv('D:/documenti/uni/data_analysis/tesi/opzione 1/datasets/final.csv')

# Drop the date columns not needed
df = df.drop(columns=['Date', 'Reference Date'])

# Separate the features and the target variable
X = df.drop(columns=['PUN Target'])
y = df['PUN Target']

# Split the dataset into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Scale the features
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Initialize the Hyperband tuner
tuner = kt.Hyperband(
    build_model,
    objective='val_mae',
    max_epochs=10,
    factor=3,
    directory='my_dir',
    project_name='intro_to_kt')

# Define the early stopping callback
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# Search for the best hyperparameters
tuner.search(X_train, y_train, epochs=50, validation_data=(X_val, y_val), callbacks=[stop_early])

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of layers is {best_hps.get('num_layers')} and the optimal learning rate for the optimizer is {best_hps.get('learning_rate')}.
""")

# Build the model with the optimal hyperparameters and train it.
model = tuner.hypermodel.build(best_hps)
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val), callbacks=[stop_early])

# Evaluate the model on the test set
eval_result = model.evaluate(X_test, y_test)
print(f"Test Loss: {eval_result[0]}, Test MAE: {eval_result[1]}")

# Make predictions on the test data
y_pred = model.predict(X_test)

# Compare the predictions with the actual values
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred.flatten()})
print(results.head())








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

# Print the predicted values
print(tabulate(predicted_df[['Predicted PUN', 'Actual PUN', 'Difference']], headers='keys', tablefmt='fancy_grid', showindex=False))


# eample of testing with todays data

new_data = [94.88, 81.83, 76.99, 105.50, 89.5, 84.2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]

new_data = np.array(new_data).reshape(1, -1)

new_data_scaled = scaler.transform(new_data)

predicted_pun = model.predict(new_data_scaled)

print("predicted value: ", np.expm1(predicted_pun), "actual value: ", 102.3)





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