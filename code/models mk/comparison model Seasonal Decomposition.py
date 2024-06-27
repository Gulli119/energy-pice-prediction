import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Load the dataset
df = pd.read_csv('D:/documenti/uni/data_analysis/tesi/opzione 1/datasets/final.csv')

# Drop the date columns not needed
df = df.drop(columns=['Date', 'Reference Date'])

# Extract the target variable
y_actual = df['PUN Target']

# Decompose the time series (assuming monthly frequency)
decomposition = seasonal_decompose(y_actual, model='additive', period=12)

# Predictions using seasonal component
trend = decomposition.trend
seasonal = decomposition.seasonal
y_pred_all = trend + seasonal

# Convert predictions to a DataFrame for easier handling
y_pred_df = pd.DataFrame(y_pred_all, columns=['Predicted PUN Target'])

# Plot the actual vs predicted PUN Target
plt.figure(figsize=(10, 6))
plt.plot(y_actual.values, label='Actual PUN', color='red', linestyle='--')
plt.plot(y_pred_df.values, label='Predicted PUN', color='blue')
plt.xlabel('Index')
plt.ylabel('PUN Target')
plt.title('Actual vs Predicted PUN (Seasonal Decomposition)')
plt.legend()
plt.show()

# Calculate differences between actual and predicted PUN
differences = y_actual.values - y_pred_df.values.flatten()

# Create a bar plot for the differences
plt.figure(figsize=(10, 6))
plt.bar(range(len(differences)), differences, color='green')
plt.xlabel('Index')
plt.ylabel('Difference (Actual - Predicted)')
plt.title('Difference between Actual and Predicted PUN (Seasonal Decomposition)')
plt.show()



