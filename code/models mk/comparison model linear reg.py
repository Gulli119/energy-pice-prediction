import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('D:/documenti/uni/data_analysis/tesi/opzione 1/datasets/final.csv')

# Drop the date columns not needed
df = df.drop(columns=['Date', 'Reference Date'])

# Feature selection and target variable
X = df.drop(['PUN Target'], axis=1)
y = df['PUN Target']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Predicting the PUN Target for all data
# Separate features and target
X_all = df.drop(columns=['PUN Target'])
y_actual = df['PUN Target']

# Scale the features
X_scaled = scaler.transform(X_all)

# Predict using the trained model
y_pred_all = model.predict(X_scaled)

# Convert predictions to a DataFrame for easier handling
y_pred_df = pd.DataFrame(y_pred_all, columns=['Predicted PUN Target'])

# Plot the actual vs predicted PUN Target
plt.figure(figsize=(10, 6))
plt.plot(y_actual.values, label='Actual PUN', color='red', linestyle='--')
plt.plot(y_pred_df.values, label='Predicted PUN', color='blue')
plt.xlabel('Index')
plt.ylabel('PUN Target')
plt.title('Actual vs Predicted PUN')
plt.legend()
plt.show()

# Calculate differences between actual and predicted PUN
differences = y_actual.values - y_pred_df.values.flatten()

# Create a bar plot for the differences
plt.figure(figsize=(10, 6))
plt.bar(range(len(differences)), differences, color='green')
plt.xlabel('Index')
plt.ylabel('Difference (Actual - Predicted)')
plt.title('Difference between Actual and Predicted PUN')
plt.show()
