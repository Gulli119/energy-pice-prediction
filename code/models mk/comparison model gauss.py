import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('D:/documenti/uni/data_analysis/tesi/opzione 1/datasets/final.csv')

column_data = df['PUN Target']

# Calculate mean and standard deviation
mu = np.mean(column_data)
sigma = np.std(column_data)

# Standardize data
standardized_data = (column_data - mu) / sigma

a = []

for i in range(len(standardized_data)):
    # Generate random value from standard normal distribution
    random_value = np.random.randn()

    # Transform random value to original distribution
    value_from_gaussian = random_value * sigma + mu

    a.append(value_from_gaussian)

# Create a DataFrame from list a with a single column named "predicted pun"
data = pd.DataFrame(a, columns=["Predicted PUN"])

data = pd.concat([data, df['PUN Target']], axis=1)

# Create figure and axis objects for the PUN comparison plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot Predicted PUN and PUN Target
ax.plot(data.index, data['Predicted PUN'], label='Predicted PUN', color='blue')
ax.plot(data.index, data['PUN Target'], label='PUN Target', color='red', linestyle='--')

# Set labels and title for the PUN comparison plot
ax.set_title('Comparison of Predicted and Actual PUN Values')
ax.set_xlabel('Index or Time')
ax.set_ylabel('PUN Value')
ax.legend()
plt.tight_layout()

# Show PUN comparison plot
plt.show()
