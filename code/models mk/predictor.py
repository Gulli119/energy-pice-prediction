import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # eliminate tensotflow floating point errors due to TensorFlow using oneDNN
                                          # (oneAPI Deep Neural Network Library) custom operations
import pandas as pd
import numpy as np
import random
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import load_model
from joblib import load
from tabulate import tabulate
import matplotlib.pyplot as plt


# pun
# load and prepare model
# Load the saved model
model = load_model('model_mk_6.keras')

# Load the dataset to get the original data for reference (if needed)
df = pd.read_csv('D:/documenti/uni/data_analysis/tesi/opzione 1/datasets/final.csv')

# Drop the date columns not needed
df = df.drop(columns = ['Date', 'Reference Date'])

def pun_pred(new_data):
    # Convert to numpy array and reshape for prediction
    new_data = np.array(new_data).reshape(1, -1)

    # Scale the new data using the same scaler used for training data
    scaler = RobustScaler()
    X_train = df.drop(columns=['PUN Target'])
    scaler.fit(X_train)
    new_data_scaled = scaler.transform(new_data)

    # Make prediction
    predicted_pun = model.predict(new_data_scaled)

    # Inverse transform the prediction
    predicted_pun_value = np.expm1(predicted_pun)

    data = {'PUN': [predicted_pun_value]}  # Replace 42 with your desired value

    predicted_pun_df = pd.DataFrame(data)
    return predicted_pun_df







# numeric values

# load and prepare the models
# Set the path to your models
model_paths = {
    'Gas Price Target': 'D:/documenti/uni/data_analysis/tesi/opzione 1/code/num models/model_Gas Price Target.keras',
    'Petrol Price Target': 'D:/documenti/uni/data_analysis/tesi/opzione 1/code/num models/model_Petrol Price Target.keras',
    'Coal Price Target': 'D:/documenti/uni/data_analysis/tesi/opzione 1/code/num models/model_Coal Price Target.keras',
    'estrazione di petrolio greggio Target': 'D:/documenti/uni/data_analysis/tesi/opzione 1/code/num models/model_estrazione di petrolio greggio Target.keras',
    'estrazione di gas naturale Target': 'D:/documenti/uni/data_analysis/tesi/opzione 1/code/num models/model_estrazione di gas naturale Target.keras'
}

# Load the models
models = {name: load_model(path) for name, path in model_paths.items()}

# Load the dataset to get the original data for reference (if needed)
df1 = pd.read_excel('D:/documenti/uni/data_analysis/tesi/opzione 1/datasets/final1.xlsx')

# Drop the date columns not needed
df1 = df1.drop(columns=['Date Target', 'Date'])



def num_pred(new_data):
    new_data = new_data[1:6]

    # Prepare the feature set (exclude target columns)
    target_columns = [
        'Gas Price Target', 'Petrol Price Target', 'Coal Price Target',
        'estrazione di petrolio greggio Target', 'estrazione di gas naturale Target'
    ]
    X = df1.drop(columns=target_columns)

    # Scale the features using the same scaler used during training
    scaler = RobustScaler()
    scaler.fit(X)  # Fit the scaler on the training data

    # Convert new data to numpy array and reshape for prediction
    new_data_array = np.array(new_data).reshape(1, -1)
    new_data_scaled = scaler.transform(new_data_array)

    # Dictionary to store the predictions
    predictions = {}

    # Predict each value using its corresponding model
    for target, model in models.items():
        prediction = model.predict(new_data_scaled).flatten()[0]
        predictions[target] = prediction

    # Create a dataframe from the predictions dictionary
    predictions_df = pd.DataFrame(list(predictions.items()), columns=['Variable', 'Prediction']).set_index('Variable').T

    return predictions_df




# bool values

# Set the path to your models
model_paths_bool = {
    'inizio guerra usa': 'D:/documenti/uni/data_analysis/tesi/opzione 1/code/bool models/model_inizio guerra usa Target.joblib',
    'inizio guerra iraq': 'D:/documenti/uni/data_analysis/tesi/opzione 1/code/bool models/model_inizio guerra iraq Target.joblib',
    'inizio guerra russia': 'D:/documenti/uni/data_analysis/tesi/opzione 1/code/bool models/model_inizio guerra russia Target.joblib',
    'inizio guerra civile iraq': 'D:/documenti/uni/data_analysis/tesi/opzione 1/code/bool models/model_inizio guerra civile iraq Target.joblib',
    'inizio guerra civile libia': 'D:/documenti/uni/data_analysis/tesi/opzione 1/code/bool models/model_inizio guerra civile libia Target.joblib',
    'avvisaglie guerra russia': 'D:/documenti/uni/data_analysis/tesi/opzione 1/code/bool models/model_avvisaglie guerra russia Target.joblib',
    'sanzioni verso russia': 'D:/documenti/uni/data_analysis/tesi/opzione 1/code/bool models/model_sanzioni verso russia Target.joblib',
    'crisi economica': 'D:/documenti/uni/data_analysis/tesi/opzione 1/code/bool models/model_crisi economica Target.joblib',
    'pandemia': 'D:/documenti/uni/data_analysis/tesi/opzione 1/code/bool models/model_pandemia Target.joblib',
    'lockdown': 'D:/documenti/uni/data_analysis/tesi/opzione 1/code/bool models/model_lockdown Target.joblib',
    'stato di emergenza': 'D:/documenti/uni/data_analysis/tesi/opzione 1/code/bool models/model_stato di emergenza Target.joblib'
}

# Load the models
models_bool = {name: load(path) for name, path in model_paths_bool.items()}

# Load the dataset to get the original data for reference (if needed)
df_bool = pd.read_excel('D:/documenti/uni/data_analysis/tesi/opzione 1/datasets/final1_bool.xlsx')

# Drop the date columns not needed
df_bool = df_bool.drop(columns=['Date Target', 'Date'])

def bool_pred(new_data):
    new_data = new_data[6:]

    # Prepare the feature set (exclude target columns)
    target_columns_bool = [
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
    X_bool = df_bool.drop(columns=target_columns_bool)

    # Scale the features using the same scaler used during training
    scaler_bool = RobustScaler()
    scaler_bool.fit(X_bool)  # Fit the scaler on the training data

    # Convert new data to numpy array and reshape for prediction
    new_data_array_bool = np.array(new_data).reshape(1, -1)
    new_data_scaled_bool = scaler_bool.transform(new_data_array_bool)

    # Dictionary to store the predictions
    predictions_bool = {}

    # Predict each value using its corresponding model
    for target in df_bool.iloc[:, :40].items():
        t_name = str(target[0])
        target_model = t_name.replace(" Target", "")
        column_data = target[1]
        unique_values = column_data.unique()

        if len(unique_values) == 1:  # Check if the column has only one unique value
            prediction_bool = unique_values[0]  # Set prediction to the unique value
        else:
            prediction_bool = models_bool[target_model].predict(new_data_scaled_bool).flatten()[0]

        # Store prediction in the dictionary with column name as key
        predictions_bool[target_model] = prediction_bool

    # Create a dataframe from the predictions dictionary
    predictions_df = pd.DataFrame(predictions_bool, index=[0])

    return predictions_df



# Example of new data point for prediction
new_data = [94.88, 81.83, 76.99, 105.50, 89.5, 84.2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]

# Initialize DataFrame placeholders
predicted_pun_df = pun_pred(new_data)
predictions_num_values = num_pred(new_data)
predictions_bool_values = bool_pred(new_data)

# Concatenate initial predictions into one DataFrame
predicted_pun_df.reset_index(drop=True, inplace=True)
predictions_num_values.reset_index(drop=True, inplace=True)
predictions_bool_values.reset_index(drop=True, inplace=True)
all_df = pd.concat([predicted_pun_df, predictions_num_values, predictions_bool_values], axis=1)

# Flatten and extract numeric values
vector_values = all_df.values.flatten()
vector_values = [item.item() if isinstance(item, np.ndarray) else item for item in vector_values]


a = {
    'PUN': [new_data[0]]
}

# Run the prediction process 10 times
for i in range(100):
    if i > 0:
        # Use vector_values from the previous iteration as new_data
        new_data = vector_values


    # Perform predictions for the current new_data
    predicted_pun_df = pun_pred(new_data)
    predictions_num_values = num_pred(new_data)
    predictions_bool_values = bool_pred(new_data)

    # Concatenate predictions into one DataFrame
    predicted_pun_df.reset_index(drop=True, inplace=True)
    predictions_num_values.reset_index(drop=True, inplace=True)
    predictions_bool_values.reset_index(drop=True, inplace=True)
    all_df = pd.concat([predicted_pun_df, predictions_num_values, predictions_bool_values], axis=1)

    # Flatten and extract numeric values for the next iteration
    vector_values = all_df.values.flatten()
    vector_values = [item.item() if isinstance(item, np.ndarray) else item for item in vector_values]

    # Optional: Print or use vector_values as needed for each iteration
    print(f"Iteration {i + 1}:")
    print(vector_values)
    print()

    a.setdefault('PUN', []).append(vector_values[0])

# After 10 iterations, vector_values will contain the results of the final iteration
print("Final vector_values after 10 iterations:")
print(vector_values)

all_df_final = pd.DataFrame(a)
print(all_df_final)



# Plotting
plt.figure(figsize=(10, 6))  # Optional: Adjust the figure size
plt.plot(all_df_final.index, all_df_final['PUN'], marker='o', linestyle='-', color='b', label='PUN')
plt.xlabel('Index')
plt.ylabel('PUN Values')
plt.title('Plot of PUN Values')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()