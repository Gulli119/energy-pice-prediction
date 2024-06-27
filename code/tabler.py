import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate



# DATASET LOADING
gas_ds = pd.read_csv('D:/documenti/uni/data_analysis/tesi/opzione 1/datasets/Future_Gas_naturale_Dati_Storici.csv')
print("gas", tabulate(gas_ds, headers='keys', tablefmt='fancy_grid', showindex=False))
# 2000 - 05/2024

petrol_ds = pd.read_csv('D:/documenti/uni/data_analysis/tesi/opzione 1/datasets/Future_Petrolio_Greggio_Dati_Storici.csv')
print("petrolio", tabulate(petrol_ds, headers='keys', tablefmt='fancy_grid', showindex=False))
# 2000 - 05/2024

coal_ds = pd.read_csv('D:/documenti/uni/data_analysis/tesi/opzione 1/datasets/Coal_Historical_Data.csv')
print("carbone", tabulate(coal_ds, headers='keys', tablefmt='fancy_grid', showindex=False))
# 08/2006 - 05/2024

fonti_interne_ds = pd.read_excel('D:/documenti/uni/data_analysis/tesi/opzione 1/datasets/produzione_interna_gas_e_petolio.xlsx')
print("fonti fossili interne", tabulate(fonti_interne_ds, headers='keys', tablefmt='fancy_grid', showindex=False))
# 08/2006 - 04/2024

#rinnovabili_ds = pd.read_excel('D:/documenti/uni/data_analysis/tesi/opzione 1/datasets/rinnovabili_e_biomasse_italia.xlsx')
#print(rinnovabili_ds)
# 2000 - 2017  (in anni)   PROBLEMA

pun_ds = pd.read_excel('D:/documenti/uni/data_analysis/tesi/opzione 1/datasets/pun.xlsx')
print("pun", tabulate(pun_ds, headers='keys', tablefmt='fancy_grid', showindex=False))
# 08/2006 - 04/2024

pun_target_ds = pd.read_excel('D:/documenti/uni/data_analysis/tesi/opzione 1/datasets/pun_target.xlsx')
print("pun", tabulate(pun_ds, headers='keys', tablefmt='fancy_grid', showindex=False))
# 09/2006 - 05/2024

bool_ds = pd.read_excel('D:/documenti/uni/data_analysis/tesi/opzione 1/datasets/boleani.xlsx')
print("bool", tabulate(bool_ds, headers='keys', tablefmt='fancy_grid'))
# 08/2006 - 04/2024


# TRANSFORMATIONS AND ADJUSTMENTS


# gas
# Select only 'Date' and 'Price' columns
gas_ds = gas_ds[['Date', 'Price']]

# Convert 'Date' to datetime format and then to 'YYYY/MM'
gas_ds['Date'] = pd.to_datetime(gas_ds['Date'], format='%m/%d/%Y')
gas_ds['Date'] = gas_ds['Date'].dt.strftime('%Y/%m')

# Filter rows between '08/2006' and '04/2024'
gas_ds = gas_ds[(gas_ds['Date'] >= '2006/08') & (gas_ds['Date'] <= '2024/04')]

print(tabulate(gas_ds, headers='keys', tablefmt='fancy_grid', showindex=False))



# petrol
# Select only 'Date' and 'Price' columns
petrol_ds = petrol_ds[['Date', 'Price']]

# Convert 'Date' to datetime format and then to 'YYYY/MM'
petrol_ds['Date'] = pd.to_datetime(petrol_ds['Date'], format='%m/%d/%Y')
petrol_ds['Date'] = petrol_ds['Date'].dt.strftime('%Y/%m')

# Filter rows between '08/2006' and '04/2024'
petrol_ds = petrol_ds[(petrol_ds['Date'] >= '2006/08') & (petrol_ds['Date'] <= '2024/04')]

print(tabulate(petrol_ds, headers='keys', tablefmt='fancy_grid', showindex=False))




# coal
# Select only 'Date' and 'Price' columns
coal_ds = coal_ds[['Date', 'Price']]

# Convert 'Date' to datetime format and then to 'YYYY/MM'
coal_ds['Date'] = pd.to_datetime(coal_ds['Date'], format='%m/%d/%Y')
coal_ds['Date'] = coal_ds['Date'].dt.strftime('%Y/%m')

# Filter rows between '08/2006' and '04/2024'
coal_ds = coal_ds[(coal_ds['Date'] >= '2006/08') & (coal_ds['Date'] <= '2024/04')]

print(tabulate(coal_ds, headers='keys', tablefmt='fancy_grid', showindex=False))


# fonti interne non rinnovabili

# Convert 'Periodo' to datetime format
fonti_interne_ds['Periodo'] = pd.to_datetime(fonti_interne_ds['Periodo'])

# Reformat 'Periodo' to 'YYYY/MM'
fonti_interne_ds['Periodo'] = fonti_interne_ds['Periodo'].dt.strftime('%Y/%m')
print(tabulate(fonti_interne_ds, headers='keys', tablefmt='fancy_grid', showindex=False))


# pun
# Convert 'Periodo' to datetime format
pun_ds['mese/anno'] = pd.to_datetime(pun_ds['mese/anno'])

# Reformat 'Periodo' to 'YYYY/MM'
pun_ds['mese/anno'] = pun_ds['mese/anno'].dt.strftime('%Y/%m')
print(tabulate(pun_ds, headers='keys', tablefmt='fancy_grid', showindex=False))



# pun target
# Convert 'Periodo' to datetime format
pun_target_ds['mese/anno'] = pd.to_datetime(pun_target_ds['mese/anno'])

# Reformat 'Periodo' to 'YYYY/MM'
pun_target_ds['mese/anno'] = pun_target_ds['mese/anno'].dt.strftime('%Y/%m')
print(tabulate(pun_target_ds, headers='keys', tablefmt='fancy_grid', showindex=False))





# bool
# Convert 'Periodo' to datetime format
bool_ds['mese/anno'] = pd.to_datetime(bool_ds['mese/anno'])

# Reformat 'Periodo' to 'YYYY/MM'
bool_ds['mese/anno'] = bool_ds['mese/anno'].dt.strftime('%Y/%m')
print(tabulate(bool_ds, headers='keys', tablefmt='fancy_grid', showindex=False))




# create the final dataset
gas_ds = gas_ds[['Price']]
gas_ds.rename(columns={'Price': 'Gas Price'}, inplace=True)

petrol_ds = petrol_ds[['Price']]
petrol_ds.rename(columns={'Price': 'Petrol Price'}, inplace=True)

coal_ds = coal_ds[['Price']]
coal_ds.rename(columns={'Price': 'Coal Price'}, inplace=True)

fonti_interne_ds = fonti_interne_ds[['estrazione di petrolio greggio', 'estrazione di gas naturale']]

bool_ds = bool_ds.drop(columns=['mese/anno'])

pun_ds.rename(columns={'mese/anno': 'Date', 'media': 'PUN'}, inplace=True)

pun_target_ds.rename(columns={'mese/anno': 'Reference Date', 'media': 'PUN Target'}, inplace=True)



# Reset the index of each DataFrame to ignore the original indices
pun_ds.reset_index(drop=True, inplace=True)
pun_target_ds.reset_index(drop=True, inplace=True)
gas_ds.reset_index(drop=True, inplace=True)
petrol_ds.reset_index(drop=True, inplace=True)
coal_ds.reset_index(drop=True, inplace=True)
fonti_interne_ds.reset_index(drop=True, inplace=True)
bool_ds.reset_index(drop=True, inplace=True)

# Concatenate the DataFrames side by side
final_df = pd.concat([pun_target_ds, pun_ds, gas_ds, petrol_ds, coal_ds, fonti_interne_ds, bool_ds], axis=1)

# Save the combined DataFrame to a new CSV file (optional)
final_df.to_csv('D:/documenti/uni/data_analysis/tesi/opzione 1/datasets/final.csv', index=False)
final_df.to_excel('D:/documenti/uni/data_analysis/tesi/opzione 1/datasets/final.xlsx', index=False)
print(tabulate(final_df, headers='keys', tablefmt='fancy_grid', showindex=False))
print(final_df.columns)