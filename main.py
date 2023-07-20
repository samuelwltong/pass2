#!/usr/bin/env python
# coding: utf-8

import warnings
import sys
import preprocess as PA
import MLP as MA

warnings.filterwarnings("ignore")

# Get values passed as arguments
split_ratio = float(sys.argv[1])
rand_state = int(sys.argv[2])

# Do something with the values
print(f"You entered the numbers: {split_ratio} and {rand_state}")


# ---------------------------Step 1: Processing data-------------------------------

# Get data
df = PA.get_data(db_name = 'failure.db', tbl_name = 'failure')

# Feature engineering 1: Model: Separate year from model and create new feature
df[['Model', 'Model_Year']] = df['Model'].str.split(',', expand=True)

# Feature engineering 2.1: Temperature: Convert Celsius to Fahrenheit for Factory in
# 'New York, U.S'
condition_1 = df['Temperature'].str[-2:] == '°C'
condition_2 = df['Factory'] == 'New York, U.S'
NY_F = df.loc[condition_1 & condition_2, 'Temperature'].str[:-1] + 'F'
df.loc[condition_1 & condition_2, 'Temperature'] = NY_F

# Feature engineering 2.2: Temperature: Convert Fahrenheit to Celsius, and convert data type
# to float
df['Temperature'] = df['Temperature'].apply(PA.convert_temperature)

# Feature engineering 3: RPM: Absolute the numbers to convert negative values to positive
df['RPM'] = PA.data_wrangling(df=df, col_='RPM', method_='absolute')

# Feature engineering 4: Factory: Separate country from state
df[['Factory_State', 'Factory_Country']] = df['Factory'].str.split(', ', expand=True)

# Drop original 'Factory' feature
df = df.drop(columns=['Factory'])

# Drop duplicates
df = df.drop_duplicates()

# Feature engineering 5: Membership: Fill missing values
df['Membership'] = df['Membership'].fillna('Normal')

# Feature engineering 6: Factory: Correction of errors
df['Factory_State'] = df['Factory_State'].replace('Seng Kang', 'Shang Hai', regex=True)
df['Factory_State'] = df['Factory_State'].replace('Bedok', 'Berlin', regex=True)
df['Factory_State'] = df['Factory_State'].replace('Newton', 'New York', regex=True)
df.loc[(df['Factory_Country'] == "China") & (df['Factory_State'] == "New York"), 'Factory_Country'] = 'U.S'

# Feature engineering 7: Data transformation 
df['Temperature_reci'] = PA.data_transform(df=df, col_='Temperature', method_='reci')
df['RPM_bc'] = PA.data_transform(df=df, col_='RPM', method_='boc_cox')
df['Fuel consumption_bc'] = PA.data_transform(df=df, col_='Fuel consumption', method_='boc_cox')

# ---------------------------Step 2: Machine Learning Pipeline-------------------------------

# Define features and target variables to be used in pipeline
x_col_list = ['Temperature_reci', 'RPM_bc', 'Fuel consumption_bc',
                'Model', 'Color', 'Membership',
                'Model_Year', 'Factory_State']
y_col_name = ['Failure A', 'Failure B', 'Failure C', 'Failure D', 'Failure E']

results_df = MA.ML_pipeline(split_ratio=split_ratio, rand_state=rand_state,
                            df=df, X_names_list=x_col_list,
                            Y_name=y_col_name)

print('\nSuccessful!')
