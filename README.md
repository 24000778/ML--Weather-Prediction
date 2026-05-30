# Implementation of Random Forest Algorithm for Weather Prediction

# SINDHU PRIYA REDDY G
# 212224040319
## AIM:
To write a program to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data using Random Forest Algorithm.

## Problem Statement and Dataset



## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

Load the weather dataset using pandas.

Preprocess the data by handling missing values and sorting by time.

Select features and create lag variables for temperature and PM2.5.

Train Random Forest models to predict temperature and PM2.5 and save the models. 

## Program:
```

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load dataset
df = pd.read_csv("weather-station-eee-block_2024_07_13.csv")
df.columns = df.columns.str.strip()
df['time'] = pd.to_datetime(df['time'], errors='coerce')

print("Original rows:", len(df))

# Only drop if target missing
df = df.dropna(subset=['tem', 'pm2_5'])

# Fill feature columns instead of dropping
df['hum'] = df['hum'].fillna(df['hum'].mean())
df['pressure'] = df['pressure'].fillna(df['pressure'].mean())
df['wind_speed'] = df['wind_speed'].fillna(df['wind_speed'].mean())
df['co2'] = df['co2'].fillna(df['co2'].mean())

# Sort by time
df = df.sort_values('time')

# Create lag features
df['Temp_Lag1'] = df['tem'].shift(1)
df['PM_Lag1'] = df['pm2_5'].shift(1)

# Only remove first row created by shift
df = df.iloc[1:]

print("Rows after preprocessing:", len(df))

# Features
X = df[['hum', 'pressure', 'wind_speed', 'co2',
        'Temp_Lag1', 'PM_Lag1']]

y_temp = df['tem']
y_pm = df['pm2_5']

print("Training samples:", len(X))

# Train models
model_temp = RandomForestRegressor(n_estimators=300, random_state=42)
model_pm = RandomForestRegressor(n_estimators=300, random_state=42)

model_temp.fit(X, y_temp)
model_pm.fit(X, y_pm)

# Save models
joblib.dump(model_temp, "temperature_model.pkl")
joblib.dump(model_pm, "pm25_model.pkl")

print("Models trained and saved successfully!")

```

## Output:


<img width="1384" height="643" alt="image" src="https://github.com/user-attachments/assets/aab049cd-8e3b-405f-a3a9-a8bf6ec0a632" />

<img width="1395" height="502" alt="image" src="https://github.com/user-attachments/assets/d10fc285-346d-4f53-b44a-0b865dfe9ae2" />

<img width="1384" height="619" alt="image" src="https://github.com/user-attachments/assets/eb479869-eba4-48d2-b8d6-879bc79589ed" />


## Result:

The Random Forest model successfully predicted temperature, PM2.5 pollution, and solar radiation using weather sensor data with good accuracy. The system also generated next-step predictions and visual graphs comparing actual vs predicted values and showing feature importance.
