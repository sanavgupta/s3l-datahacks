import sqlite3
import pandas as pd
import numpy as np
import joblib

# 1. SETUP
conn = sqlite3.connect('marine_observations.db')
# We need the full history to calculate the Lags for the future
df = pd.read_sql_query("SELECT * FROM observations", conn)
model = joblib.load('trophic_boosted_model.pkl')

# 2. GENERATE FUTURE GRID (Next Month)
# We take the most recent known state to act as our 'Lag' features
latest_data = df.sort_values('time').groupby(['latitude', 'longitude', 'common_name']).tail(1).copy()

# Setup for Jan 2027 (Future Month)
future_month = 1
future_year = 2027

# Prepare the features to match your 'features' list: 
# ['T_degC', 'Salnty', 'month', 'T_lag1', 'S_lag1', 'bio_type_code']
projection_df = latest_data.copy()
projection_df['T_lag1'] = projection_df['T_degC']
projection_df['S_lag1'] = projection_df['Salnty']
projection_df['month'] = future_month
projection_df['year'] = future_year

# Simulate slight environmental change for the forecast
projection_df['T_degC'] += 0.5 
projection_df['Salnty'] += 0.1

# Encode Bio Type just like you did in the class
projection_df['bio_type_code'] = projection_df['biological_type'].astype('category').cat.codes

# 3. PREDICT PROBABILITY
X_future = projection_df[['T_degC', 'Salnty', 'month', 'T_lag1', 'S_lag1', 'bio_type_code']]

# Use predict_proba to get the likelihood of "High Density" (Class 1)
# We'll map this probability back to the 'count' column so the dots render
probs = model.predict_proba(X_future)[:, 1]
projection_df['count'] = probs * 100  # Scale up so dots are visible on the map
projection_df['is_forecast'] = 1
projection_df['time'] = pd.to_datetime('2027-01-01').tz_localize('UTC')

# 4. APPEND TO DB
final_to_save = projection_df[['time', 'latitude', 'longitude', 'common_name', 'count', 
                               'biological_type', 'T_degC', 'Salnty', 'is_forecast']]

final_to_save.to_sql('observations', conn, if_exists='append', index=False)
conn.close()

print("Future Likelihood projections added to database!")
