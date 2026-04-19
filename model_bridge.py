import joblib
import pandas as pd
import sqlite3

class DashboardAI:
    def __init__(self, model_path='trophic_boosted_model.pkl', db_path='marine_observations.db'):
        self.model = joblib.load(model_path)
        self.db_path = db_path
        self.features = ['T_degC', 'Salnty', 'month', 'T_lag1', 'S_lag1', 'bio_type_code']

    def get_prediction(self, lat, lon, month, bio_type='larval_fish'):
        """Helper for the frontend to get an AI prediction for a specific map point."""
        # 1. Look up the most recent physical data for this spot to get 'lags'
        conn = sqlite3.connect(self.db_path)
        query = f"SELECT T_degC, Salnty FROM observations WHERE latitude={round(lat,2)} AND longitude={round(lon,2)} LIMIT 1"
        recent = pd.read_sql_query(query, conn)
        conn.close()

        if recent.empty:
            return {"error": "No historical data for this coordinate"}

        # 2. Prepare the input data
        input_data = pd.DataFrame([{
            'T_degC': recent['T_degC'].iloc[0],
            'Salnty': recent['Salnty'].iloc[0],
            'month': month,
            'T_lag1': recent['T_degC'].iloc[0], # Using recent as proxy for lag
            'S_lag1': recent['Salnty'].iloc[0],
            'bio_type_code': 0 if bio_type == 'larval_fish' else 1
        }])

        # 3. Predict Probability
        prob = self.model.predict_proba(input_data[self.features])[0][1]
        
        return {
            "prediction": "High Density" if prob > 0.5 else "Low Density",
            "confidence": f"{round(prob * 100, 1)}%",
            "insight": "High probability of biological hotspot" if prob > 0.8 else "Stable conditions"
        }
