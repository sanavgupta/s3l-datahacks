import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrophicPredictorWinner:
    def __init__(self, db_path='marine_observations.db', model_path='trophic_boosted_model.pkl'):
        self.db_path = db_path
        self.model_path = model_path
        # Added biological_type_code to features
        self.features = ['T_degC', 'Salnty', 'month', 'T_lag1', 'S_lag1', 'bio_type_code']

    def load_and_engineer_features(self):
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM observations", conn)
        conn.close()

        # 1. Categorical Encoding for Biological Type
        # This tells the AI: "This rule is for fish, this rule is for plankton"
        df['bio_type_code'] = df['biological_type'].astype('category').cat.codes

        # 2. Sort and Create Lags
        df = df.sort_values(['latitude', 'longitude', 'year', 'month'])
        df['T_lag1'] = df.groupby(['latitude', 'longitude'])['T_degC'].shift(1)
        df['S_lag1'] = df.groupby(['latitude', 'longitude'])['Salnty'].shift(1)

        # 3. Clean and Threshold
        df = df.dropna(subset=self.features + ['count'])
        threshold = df['count'].median()
        df['density_level'] = (df['count'] > threshold).astype(int)
        
        return df

    def train_and_evaluate(self):
        df = self.load_and_engineer_features()
        
        # 4. Chronological Split
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        X_train, y_train = train_df[self.features], train_df['density_level']
        X_test, y_test = test_df[self.features], test_df['density_level']

        # 5. THE WINNING CONFIGURATION
        # categorical_features=[2, 5] -> Month (idx 2) and Bio Type (idx 5)
        # class_weight='balanced' -> Forces model to care about Class 0 (Low Density)
        model = HistGradientBoostingClassifier(
            max_iter=150, 
            learning_rate=0.08, 
            categorical_features=[2, 5], 
            class_weight='balanced',
            random_state=42
        )
        
        logger.info("Training Optimized Model with Class Balancing...")
        model.fit(X_train, y_train)

        # Add this after model.fit()
        import matplotlib.pyplot as plt
        
        # HistGradientBoosting uses permutation importance for its "story"
        from sklearn.inspection import permutation_importance
        
        r = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
        
        print("\n--- What drives the Trophic Cascade? ---")
        for i in r.importances_mean.argsort()[::-1]:
            print(f"{self.features[i]}: {r.importances_mean[i]:.3f} +/- {r.importances_std[i]:.3f}")

        # 6. Evaluation
        y_pred = model.predict(X_test)
        print(f"\n--- Optimized Performance ---")
        print(classification_report(y_test, y_pred))
        
        # 7. Save for Dashboard
        joblib.dump(model, self.model_path)
        return model
        
        # --- ADD THIS TO THE BOTTOM OF train_and_evaluate ---
        from sklearn.inspection import permutation_importance
        
        logger.info("Calculating Feature Importance (this may take a minute)...")
        r = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
        
        print("\n--- WHAT DRIVES THE MARINE ECOSYSTEM? ---")
        # Sort features by their importance
        sorted_idx = r.importances_mean.argsort()[::-1]
        
        for i in sorted_idx:
            print(f"{self.features[i]:<15}: {r.importances_mean[i]:.3f} +/- {r.importances_std[i]:.3f}")

if __name__ == "__main__":
    predictor = TrophicPredictorWinner()
    predictor.train_and_evaluate()
