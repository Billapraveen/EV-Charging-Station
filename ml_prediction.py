import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb

def prepare_data(sessions_df):
    """
    Extract features and target for ML models.
    Target: energy_kwh
    Features: cyclic time features, historical averages, claimed status.
    """
    df = sessions_df.copy()
    
    # Target variable
    y = df['energy_kwh'].values
    
    # Extract features
    features = [
        'claimed',
        'hist_mean_dur',
        'hist_mean_energy',
        'hist_mean_dep_hour',
        'hour_sin',
        'hour_cos',
        'dow_sin',
        'dow_cos'
    ]
    
    # Handle any potential missing values (though data_loader handles most)
    df[features] = df[features].fillna(-1.0)
    
    X = df[features].values
    
    return X, y, features

def train_and_evaluate_models(X, y, feature_names):
    """
    Train RF, XGBoost, and SVM. Return metrics and feature importances.
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale data (important for SVM)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1),
        'SVM': SVR(kernel='rbf', C=1.0, epsilon=0.1)
    }
    
    for name, model in models.items():
        # Train
        if name == 'SVM':
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
        # Evaluate
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        
        # Feature Importance (not available for SVR rbf)
        importance = None
        if name in ['Random Forest', 'XGBoost']:
            importance = dict(zip(feature_names, model.feature_importances_))
            
        results[name] = {
            'RMSE': rmse,
            'MAE': mae,
            'importance': importance,
            'model': model
        }
        
    return results

def run_ml_pipeline(sessions_df):
    """Main entry point for ML prediction from the dashboard."""
    if len(sessions_df) < 50:
        return None # Not enough data
        
    X, y, feature_names = prepare_data(sessions_df)
    results = train_and_evaluate_models(X, y, feature_names)
    return results

if __name__ == "__main__":
    from simulator import make_synthetic_sessions
    print("Generating synthetic data for ML test...")
    df, _ = make_synthetic_sessions(n=1000, seed=42)
    
    print("Running ML Pipeline...")
    results = run_ml_pipeline(df)
    
    for name, metrics in results.items():
        print(f"\n--- {name} ---")
        print(f"RMSE: {metrics['RMSE']:.4f}")
        print(f"MAE:  {metrics['MAE']:.4f}")
        if metrics['importance']:
            print("Feature Importances:")
            sorted_imp = sorted(metrics['importance'].items(), key=lambda x: x[1], reverse=True)
            for f, imp in sorted_imp:
                print(f"  {f}: {imp:.4f}")
