# data_processing.py
"""
Data processing module for the heart disease prediction project.
Handles loading, preprocessing, and preparing data for the model.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import json
from cryptography.fernet import Fernet

class HeartDataProcessor:
    def __init__(self, config_path='config.json'):
        """Initialize data processor with configuration settings."""
        self.config = self._load_config(config_path)
        self.scaler = StandardScaler()
        
    def _load_config(self, config_path):
        """Load configuration from JSON file."""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
                
        else:
            # Default configuration
            return {
                'test_size': 0.2,
                'random_state': 42,
                'features': [
                    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
                ]
            }
            
    def load_data(self, filepath, encrypted=False, key=None):
        """
        Load data from CSV file, with option for decryption.
        
        Args:
            filepath: Path to the data file
            encrypted: Boolean indicating if the file is encrypted
            key: Encryption key if file is encrypted
            
        Returns:
            Pandas DataFrame with the loaded data
        """
        if encrypted and key:
            # Decrypt file before loading
            with open(filepath, 'rb') as f:
                encrypted_data = f.read()
            
            fernet = Fernet(key)
            decrypted_data = fernet.decrypt(encrypted_data)
            
            # Load from the decrypted data
            from io import StringIO
            return pd.read_csv(StringIO(decrypted_data.decode()))
        else:
            # Load directly if not encrypted
            return pd.read_csv(filepath)
    
    def preprocess_data(self, data):
        """
        Preprocess the data for modeling.
        
        Args:
            data: Pandas DataFrame with raw data
            
        Returns:
            Processed DataFrame ready for modeling
        """
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Handle missing values
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].median())
        
        # Extract features and target
                
        # Get features and target from config
        data_params = self.config.get("data_params", {})
        feature_cols = data_params.get("features", [])
        target_col = data_params.get("target")

        # Extract features safely
        X = df[feature_cols] if all(col in df.columns for col in feature_cols) else None
        y = df[target_col] if target_col in df.columns else None
        
        return X, y
    
    def prepare_data(self, X, y=None, training=True):
        """
        Prepare data for model training or prediction.
        
        Args:
            X: Feature DataFrame
            y: Target variable (optional)
            training: Whether this is for training (True) or prediction (False)
            
        Returns:
            Processed data ready for the model
        """
        # Get data params from config
        data_params = self.config.get('data_params', {})
        test_size = data_params.get('test_size', 0.2)
        random_state = data_params.get('random_state', 42)
        
        # Handle categorical features if present
        X_processed = pd.get_dummies(X, drop_first=True)
        
        # Scale features
        if training:
            X_scaled = self.scaler.fit_transform(X_processed)
            if y is not None:
                # Split into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, 
                    test_size=test_size, 
                    random_state=random_state
                )
                return X_train, X_test, y_train, y_test
            return X_scaled, None, None, None
        else:
            # For prediction, just transform using the pre-fit scaler
            X_scaled = self.scaler.transform(X_processed)
            return X_scaled
    
    def save_scaler(self, filepath):
        """Save the fitted scaler for later use."""
        import joblib
        joblib.dump(self.scaler, filepath)
    
    def load_scaler(self, filepath):
        """Load a previously fitted scaler."""
        import joblib
        self.scaler = joblib.load(filepath)