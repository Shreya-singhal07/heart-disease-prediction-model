# model.py
"""
Heart disease prediction model implementation using logistic regression.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import joblib
import json
import os

class HeartDiseaseModel:
    def __init__(self, config_path='config.json'):
        """Initialize the heart disease prediction model."""
        self.config = self._load_config(config_path)
        self.model = LogisticRegression(
            C=self.config.get('C', 1.0),
            penalty=self.config.get('penalty', 'l2'),
            solver=self.config.get('solver', 'liblinear'),
            max_iter=self.config.get('max_iter', 1000),
            random_state=self.config.get('random_state', 42)
        )
        
    def _load_config(self, config_path):
        """Load model configuration from JSON file."""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            # Default configuration
            return {
                'C': 1.0,
                'penalty': 'l2',
                'solver': 'liblinear',
                'max_iter': 1000,
                'random_state': 42
            }
    
    def train(self, X_train, y_train):
        """
        Train the logistic regression model.
        
        Args:
            X_train: Training features
            y_train: Training target values
            
        Returns:
            Trained model
        """
        self.model.fit(X_train, y_train)
        return self.model
    
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predicted classes (0 or 1)
        """
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Generate probability estimates for heart disease.
        
        Args:
            X: Features to predict on
            
        Returns:
            Probability estimates for each class
        """
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model performance.
        
        Args:
            X_test: Test features
            y_test: Test target values
            
        Returns:
            Dictionary containing evaluation metrics
        """
        y_pred = self.predict(X_test)
        y_prob = self.predict_proba(X_test)[:, 1]
        
        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': range(X_test.shape[1]),
            'importance': np.abs(self.model.coef_[0])
        }).sort_values('importance', ascending=False)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix,
            'auc': auc,
            'feature_importance': feature_importance
        }
    
    def save_model(self, filepath):
        """Save the trained model to a file."""
        joblib.dump(self.model, filepath)
    
    def load_model(self, filepath):
        """Load a previously trained model from a file."""
        self.model = joblib.load(filepath)
        return self.model