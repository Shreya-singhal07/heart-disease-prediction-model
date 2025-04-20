# evaluation.py
"""
Model evaluation and visualization module for heart disease prediction.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.metrics import confusion_matrix, classification_report
import json
import os
from datetime import datetime

class ModelEvaluator:
    def __init__(self, output_dir='evaluation_results'):
        """Initialize model evaluator with output directory."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def evaluate_model(self, model, X_test, y_test, feature_names=None):
        """
        Evaluate the model and generate performance metrics.
        
        Args:
            model: Trained model object
            X_test: Test features
            y_test: Test target values
            feature_names: Optional list of feature names
            
        Returns:
            Dictionary with evaluation results
        """
        # Get predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate performance metrics
        results = {}
        
        # Basic metrics
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        results['accuracy'] = (tp + tn) / (tp + tn + fp + fn)
        results['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        results['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        results['f1_score'] = 2 * results['precision'] * results['recall'] / (
            results['precision'] + results['recall']) if (results['precision'] + results['recall']) > 0 else 0
        results['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # ROC and AUC
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        results['auc'] = auc(fpr, tpr)
        results['roc_curve'] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}
        
        # Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        results['pr_curve'] = {'precision': precision.tolist(), 'recall': recall.tolist()}
        
        # Classification report
        clf_report = classification_report(y_test, y_pred, output_dict=True)
        results['classification_report'] = clf_report
        
        # Feature importance (if available)
        if hasattr(model, 'coef_') and feature_names is not None:
            coef = model.coef_[0]
            importance = pd.DataFrame({
                'feature': feature_names,
                'importance': np.abs(coef)
            }).sort_values('importance', ascending=False)
            results['feature_importance'] = importance.to_dict('records')
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(os.path.join(self.output_dir, f'evaluation_results_{timestamp}.json'), 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json.dump(results, f, indent=4)
        
        return results
    
    def plot_roc_curve(self, results, filename=None):
        """
        Plot ROC curve from evaluation results.
        
        Args:
            results: Dictionary with evaluation results
            filename: Optional filename to save the plot
            
        Returns:
            Figure object
        """
        plt.figure(figsize=(10, 8))
        plt.plot(
            results['roc_curve']['fpr'], 
            results['roc_curve']['tpr'], 
            color='blue', lw=2, 
            label=f'ROC curve (AUC = {results["auc"]:.3f})'
        )
        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        
        if filename:
            plt.savefig(os.path.join(self.output_dir, filename))
        
        return plt.gcf()
    
    def plot_confusion_matrix(self, results, filename=None):
        """
        Plot confusion matrix from evaluation results.
        
        Args:
            results: Dictionary with evaluation results
            filename: Optional filename to save the plot
            
        Returns:
            Figure object
        """
        # Extract confusion matrix from classification report
        cm = np.array([
            [results['classification_report']['0']['support'] * (1 - results['classification_report']['0']['recall']),
             results['classification_report']['0']['support'] * results['classification_report']['0']['recall']],
            [results['classification_report']['1']['support'] * (1 - results['classification_report']['1']['recall']),
             results['classification_report']['1']['support'] * results['classification_report']['1']['recall']]
        ])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues',
                    xticklabels=['No Disease', 'Heart Disease'],
                    yticklabels=['No Disease', 'Heart Disease'])
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        
        if filename:
            plt.savefig(os.path.join(self.output_dir, filename))
        
        return plt.gcf()
    
    def plot_feature_importance(self, results, top_n=10, filename=None):
        """
        Plot feature importance from evaluation results.
        
        Args:
            results: Dictionary with evaluation results
            top_n: Number of top features to display
            filename: Optional filename to save the plot
            
        Returns:
            Figure object
        """
        if 'feature_importance' not in results:
            return None
        
        # Convert feature importance dict to DataFrame if needed
        if isinstance(results['feature_importance'], list):
            importance_df = pd.DataFrame(results['feature_importance'])
        else:
            importance_df = results['feature_importance']
        
        # Select top N features
        importance_df = importance_df.sort_values('importance', ascending=False).head(top_n)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=importance_df)
        plt.title(f'Top {top_n} Feature Importance')
        plt.xlabel('Absolute Coefficient Value')
        plt.ylabel('Feature')
        
        if filename:
            plt.savefig(os.path.join(self.output_dir, filename))
        
        return plt.gcf()