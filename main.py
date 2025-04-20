# main.py
"""
Main module for the heart disease prediction system.
This script orchestrates the overall workflow.
"""

import os
import argparse
import json
import logging
from datetime import datetime

# Import project modules
from data_processing import HeartDataProcessor
from model import HeartDiseaseModel
from security import HealthDataSecurity
from evaluation import ModelEvaluator

def setup_logging():
    """Set up logging configuration."""
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f'app_{datetime.now().strftime("%Y%m%d")}.log')
    
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Also log to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    
    return logging.getLogger('heart_disease_prediction')

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Heart Disease Prediction System')
    
    parser.add_argument('--data', type=str, default='data/heart.csv',
                        help='Path to the heart disease dataset')
    parser.add_argument('--encrypted', action='store_true',
                        help='Indicate if the data file is encrypted')
    parser.add_argument('--key', type=str, default=None,
                        help='Path to encryption key file if data is encrypted')
    parser.add_argument('--config', type=str, default='config.json',
                        help='Path to configuration file')
    parser.add_argument('--output', type=str, default='output',
                        help='Directory for output files')
    parser.add_argument('--mode', choices=['train', 'predict', 'evaluate'], default='train',
                        help='Operation mode: train, predict, or evaluate')
    parser.add_argument('--model_path', type=str, default='models/heart_disease_model.pkl',
                        help='Path to save/load the model')
    parser.add_argument('--user_id', type=str, default=None,
                        help='User ID for access control and auditing')
    parser.add_argument('--user_role', type=str, default='admin',
                        help='User role for access control')
    
    return parser.parse_args()

def main():
    """Main execution function."""
    args = parse_arguments()
    logger = setup_logging()
    
    logger.info(f"Starting Heart Disease Prediction System in {args.mode} mode")
    
    # Set up directories
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    os.makedirs(args.output, exist_ok=True)
    
    try:
        # Initialize security module
        security = HealthDataSecurity()
        
        # Check user access permissions
        operation_map = {
            'train': 'write',
            'predict': 'read',
            'evaluate': 'read'
        }
        operation = operation_map.get(args.mode, 'read')
        
        if not security.check_access(args.user_id, args.user_role, operation):
            logger.error(f"Access denied: User {args.user_id} with role {args.user_role} "
                         f"cannot perform {operation} operation")
            return
        
        # Initialize data processor
        data_processor = HeartDataProcessor(args.config)
        
        # Load encryption key if needed
        key = None
        if args.encrypted and args.key:
            key = security.load_key(args.key, args.user_id)
        
        # Mode-specific operations
        if args.mode == 'train':
            # Load and preprocess data
            logger.info(f"Loading data from {args.data}")
            data = data_processor.load_data(args.data, args.encrypted, key)
            logger.info(f"Loaded data with shape {data.shape}")
            
            X, y = data_processor.preprocess_data(data)
            X_train, X_test, y_train, y_test = data_processor.prepare_data(X, y)
            
            # Train model
            logger.info("Training heart disease prediction model")
            model = HeartDiseaseModel(args.config)
            model.train(X_train, y_train)
            
            # Evaluate model
            logger.info("Evaluating model performance")
            evaluator = ModelEvaluator(args.output)
            feature_names = X.columns.tolist() if hasattr(X, 'columns') else None
            results = evaluator.evaluate_model(model.model, X_test, y_test, feature_names)
            
            # Save model and preprocessing state
            logger.info(f"Saving model to {args.model_path}")
            model.save_model(args.model_path)
            
            scaler_path = os.path.join(os.path.dirname(args.model_path), 'scaler.pkl')
            logger.info(f"Saving scaler to {scaler_path}")
            data_processor.save_scaler(scaler_path)
            
            # Generate visualization
            evaluator.plot_roc_curve(results, 'roc_curve.png')
            evaluator.plot_confusion_matrix(results, 'confusion_matrix.png')
            evaluator.plot_feature_importance(results, filename='feature_importance.png')
            
            # Log performance
            logger.info(f"Model accuracy: {results['accuracy']:.3f}")
            logger.info(f"Model AUC: {results['auc']:.3f}")
        
        elif args.mode == 'predict':
            # Load model and scaler
            logger.info(f"Loading model from {args.model_path}")
            model = HeartDiseaseModel()
            model.load_model(args.model_path)
            
            scaler_path = os.path.join(os.path.dirname(args.model_path), 'scaler.pkl')
            logger.info(f"Loading scaler from {scaler_path}")
            data_processor.load_scaler(scaler_path)
            
            # Load and preprocess data
            logger.info(f"Loading data from {args.data}")
            data = data_processor.load_data(args.data, args.encrypted, key)
            
            X, _ = data_processor.preprocess_data(data)
            X_processed = data_processor.prepare_data(X, training=False)
            
            # Make predictions
            logger.info("Generating predictions")
            predictions = model.predict(X_processed)
            probabilities = model.predict_proba(X_processed)
            
            # Save predictions
            results_df = data.copy()
            results_df['predicted_class'] = predictions
            results_df['probability'] = probabilities[:, 1]
            
            output_file = os.path.join(args.output, 'predictions.csv')
            logger.info(f"Saving predictions to {output_file}")
            results_df.to_csv(output_file, index=False)
                              
            
            # Encrypt predictions if needed
        if args.encrypted and key:
            encrypted_output = output_file + '.encrypted'
            security.encrypt_file(output_file, encrypted_output, key, args.user_id)
            logger.info(f"Encrypted predictions saved to {encrypted_output}")
        
        elif args.mode == 'evaluate':
            # Load model
            logger.info(f"Loading model from {args.model_path}")
            model = HeartDiseaseModel()
            model.load_model(args.model_path)
            
            # Load scaler
            scaler_path = os.path.join(os.path.dirname(args.model_path), 'scaler.pkl')
            logger.info(f"Loading scaler from {scaler_path}")
            data_processor.load_scaler(scaler_path)
            
            # Load and preprocess data
            logger.info(f"Loading data from {args.data}")
            data = data_processor.load_data(args.data, args.encrypted, key)
            X, y = data_processor.preprocess_data(data)
            
            if y is None:
                logger.error("Target variable not found in dataset. Cannot evaluate model.")
                return
            
            X_processed = data_processor.prepare_data(X, training=False)
            
            # Evaluate model
            logger.info("Evaluating model performance")
            evaluator = ModelEvaluator(args.output)
            feature_names = X.columns.tolist() if hasattr(X, 'columns') else None
            results = evaluator.evaluate_model(model.model, X_processed, y, feature_names)
            
            # Generate visualizations
            evaluator.plot_roc_curve(results, 'roc_curve.png')
            evaluator.plot_confusion_matrix(results, 'confusion_matrix.png')
            evaluator.plot_feature_importance(results, filename='feature_importance.png')
            
            # Log performance
            logger.info(f"Model accuracy: {results['accuracy']:.3f}")
            logger.info(f"Model AUC: {results['auc']:.3f}")
        
        logger.info(f"Heart Disease Prediction System {args.mode} completed successfully")
    
    except Exception as e:
        logger.exception(f"Error in Heart Disease Prediction System: {str(e)}")
        raise

if __name__ == "__main__":
    main()