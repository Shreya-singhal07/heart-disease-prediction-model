# api.py
"""
API module for the heart disease prediction system.
Provides a REST API interface using Flask.
"""

from flask import Flask, request, jsonify
import pandas as pd
import os
import json
import logging
from datetime import datetime
import joblib

# Import project modules
from data_processing import HeartDataProcessor
from model import HeartDiseaseModel
from security import HealthDataSecurity

# Initialize Flask app
app = Flask(__name__)

# Set up logging
def setup_logging():
    """Set up logging configuration."""
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f'api_{datetime.now().strftime("%Y%m%d")}.log')
    
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    return logging.getLogger('heart_disease_api')

logger = setup_logging()

# Load configuration
def load_config(config_path='config.json'):
    """Load application configuration."""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        # Default configuration
        return {
            'model_path': 'models/heart_disease_model.pkl',
            'scaler_path': 'models/scaler.pkl',
            'features': [
                'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
            ]
        }

config = load_config()

# Initialize security
security = HealthDataSecurity()

# Load model and scaler
try:
    data_processor = HeartDataProcessor()
    data_processor.load_scaler(config['scaler_path'])
    
    model = HeartDiseaseModel()
    model.load_model(config['model_path'])
    
    logger.info("Model and scaler loaded successfully")
except Exception as e:
    logger.error(f"Error loading model or scaler: {str(e)}")
    # We'll continue and initialize them when needed

# API routes
@app.route('/api/predict', methods=['POST'])
def predict():
    """
    API endpoint for making predictions.
    
    Expects JSON input with patient data.
    Returns prediction and probability.
    """
    api_key = request.headers.get('X-API-Key')
    
    # Load api keys from config
    with open('security_config.json', 'r') as f:
        config = json.load(f)
    
    api_keys = config.get('api_keys', {})
    
    if api_key not in api_keys:
        return jsonify({
            'error': 'Unauthorized',
            'message': 'Invalid API key'
        }), 401
        
    try:
        # Check authentication
        auth_header = request.headers.get('Authorization')
        user_id = 'admin'
        user_role = 'admin'
        
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]
            # In a real system, validate token and extract user info
            # For demo, we'll use a simple approach
            try:
                import jwt
                decoded = jwt.decode(token, 'secret_key', algorithms=['HS256'])
                user_id = decoded.get('user_id')
                user_role = decoded.get('role', 'guest')
            except:
                pass
        
        # Check access permission
        if not security.check_access(user_id, user_role, 'write'):
            return jsonify({
                'error': 'Access denied',
                'message': 'You do not have permission to use this API'
            }), 403
        
        # Get data from request
        data = request.json
        
        if not data:
            return jsonify({
                'error': 'No data provided',
                'message': 'Request must include patient data'
            }), 400
        
        # Validate required fields
        required_fields = config['features']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({
                'error': 'Missing data',
                'message': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400
        
        # Convert to DataFrame
        patient_data = pd.DataFrame([data])
        
        # Preprocess data
        X, _ = data_processor.preprocess_data(patient_data)
        X_processed = data_processor.prepare_data(X, training=False)
        
        # Make prediction
        prediction = int(model.predict(X_processed)[0])
        probability = float(model.predict_proba(X_processed)[0, 1])
        
        # Log prediction (excluding PII)
        logger.info(f"Prediction made: {prediction}, probability: {probability:.3f}, user: {user_id}")
        
        # Return result
        return jsonify({
            'prediction': prediction,
            'probability': probability,
            'risk_level': 'High' if probability > 0.7 else 'Medium' if probability > 0.3 else 'Low',
            'message': 'Heart disease predicted' if prediction == 1 else 'No heart disease predicted'
        })
    
    except Exception as e:
        logger.exception(f"Error in prediction API: {str(e)}")
        return jsonify({
            'error': 'Server error',
            'message': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """API endpoint for health check."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': data_processor is not None
    })

@app.route('/api/model_info', methods=['GET'])
def model_info():
    """API endpoint for model information."""
    try:
        # Check authentication
        auth_header = request.headers.get('Authorization')
        user_id = 'admin'
        user_role = 'admin'
        
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]
            try:
                import jwt
                decoded = jwt.decode(token, 'secret_key', algorithms=['HS256'])
                user_id = decoded.get('user_id')
                user_role = decoded.get('role', 'guest')
            except:
                pass
        
        # Check access permission
        if not security.check_access(user_id, user_role, 'write'):
            return jsonify({
                'error': 'Access denied',
                'message': 'You do not have permission to access model information'
            }), 403
        
        # Get model metadata
        model_path = config['model_path']
        model_stats = {}
        
        # Try to load model metadata if available
        metadata_path = os.path.join(os.path.dirname(model_path), 'model_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                model_stats = json.load(f)
        
        # Get basic model info
        model_info = {
            'model_type': 'Logistic Regression',
            'features': config['features'],
            'model_path': model_path,
            'model_stats': model_stats
        }
        
        return jsonify(model_info)
    
    except Exception as e:
        logger.exception(f"Error in model_info API: {str(e)}")
        return jsonify({
            'error': 'Server error',
            'message': str(e)
        }), 500

@app.route('/api/batch_predict', methods=['POST'])
def batch_predict():
    """
    API endpoint for batch predictions.
    
    Expects a CSV or JSON file uploaded with patient data.
    Returns predictions for all patients.
    """
    try:
        # Check authentication
        auth_header = request.headers.get('Authorization')
        user_id = 'admin'
        user_role = 'admin'
        
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]
            try:
                import jwt
                decoded = jwt.decode(token, 'secret_key', algorithms=['HS256'])
                user_id = decoded.get('user_id')
                user_role = decoded.get('role', 'guest')
            except:
                pass
        
        # Check access permission
        if not security.check_access(user_id, user_role, 'write'):
            return jsonify({
                'error': 'Access denied',
                'message': 'You do not have permission to use this API'
            }), 403
        
        # Check if file is uploaded
        if 'file' not in request.files:
            return jsonify({
                'error': 'No file provided',
                'message': 'Request must include a file'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'error': 'No file selected',
                'message': 'No file selected for uploading'
            }), 400
        
        # Determine file type and load data
        if file.filename.endswith('.csv'):
            data = pd.read_csv(file)
        elif file.filename.endswith('.json'):
            data = pd.read_json(file)
        else:
            return jsonify({
                'error': 'Invalid file type',
                'message': 'File must be CSV or JSON'
            }), 400
        
        # Validate required fields
        required_fields = config['features']
        missing_fields = [field for field in required_fields if field not in data.columns]
        
        if missing_fields:
            return jsonify({
                'error': 'Missing data',
                'message': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400
        
        # Preprocess data
        X, _ = data_processor.preprocess_data(data)
        X_processed = data_processor.prepare_data(X, training=False)
        
        # Make predictions
        predictions = model.predict(X_processed).tolist()
        probabilities = model.predict_proba(X_processed)[:, 1].tolist()
        
        # Prepare results
        results = {
            'predictions': predictions,
            'probabilities': probabilities,
            'risk_levels': ['High' if p > 0.7 else 'Medium' if p > 0.3 else 'Low' for p in probabilities]
        }
        
        # Log batch prediction
        logger.info(f"Batch prediction made for {len(predictions)} patients, user: {user_id}")
        
        return jsonify(results)
    
    except Exception as e:
        logger.exception(f"Error in batch_predict API: {str(e)}")
        return jsonify({
            'error': 'Server error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)