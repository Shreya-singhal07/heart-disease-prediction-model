<<<<<<< HEAD
# heart-disease-prediction-model
=======
// README.md - Project documentation
# Heart Disease Prediction System

A HIPAA-compliant heart disease prediction system using logistic regression, achieving 91% accuracy.

## Features

- Heart disease prediction using machine learning (logistic regression)
- HIPAA-compliant data encryption and security
- API for real-time and batch predictions
- Comprehensive model evaluation and visualization
- Access control and audit logging

## Project Structure

```
heart_disease_prediction/
├── config.json                # Main configuration
├── security_config.json       # Security configuration
├── main.py                    # Main execution script
├── data_processing.py         # Data processing module
├── model.py                   # Heart disease model implementation
├── security.py                # HIPAA security module
├── evaluation.py              # Model evaluation and visualization
├── api.py                     # REST API implementation
├── data/                      # Data directory
│   └── heart.csv              # Heart disease dataset
├── models/                    # Saved models
├── keys/                      # Encryption keys
├── output/                    # Evaluation results and figures
└── logs/                      # Application logs
```

## Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Configure your environment in `config.json` and `security_config.json`

## Usage

### Training a Model

```bash
python main.py --mode train --data data/heart.csv --config config.json
```

### Making Predictions

```bash
python main.py --mode predict --data new_patients.csv --model_path models/heart_disease_model.pkl
```

### Starting the API

```bash
python api.py
```

## Security Features

- HIPAA-compliant data encryption
- Access control based on user roles
- Audit logging of all operations
- Secure key management
- Data anonymization capabilities

## API Endpoints

- `POST /api/predict` - Predict heart disease for a single patient
- `POST /api/batch_predict` - Batch prediction for multiple patients
- `GET /api/health` - API health check
- `GET /api/model_info` - Information about the model

## License

MIT
>>>>>>> de4d5ed (firstcommit)
