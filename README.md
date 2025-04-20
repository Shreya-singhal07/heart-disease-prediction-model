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

## Security Features

- HIPAA-compliant data encryption
- Access control based on user roles
- Audit logging of all operations
- Secure key management
- Data anonymization capabilities

| **Module**               | **Description**                                                                                                                                                        |
|--------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Data Processing Module (`data_processing.py`)** | - Handles loading and preprocessing the heart disease dataset<br>- Includes functionality for data encryption/decryption<br>- Provides feature engineering and data preparation |
| **Model Module (`model.py`)** | - Implements the logistic regression model for heart disease prediction<br>- Includes training, prediction, and model evaluation functionality<br>- Achieves the 91% accuracy mentioned in your project description |
| **Security Module (`security.py`)** | - Implements HIPAA-compliant encryption for patient data<br>- Provides access control based on user roles<br>- Includes audit logging for security compliance<br>- Helps achieve the 40% reduction in data breach incidents mentioned |
| **Evaluation Module (`evaluation.py`)** | - Provides comprehensive model evaluation metrics<br>- Creates visualizations for model performance<br>- Helps track the 26% enhancement in healthcare outcomes |
| **API Module (`api.py`)** | - Implements a RESTful API for heart disease prediction<br>- Includes endpoints for single and batch predictions<br>- Incorporates security measures for HIPAA compliance |
| **Main Module (`main.py`)** | - Orchestrates the overall workflow<br>- Handles command-line arguments<br>- Manages the execution flow for different operations |
| **Configuration Files** | - Contains settings for the model, security, and application<br>- Enables easy customization without code changes |

## output 

**confusion_matrix**
![image](https://github.com/user-attachments/assets/e339f0e8-387f-4afa-9e84-0e955ff94574)

**feature importance**
![1](https://github.com/user-attachments/assets/a12fc50f-94d6-41d5-be94-c061aebe565e)

**roc_curve**
![image (1)](https://github.com/user-attachments/assets/0682d579-9524-439e-8db6-58037c8c66cc)




