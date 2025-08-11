# Banking ML Pipeline

A comprehensive machine learning pipeline for customer segmentation and loan eligibility prediction in the banking sector.

## 🎯 Project Overview

This project implements an integrated ML pipeline that combines:
- **Customer Segmentation**: Using unsupervised learning (K-Means clustering) to identify distinct customer groups
- **Loan Eligibility Prediction**: Using supervised learning to predict loan approval probability
- **Production-Ready API**: RESTful API for real-time predictions
- **Automated Pipeline**: End-to-end automation from data preprocessing to model deployment

## 📁 Project Structure

```
banking-ml-pipeline/
├── src/
│   ├── data/               # Data processing modules
│   ├── models/             # ML model implementations
│   ├── pipeline/           # Integrated pipeline
│   ├── api/                # Flask API
│   ├── monitoring/         # Model monitoring
│   └── utils/              # Utilities
├── scripts/                # Training and evaluation scripts
├── tests/                  # Unit tests
├── configs/                # Configuration files
├── notebooks/              # Jupyter notebooks
├── models/                 # Saved models
├── logs/                   # Application logs
└── data/                   # Data files
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip or conda
- Docker (optional)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/banking-ml-pipeline.git
cd banking-ml-pipeline
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install the package:
```bash
pip install -e .
```

### Training the Model

1. With synthetic data:
```bash
python scripts/train_model.py --synthetic --n_samples 5000
```

2. With your own data:
```bash
python scripts/train_model.py --data path/to/your/data.csv
```

### Running the API

1. Start the Flask server:
```bash
python -m src.api.app
```

2. The API will be available at `http://localhost:5000`

### Using Docker

1. Build and run with Docker Compose:
```bash
docker-compose up
```

2. Train models in Docker:
```bash
docker-compose --profile training up
```

## 📊 API Endpoints

### Health Check
```http
GET /health
```

### Single Prediction
```http
POST /predict
Content-Type: application/json

{
  "age": 35,
  "income": 75000,
  "education": "Master",
  "occupation": "Professional",
  "marital_status": "Married",
  "account_age_months": 48,
  "avg_balance": 25000,
  "num_products": 3,
  "monthly_transactions": 25,
  "avg_transaction_amount": 1500,
  "max_transaction_amount": 5000,
  "credit_score": 720,
  "existing_loans": 1,
  "loan_amount_requested": 200000,
  "loan_term_months": 36,
  "previous_defaults": 0,
  "payment_history_score": 0.95
}
```

Response:
```json
{
  "customer_segment": 2,
  "segment_name": "Premium High-Value",
  "loan_approved": true,
  "approval_probability": 0.87,
  "recommendation": "Strong candidate for loan approval",
  "confidence": "High",
  "timestamp": "2024-01-01T12:00:00"
}
```

### Batch Prediction
```http
POST /batch_predict
Content-Type: multipart/form-data

file: customers.csv
```

### Model Information
```http
GET /model_info
```

### Segment Information
```http
GET /segments
```

### Feature Importance
```http
GET /feature_importance?top_n=20
```

## 🔧 Configuration

Configuration can be customized in `configs/model_config.yaml`:

```yaml
project:
  name: Banking ML Pipeline
  version: 1.0.0

preprocessing:
  scaling_method: robust
  encoding_method: onehot

segmentation:
  algorithm: kmeans
  n_clusters_range: [2, 10]

prediction:
  models: [logistic_regression, random_forest, xgboost]
  test_size: 0.3
  use_smote: true
```

## 📈 Model Performance

Expected performance metrics (on synthetic data):

- **Segmentation**:
  - Silhouette Score: 0.35-0.45
  - 4-5 distinct customer segments

- **Loan Prediction**:
  - AUC-ROC: 0.85-0.92
  - Accuracy: 82-88%
  - Precision: 80-85%
  - Recall: 75-82%

## 🧪 Testing

Run unit tests:
```bash
pytest tests/
```

With coverage:
```bash
pytest --cov=src tests/
```

## 📊 Data Requirements

The pipeline expects the following features:

### Demographic Features
- age, income, education, occupation, marital_status

### Account Features
- account_age_months, avg_balance, num_products

### Transaction Features
- monthly_transactions, avg_transaction_amount, max_transaction_amount

### Credit Features
- credit_score, existing_loans, previous_defaults, payment_history_score

### Loan Application Features
- loan_amount_requested, loan_term_months, loan_purpose

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- BITS Pilani for the academic guidance
- Scikit-learn, XGBoost, and other open-source libraries
- The banking and fintech community for domain insights

## 📧 Contact

Your Name - your.email@example.com

Project Link: [https://github.com/yourusername/banking-ml-pipeline](https://github.com/yourusername/banking-ml-pipeline)