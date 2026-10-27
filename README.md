# Customer Churn Prediction Using Machine Learning

## Project Overview
Customer Churn Prediction is a machine learning project that estimates whether a telecom customer is likely to discontinue a service based on customer demographics, service usage, contract information, and billing details.

## Objectives
- Analyze customer data to identify churn patterns.
- Perform exploratory data analysis and data preprocessing.
- Train and compare classification models.
- Evaluate model performance using classification metrics.
- Build an interactive web application using Streamlit.

## Technologies Used
- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Streamlit
- Git and GitHub

## Machine Learning Workflow
1. Load and inspect the dataset.
2. Perform exploratory data analysis.
3. Handle missing values and encode categorical features.
4. Split the dataset into training and testing sets.
5. Train Logistic Regression and Random Forest models.
6. Evaluate precision, recall, F1-score, ROC-AUC, and the confusion matrix.
7. Use the trained model in a Streamlit application.

## Project Structure
- `main.py` — Exploratory data analysis
- `train_model.py` — Initial model training
- `evaluate_model.py` — Model comparison and evaluation
- `app.py` — Interactive prediction application
- `requirements.txt` — Required Python libraries

## How to Run

### 1. Install dependencies
```bash
python -m pip install -r requirements.txt
```

### 2. Obtain the dataset
Download the Telco Customer Churn dataset from Kaggle and place the CSV in the project folder. See the dataset source below.

### 3. Train and evaluate the model
```bash
python evaluate_model.py
```

This script trains candidate models, performs cross-validation, evaluates the selected model on a held-out test set, and saves `churn_model.pkl`.

### 4. Launch the application
```bash
python -m streamlit run app.py
```

## Evaluation
Model performance is evaluated using accuracy, precision, recall, F1-score, ROC-AUC, and a confusion matrix.

**Final model and metrics:** Add the selected model name and the actual held-out test metrics after completing evaluation.

## Dataset
IBM Telco Customer Churn dataset:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Review the dataset's terms before redistributing the data.

## Limitations
Predictions depend on the dataset, selected features, training process, and decision threshold. Predicted churn probabilities are estimates, not guarantees of future customer behavior.

## Future Improvements
- Hyperparameter tuning using cross-validation.
- Prediction threshold optimization using validation data.
- Feature importance and model interpretability.
- Deployment to a hosting platform.
