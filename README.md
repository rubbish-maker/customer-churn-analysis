# Customer Churn Analysis

## Project Overview

This project analyzes customer churn behavior in an e-commerce platform using machine learning techniques.

The goal is to identify key factors related to customer churn and build predictive models to classify whether a customer is likely to leave.

---

## Dataset

Dataset: Ecommerce Customer Churn Dataset (Kaggle)

The dataset contains 50,000 customer records and includes:

- User behavior
- Purchase activity
- Engagement metrics
- Customer service interactions
- Financial indicators

Target variable:

- `Churned`
  - 0 = retained customer
  - 1 = churned customer

---

## Exploratory Data Analysis

Performed exploratory data analysis to compare churned and retained customers.

Variables with larger mean differences between churn groups were further selected for visualization and modeling.

Selected features included:

- Cart Abandonment Rate
- Lifetime Value
- Days Since Last Purchase
- Credit Balance
- Average Order Value

Boxplots were used to visualize distribution differences between churned and non-churned users.

## Machine Learning Models

### 1. Logistic Regression
Used to identify interpretable factors associated with customer churn.

Key findings:
- Higher cart abandonment rate increased churn risk
- Longer inactivity periods increased churn risk
- Higher purchase frequency reduced churn risk

### 2. Random Forest Classifier
Built a predictive churn classification model using all numerical behavioral features.

Model improvements included:
- Using full-feature training instead of only selected variables
- Applying class balancing to improve minority-class recall
- Evaluating feature importance for business interpretation

Final Random Forest Performance:
- Accuracy: 0.91
- Recall (Churned Users): 0.76
- F1-score: 0.84

---

## Key Findings

Important churn-related features include:

- Cart Abandonment Rate
- Lifetime Value
- Days Since Last Purchase
- Customer Service Calls
- Returns Rate

Customers with higher cart abandonment and longer inactivity periods were more likely to churn.

## Key Predictive Features

Random Forest feature importance analysis identified the following major predictors of customer churn:

- Cart Abandonment Rate
- Lifetime Value
- Days Since Last Purchase
- Email Open Rate
- Login Frequency

---

## Technologies Used

- Python
- pandas
- matplotlib
- scikit-learn

---

## Project Structure

```plaintext
DA/
│
├── DA.py
├── ecommerce_customer_churn_dataset.csv
├── images/
└── README.md
```

---

## Future Improvements

Possible future improvements:

- XGBoost modeling
- Interactive dashboard
- SHAP explainability analysis
- Streamlit web app deployment
