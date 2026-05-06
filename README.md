<<<<<<< Updated upstream
# Predicting-Insurance-Claim-Amounts
=======
# Predicting Insurance Claim Amounts

A machine learning project that predicts annual medical insurance claim amounts using personal, medical, and policy-related information.  
The project uses **Linear Regression** and includes a premium **Streamlit dashboard** for interactive insurance claim prediction.

---

## Project Objective

The objective of this project is to estimate medical insurance claim amounts based on user-related health, lifestyle, and insurance policy data.

This project applies **Linear Regression** to predict the annual medical insurance cost using features such as:

- Age
- BMI
- Smoking status
- Income
- Risk score
- Chronic conditions
- Insurance plan details
- Region and other policy-related attributes

The target column used for prediction is:

```text
annual_medical_cost
```

This column represents the estimated annual medical insurance claim amount.

---

## Dataset

The dataset used in this project is a medical insurance dataset containing approximately **100,000 records**.

It includes personal, health-related, and insurance-related information that helps estimate the expected claim amount for each individual.

### Target Variable

| Column Name | Description |
|---|---|
| `annual_medical_cost` | Estimated annual medical insurance claim amount |

---

## Tools and Technologies

This project was developed using the following tools and technologies:

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Plotly
- Streamlit
- VS Code
- Jupyter Notebook

---

## Project Workflow

The complete workflow of the project includes:

1. Loaded and explored the dataset
2. Cleaned missing values
3. Selected useful numerical and categorical features
4. Performed exploratory data analysis
5. Visualized important relationships
6. Trained a Linear Regression model
7. Evaluated the model using regression metrics
8. Built a premium Streamlit dashboard for interactive prediction

---

## Exploratory Data Analysis

During EDA, the following relationships were analyzed and visualized:

- Age vs claim amount
- BMI vs claim amount
- Smoking status vs claim amount
- Region-wise claim analysis
- Risk score impact on claim amount
- Chronic conditions and medical cost relationship

These visualizations helped identify the major factors affecting insurance claim amounts.

---

## Model Used

The machine learning model used in this project is:

### Linear Regression

Linear Regression was selected to predict continuous insurance claim amounts and to understand how different features influence medical costs.

---

## Evaluation Metrics

The model was evaluated using standard regression metrics:

| Metric | Description |
|---|---|
| MAE | Mean Absolute Error |
| RMSE | Root Mean Squared Error |
| R² Score | Measures how well the model explains variation in the target variable |

---

## Streamlit Dashboard Features

A premium Streamlit dashboard was created to make the project interactive and user-friendly.

The dashboard allows users to enter personal and health-related information and receive an estimated insurance claim amount.

### Dashboard Includes

- Interactive input form
- Predicted claim amount
- Model performance metrics
- Age vs claim amount graph
- BMI vs claim amount graph
- Smoking status impact graph
- Region-wise claim analysis
- Premium dark luxury UI design

---

## Key Insights

- Smoking status has a strong impact on medical insurance charges.
- Higher BMI can contribute to higher claim amounts.
- Age is an important factor in predicting medical costs.
- Risk score and chronic conditions improve prediction quality.
- Insurance plan details also help improve the accuracy of claim estimation.

---

## Project Structure

```text
Predicting-Insurance-Claim-Amounts/
│
├── dataset/
│   └── insurance_data.csv
│
├── notebooks/
│   └── insurance_claim_prediction.ipynb
│
├── app.py
├── model.pkl
├── requirements.txt
└── README.md
```

---

## How to Run the Project

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Predicting-Insurance-Claim-Amounts
```

### 2. Install Required Libraries

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App

```bash
streamlit run app.py
```

---

## Future Improvements

- Try advanced models such as Random Forest, XGBoost, or Gradient Boosting
- Improve feature engineering
- Add model comparison
- Deploy the dashboard online
- Add more interactive visualizations
- Improve prediction explainability

---

## Author

**Wajiha Babar**

---

## Project Summary

This project demonstrates how machine learning can be used in the insurance domain to estimate medical claim amounts.  
By combining data preprocessing, exploratory data analysis, Linear Regression modeling, and an interactive Streamlit dashboard, the project provides a complete end-to-end solution for insurance claim prediction.
>>>>>>> Stashed changes
