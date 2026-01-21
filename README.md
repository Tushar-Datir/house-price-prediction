# House Price Prediction Project

A complete Machine Learning project that predicts house prices based on multiple features such as area, bedrooms, location preferences, and amenities. The project includes data preprocessing, model training, evaluation, serialization, and a Streamlit web application for real-time predictions.

## Project Overview

This project aims to build a regression model that accurately predicts house prices using historical housing data.

### Key Features

* Data cleaning & preprocessing
* Feature engineering (One-Hot Encoding)
* Model comparison (Linear Regression vs Random Forest)
* Model evaluation using MAE, RMSE, R²
* Model & scaler persistence using Pickle
* Interactive web interface using Streamlit


## Project Structure

house_price_prediction/
│
├── data/
│   └── Housing.csv
│
├── model/
│   ├── model.pkl
│   └── scaler.pkl
│
├── train.py
├── app.py
├── requirements.txt
└── README.md

---------------------------------------------------------------------------------------
## Technologies Used

* Python 3.x
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* Matplotlib / Seaborn


## Machine Learning Workflow

### 1 Data Loading

* Dataset loaded from `Housing.csv`

### 2 Data Preprocessing

* Numerical features: Median imputation
* Categorical features: Most frequent imputation
* One-hot encoding using `pd.get_dummies()`
* Feature scaling using `StandardScaler`

### 3 Model Training

* Linear Regression
* Random Forest Regressor

### 4 Model Evaluation

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score

Best-performing model is selected automatically.

### 5 Model Saving

* Trained model saved as `model.pkl`
* Scaler saved as `scaler.pkl`


##  How to Run the Project

###  Step 1: Clone Repository

git clone <repository-url>
cd house_price_prediction

###  Step 2: Install Dependencies

pip install -r requirements.txt

###  Step 3: Train the Model

py train.py

### Step 4: Run Streamlit App

streamlit run app.py

## Streamlit Application

The Streamlit app allows users to:

* Input house details
* Predict house prices instantly
* View results in a clean UI

## Model Performance (Example)

Linear Regression:
MAE  : ~970,000
RMSE : ~1,324,000
R²   : ~0.65

## Future Improvements

* Feature importance visualization
* Hyperparameter tuning
* Model deployment (Streamlit Cloud / Render)
* CI/CD integration

## Author
-- Tushar Datir