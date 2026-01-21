import pandas as pd;
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import pickle
import os

df = pd.read_csv('data/Housing.csv')

x = df.drop(columns=['price'])
y = df['price']

num_cols = x.select_dtypes(include=['int64', 'float64']).columns
cat_cols = x.select_dtypes(include=['object']).columns
#print("Numerical Columns:", num_cols)
#print("Categorical Columns:", cat_cols)

num_imputer = SimpleImputer(strategy='median')
x[num_cols] = num_imputer.fit_transform(x[num_cols])

cat_imputer = SimpleImputer(strategy='most_frequent')
x[cat_cols] = cat_imputer.fit_transform(x[cat_cols])

x = pd.get_dummies(x, columns=cat_cols, drop_first=True)
feature_names = x.columns

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(x_train, y_train)

y_pred_lr = lr.predict(x_test)

mae_lr = mean_absolute_error(y_test, y_pred_lr)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr = r2_score(y_test, y_pred_lr)

print("Linear Regression Performance:")
print("MAE:", mae_lr)
print("RMSE:", rmse)
print("R2 Score:", r2_lr)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(x_train, y_train)
y_pred_rf = rf.predict(x_test)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

print("\nRandom Forest Regressor Performance:")
print("MAE:", mae_rf)
print("RMSE:", rmse_rf)
print("R2 Score:", r2_rf)

if r2_rf > r2_lr:
    best_model = rf
    print("\nRandom Forest Regressor performs better than Linear Regression.")
else:
    best_model = lr
    print("\nLinear Regression performs better than Random Forest Regressor.")

os.makedirs("model", exist_ok=True)

with open("model/model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("model/feature_names.pkl", "wb") as f:
    pickle.dump(feature_names, f)

with open("model/num_imputer.pkl", "wb") as f:
    pickle.dump(num_imputer, f)

with open("model/cat_imputer.pkl", "wb") as f:
    pickle.dump(cat_imputer, f)