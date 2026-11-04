import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import joblib


# -----------------------------
# 1. Generate Retail Dataset
# -----------------------------

np.random.seed(42)

n_samples = 1000

store_size = np.random.randint(500, 10000, n_samples)
customers = np.random.randint(100, 3000, n_samples)
marketing = np.random.randint(1000, 100000, n_samples)
inventory = np.random.randint(100, 5000, n_samples)
employees = np.random.randint(5, 100, n_samples)
online_orders = np.random.randint(20, 1500, n_samples)
discount = np.random.uniform(0, 40, n_samples)


# Revenue formula
revenue = (
    store_size * 5
    + customers * 50
    + marketing * 0.8
    + inventory * 10
    + employees * 500
    + online_orders * 40
    - discount * 1000
    + np.random.normal(0, 20000, n_samples)
)


# -----------------------------
# 2. Create DataFrame
# -----------------------------

df = pd.DataFrame({
    "Store_Size": store_size,
    "Customers": customers,
    "Marketing_Spend": marketing,
    "Inventory_Level": inventory,
    "Employees": employees,
    "Online_Orders": online_orders,
    "Discount_Percentage": discount,
    "Revenue": revenue
})


# Save dataset
df.to_csv("retail_store_data.csv", index=False)

print("Dataset created successfully!")


# -----------------------------
# 3. Features and Target
# -----------------------------

X = df[
    [
        "Store_Size",
        "Customers",
        "Marketing_Spend",
        "Inventory_Level",
        "Employees",
        "Online_Orders",
        "Discount_Percentage"
    ]
]

y = df["Revenue"]


# -----------------------------
# 4. Train Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# -----------------------------
# 5. Random Forest Model
# -----------------------------

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    max_depth=15
)

model.fit(X_train, y_train)


# -----------------------------
# 6. Evaluation
# -----------------------------

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("--------------------------------")
print("MODEL PERFORMANCE")
print("--------------------------------")
print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 4))


# -----------------------------
# 7. Save Model
# -----------------------------

joblib.dump(model, "revenue_model.pkl")

print("--------------------------------")
print("Model saved as revenue_model.pkl")
print("--------------------------------")