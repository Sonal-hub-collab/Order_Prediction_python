# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error

# # =========================
# # STEP 1: Load Data
# # =========================
# train = pd.read_csv("train.csv")
# meal = pd.read_csv("meal_info.csv")
# center = pd.read_csv("fulfilment_center_info.csv")

# # =========================
# # STEP 2: Merge Data
# # =========================
# df = train.merge(meal, on="meal_id")
# df = df.merge(center, on="center_id")

# print("Data Loaded and Merged")
# print(df.head())

# # =========================
# # STEP 3: Select Features
# # =========================
# features = ['base_price', 'checkout_price', 'emailer_for_promotion', 'homepage_featured']
# target = 'num_orders'

# X = df[features]
# y = df[target]

# # =========================
# # STEP 4: Train-Test Split
# # =========================
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# # =========================
# # STEP 5: Train Model
# # =========================
# model = LinearRegression()
# model.fit(X_train, y_train)

# # =========================
# # STEP 6: Prediction
# # =========================
# predictions = model.predict(X_test)

# print("Sample Predictions:", predictions[:5])

# # =========================
# # STEP 7: Evaluation
# # =========================
# mse = mean_squared_error(y_test, predictions)
# print("Mean Squared Error:", mse)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# =========================
# STEP 1: Load Data
# =========================
train = pd.read_csv("train.csv")
meal = pd.read_csv("meal_info.csv")
center = pd.read_csv("fulfilment_center_info.csv")

# =========================
# STEP 2: Merge Data
# =========================
df = train.merge(meal, on="meal_id")
df = df.merge(center, on="center_id")

print("Data Loaded and Merged Successfully")
print(df.head())

# =========================
# STEP 3: Handle Missing Values
# =========================
print("\nMissing Values:")
print(df.isnull().sum())

df = df.dropna()

# =========================
# STEP 4: Select Features
# =========================
features = [
    'week',
    'center_id',
    'meal_id',
    'base_price',
    'checkout_price',
    'emailer_for_promotion',
    'homepage_featured'
]

target = 'num_orders'

X = df[features]
y = df[target]

# =========================
# STEP 5: Train-Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# =========================
# STEP 6: Train Model
# =========================
model = LinearRegression()
model.fit(X_train, y_train)

# =========================
# STEP 7: Prediction
# =========================
predictions = model.predict(X_test)

print("\nSample Predictions:")
for actual, pred in zip(y_test.iloc[:5], predictions[:5]):
    print(f"Actual: {actual} | Predicted: {pred:.2f}")

# =========================
# STEP 8: Evaluation
# =========================
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print("\nModel Performance")
print("-----------------")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R² Score : {r2:.4f}")

# =========================
# STEP 9: Feature Importance
# =========================
coefficients = pd.DataFrame({
    'Feature': features,
    'Coefficient': model.coef_
})

print("\nFeature Impact:")
print(coefficients.sort_values(by='Coefficient', ascending=False))