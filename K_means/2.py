import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score


train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

features = ["carlength", "carwidth", "carheight", "horsepower", "peakrpm"]
X_train = train_data[features].values
y_train = train_data["price"].values

X_test = test_data[features].values
y_test = test_data["price"].values


k = 7

knn = KNeighborsRegressor(n_neighbors=k)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"KNN Regression with k={k}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")


plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color='green', alpha=0.6, label='Predicted')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', label='Ideal Fit (y = x)')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(f"Actual vs Predicted Prices (k={k})")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
