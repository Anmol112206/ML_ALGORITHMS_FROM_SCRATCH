import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

class KNearestNeighbors:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def get_neighbors(self, X_test_instance):
        distances = [
            (i, self.euclidean_distance(self.X_train[i], X_test_instance))
            for i in range(self.X_train.shape[0])
        ]
        distances.sort(key=lambda x: x[1])
        neighbors = [idx for idx, _ in distances[:self.k]]
        return neighbors

    def predict_instance(self, X_test_instance):
        neighbors = self.get_neighbors(X_test_instance)
        neighbor_values = [self.y_train[j] for j in neighbors]
        return np.mean(neighbor_values)  # For regression

    def predict(self, X_test):
        return np.array([self.predict_instance(X_test[i]) for i in range(X_test.shape[0])])



train_data = pd.read_csv('data/train.csv')
X_train = train_data[["carlength", "carwidth", "carheight", "horsepower", "peakrpm"]].values
y_train = train_data["price"].values

test_data = pd.read_csv('data/test.csv')
X_test = test_data[["carlength", "carwidth", "carheight", "horsepower", "peakrpm"]].values
y_test = test_data["price"].values

k_values = list(range(1, 36, 2))
for i in k_values:
    print(f'For K={i}')
    model = KNearestNeighbors(k=i)
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print(f"Train MSE: {train_mse:.2f}")
    print(f"Test MSE: {test_mse:.2f}")
    print(f"Test R² Score: {test_r2:.2f}\n")



k = 7
model = KNearestNeighbors(k=k)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color='dodgerblue', alpha=0.7, label='Predicted Points')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Ideal Fit (y = x)')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title(f'KNN Regression (k={k}) - Actual vs Predicted')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


