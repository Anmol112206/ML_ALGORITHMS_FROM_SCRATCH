import pandas as pd

train_data = pd.read_csv('data/train.csv')
X_train = train_data[["carlength", "carwidth", "carheight", "horsepower", "peakrpm"]].values
y_train = train_data["price"].values


test_data = pd.read_csv('data/test.csv')
X_test = test_data[["carlength", "carwidth", "carheight", "horsepower", "peakrpm"]].values
y_test = test_data["price"].values

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print(model.intercept_,model.coef_)

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
print(mean_absolute_error(y_test,y_pred))
print(mean_absolute_percentage_error(y_test,y_pred))
print(mean_squared_error(y_test,y_pred))