import pandas as pd

def normalize_features(X):
    X_normalized = X.copy()
    X_normalized['Gender'] = (X_normalized['Gender'] == 'Male').astype(int).values 
    return X_normalized.astype(float)


train_data = pd.read_csv('data/train.csv')
train_data = normalize_features(train_data)
X_train = train_data[['Gender', 'Age', 'EstimatedSalary']].values
y_train = train_data['Purchased'].values


test_data = pd.read_csv('data/test.csv')
test_data = normalize_features(test_data)
X_test = test_data[['Gender', 'Age', 'EstimatedSalary']].values
y_test = test_data['Purchased'].values


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print(model.intercept_,model.coef_)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))