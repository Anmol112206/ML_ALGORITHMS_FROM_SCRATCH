import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


def binary_cross_entropy_loss(Y_true, Y_pred):
    epsilon = 1e-15
    Y_pred = np.clip(Y_pred, epsilon, 1 - epsilon)
    return -np.mean(Y_true * np.log(Y_pred) + (1 - Y_true) * np.log(1 - Y_pred))


def sigmoid(z):
    return 1/(1+np.exp(-z))


def fit_logistic_regression(X,y):
    learning_rate = 0.1
    num_epochs = 10
    n_samples, n_features = X.shape
    weights = np.random.randn(n_features).astype(float) * 0.01
    bias = 0.0 
    y = y.astype(float)
    for epoch in range(num_epochs):
        z = np.dot(X, weights) + bias
        y_pred = sigmoid(z)
 
        loss = binary_cross_entropy_loss(y, y_pred)
        dz = y_pred - y
        dw = (1/n_samples) * np.dot(X.T, dz)
        db = (1/n_samples) * np.sum(dz)
   
        weights -= learning_rate * dw
        bias -= learning_rate * db
    
    return weights, bias



def normalize_features(X):
    X_normalized = X.copy()
    X_normalized['Gender'] = (X_normalized['Gender'] == 'Male').astype(int).values 
    X_normalized[["Age"	,"EstimatedSalary"]] = (X_normalized[["Age"	,"EstimatedSalary"]] - X_normalized[["Age"	,"EstimatedSalary"]].mean()) / X_normalized[["Age"	,"EstimatedSalary"]].std()
    return X_normalized.astype(float)


def predict(X,weights,bias,threshold = 0.5):
    z = np.dot(X,weights) + bias
    probabilities = sigmoid(z)
    return (probabilities >= threshold).astype(int)


def evaluate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return accuracy, precision, recall, f1


def plot_roc_curve(Y_true, Y_scores):
    fpr, tpr, _ = roc_curve(Y_true, Y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()
    
    return roc_auc


train_data = pd.read_csv('data/train.csv')
train_data = normalize_features(train_data)
X_train = train_data[['Gender', 'Age', 'EstimatedSalary']].values
y_train = train_data['Purchased'].values


test_data = pd.read_csv('data/test.csv')
test_data = normalize_features(test_data)
X_test = test_data[['Gender', 'Age', 'EstimatedSalary']].values
y_test = test_data['Purchased'].values

weights,bias = fit_logistic_regression(X_train, y_train)
print(weights,bias)

y_pred_test = predict(X_test, weights, bias)

accuracy,precision,recall,f1_score = evaluate_metrics(y_test, y_pred_test)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)

y_scores = sigmoid(np.dot(X_test, weights) + bias)
plot_roc_curve(y_test, y_scores)