import numpy as np

X = np.random.rand(100,1)    #this function creates a 100*1 matrix of float numbers between 0 and 1
y = 4 + 3 * X + np.random.randn(100, 1)

#np.c_[] is a NumPy shorthand for column-wise concatenation
#np.ones((100, 1)) creates a 100x1 column vector where every element is 1
#So X_b becomes a 100x2 matrix, where the first column is all 1s (for the bias term), and the second column is your original X
#X_b.T find the transpose of the matrix 
X_b = np.c_[np.ones((100, 1)), X] 
#This is the Normal Equation used to compute the best-fit parameters for Linear Regression:
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

print(theta_best)


#stochastic regression
from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.1)
sgd_reg.fit(X, y.ravel())
print(sgd_reg.intercept_, sgd_reg.coef_)