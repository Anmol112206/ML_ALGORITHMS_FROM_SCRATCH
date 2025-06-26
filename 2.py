import numpy as np

m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)

from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=2, include_bias=False)   #generate polynomial terms upto 2 without the bias term (intercept)
X_poly = poly_features.fit_transform(X)   
X[0]
X_poly[0]

# You would not want to miss a potential tumor, so you will want a low threshold. 
#A specialist will review the output of the algorithm which reduces the possibility of a ‘false positive’. 
# The key point of this question is to note that the threshold value does not need to be 0.5.

#Whenevr we have to define the column 
#col = ['id', 'diagnosis'] + [f'feature{i}' for i in range(1, 31)]