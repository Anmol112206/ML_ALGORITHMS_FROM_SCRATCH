#np.random.random_sample(X.shape[1]) 
#X.shape[1] gives the number of columns while X.shape[0] gives the number of rows
#np.random.random_sample(n) generates a array of size n with float values between 0.0 and 1.0
#np.random.random() generates a single float in range 0.0 to 1.0

#assert len(Y_true) == len(Y_pred)
#is a sanity check to check the length of both to be equal otherwise gives assertion error

#delta_A = np.clip(delta_A, -clip_value, clip_value)
#This prevents gradients from exploding (especially if data is bad or learning rate is too high).
#clips the value in between min and max value , any value less than minimum value becomes min value

#epochs are defined so to reach a good value: each sample is trained epochs times during the process