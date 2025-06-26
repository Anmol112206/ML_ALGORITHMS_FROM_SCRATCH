#X_normalized['Gender'] = (X_normalized['Gender'] == 'Male').astype(int).values
#X_normalized == 'Male' first checks for male and assigns 1 to male and 0 otherwise 
#Updates the X_normalized gender vector to 0 and 1 values

#return (probabilities >= threshold).astype(int)
#means retuns 1 if pro>=0.5 else returns 0
#here astype(int) converts the boolean expression into integral 1 or 0

#Accuracy: Overall correctness

#Precision: Out of all predicted positives, how many were actually correct:High precision = few false positives
#expecially for spam detection, cancer diagnosis

#recall(a.k.a. Sensitivity or True Positive Rate): Out of all actual positives, how many did the model detect:High recall = few false negatives
#fraud detection or diagnosing a serious disease

#f1Score: Harmonic mean of precision and recall

#ROC Curve: Receiver Operating Characteristic curve
#graphical representation used to evaluate the performance of a binary classification model
#y-axis plots (recall/TPR):x-axis plots (FPR)
#AUC(area under the curve) : 1:perfect classifier ,0.5: Noskill , <0.5 : worse than random 
#lw stands for line width which is usually 1
#plt.xlim([0.0, 1.0]) sets the limit of x axis from 0 to 1
#plt.legend shows the legend(label text (AUC curve....) ) in the curve 
#, _ is used to ignore the third output from the function which is here not needed

#Logistic Regression
#weigts : small random values for starting weights
#bias is initialised to 0

#To avoid log(0) clipping is important