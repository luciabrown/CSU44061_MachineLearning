# First line of the data file: # id:7-14-7 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import LinearSVC 
from collections import Counter

df = pd.read_csv("Week2Assignment/week2.php.csv",header=None,comment="#",sep=",",skipinitialspace=True)
#print(df.head())
X1=df.iloc[:,0] # Col1
X2=df.iloc[:,1] # Col2
X=np.column_stack((X1,X2)) # Stack into 2D array
y=df.iloc[:,2]  # Target vals

# QUESTION A
# PART I 
plt.figure(figsize=(12,12))
plt.scatter(X1[y==1],X2[y==1],marker="+",color="lime",label="Target = +1")
plt.scatter(X1[y==-1],X2[y==-1],marker="o",color="blue",label="Target = -1")
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Question A - Part I")
plt.legend()
plt.grid(False)
plt.show()

# PART II - https://youtube.com/shorts/8it0yrJzQfU?si=1UoVpwNGf7flDZeh

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
X1_test = X_test[:, 0]
X2_test = X_test[:, 1]

classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)

# Get model parameters
print("Intercept (b0):", classifier.intercept_[0])
print("Coefficients (b1, b2):", classifier.coef_[0])

for i, coef in enumerate(classifier.coef_[0]):
    effect = "increases" if coef > 0 else "decreases"
    print(f"Feature X{i+1} {effect} the probability of predicting +1 (coefficient={coef:.3f})")

print("Accuracy of Logistical Regression:", round(accuracy_score(y_test, y_pred),7))

# PART III - https://youtu.be/ZsM2z0pTbnk?si=iMnwQTl_QFWL7ani

#Add predictions to plot
plt.figure(figsize=(12,12))
plt.scatter(X1[y==1],X2[y==1],marker="+",color="lime",label="Target = +1")
plt.scatter(X1[y==-1],X2[y==-1],marker="o",color="blue",label="Target = -1")
plt.scatter(X1_test[(y_test == y_pred) & (y_test == 1)], X2_test[(y_test == y_pred) & (y_test == 1)], facecolors='none', edgecolors='red', s=100, label="Classifier Predicted +1")
plt.scatter(X1_test[(y_test == y_pred) & (y_test == -1)], X2_test[(y_test == y_pred) & (y_test == -1)], facecolors='none', edgecolors='darkred', s=100, label="Classifier Predicted -1")

# Add decision boundary as line on plot
b0 = classifier.intercept_[0]
b1, b2 = classifier.coef_[0]
x_vals = np.linspace(X1.min()-0.1, X1.max()+0.1, 200)
y_vals = -(b0 + b1 * x_vals) / b2
plt.plot(x_vals, y_vals, color='black', linewidth=1.5, label="Decision Boundary for Logistic Regression Classifier")

plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Question A - Part III")
plt.legend(bbox_to_anchor=(1, 0.5))
plt.tight_layout() 
plt.grid(False)
plt.show()

# PART IV - SEE REPORT PDF

# QUESTION B

# PART I - https://youtu.be/kPkwf1x7zpU?si=z3680NVTyFd0b_h4
svcScoreDictionary={}
cVals=[0.001,1,100]

# https://youtu.be/joTa_FeMZ2s?si=3DI8TfqasbA9BtgO
# https://youtu.be/5oVQBF_p6kY?si=8PbeAmYH38awD8_9 
X_train_SVM,X_test_SVM,y_train_SVM,y_test_SVM = train_test_split(X,y,test_size=0.2,random_state=0)
modelZero =LinearSVC(C=0.001)
modelOne=LinearSVC(C=1)
modelHundred =LinearSVC(C=100)
modelZero.fit(X_train_SVM, y_train_SVM)
modelOne.fit(X_train_SVM, y_train_SVM)
modelHundred.fit(X_train_SVM, y_train_SVM)
models = {
    0.001: modelZero,
    1: modelOne,
    100: modelHundred
}

for cVal, model in models.items():
    X1_test_SVM = X_test_SVM[:, 0]
    X2_test_SVM = X_test_SVM[:, 1]
    # Print model parameters
    print(f"\nLinear SVC Model for C = {cVal}")
    print("Model Coefficients:\n", model.coef_)
    print("Model Intercept:", model.intercept_)

# PART II - https://youtu.be/_YPScrckx28?si=fBxs_9gB27Ey7EYp

y_pred_Zero=modelZero.predict(X_test_SVM)
y_pred_One=modelOne.predict(X_test_SVM)
y_pred_Hundred=modelHundred.predict(X_test_SVM)
print("\nAccuracy Score for C=0.001: ",round(accuracy_score(y_test_SVM,y_pred_Zero),7))
print("Accuracy Score for C=1: ",round(accuracy_score(y_test_SVM,y_pred_One),7))
print("Accuracy Score for C=100: ",round(accuracy_score(y_test_SVM,y_pred_Hundred),7),"\n")

# PART III

b0_zero = modelZero.intercept_[0]
b1_zero, b2_zero = modelZero.coef_[0]

b0_one = modelOne.intercept_[0]
b1_one, b2_one = modelOne.coef_[0]

b0_hundred = modelHundred.intercept_[0]
b1_hundred, b2_hundred = modelHundred.coef_[0]

# Add decision boundary as line on plot
x_vals = np.linspace(X1.min()-0.1, X1.max()+0.1, 200)
y_vals_Zero = -(b0_zero + b1_zero * x_vals) / b2_zero

x_vals = np.linspace(X1.min()-0.1, X1.max()+0.1, 200)
y_vals_One = -(b0_one + b1_one * x_vals) / b2_one

x_vals = np.linspace(X1.min()-0.1, X1.max()+0.1, 200)
y_vals_Hundred = -(b0_hundred + b1_hundred * x_vals) / b2_hundred


# PLOT C=0.001
plt.figure(figsize=(12,12))
plt.scatter(X1[y==1], X2[y==1], marker="+", color="lime", label="Target = +1")
plt.scatter(X1[y==-1], X2[y==-1], marker="o", color="blue", label="Target = -1")
plt.scatter(X1_test_SVM[(y_test_SVM == y_pred_Zero) & (y_test_SVM == 1)], X2_test_SVM[(y_test_SVM == y_pred_Zero) & (y_test_SVM == 1)],facecolors='none', edgecolors='orange', s=100, label="Predicted +1")
plt.scatter(X1_test_SVM[(y_test_SVM == y_pred_Zero) & (y_test_SVM == -1)], X2_test_SVM[(y_test_SVM == y_pred_Zero) & (y_test_SVM == -1)],facecolors='none', edgecolors='darkorange', s=100, label="Predicted -1")
plt.plot(x_vals, y_vals_Zero, color='black', linewidth=1.5, label="Decision Boundary for Linear SVC Model where C=0.001")
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Question B - Part III Where C=0.001")
plt.legend(bbox_to_anchor=(1, 0.5))
plt.grid(False)
plt.tight_layout() 
plt.show()

# PLOT C=1
plt.figure(figsize=(12,12))
plt.scatter(X1[y==1], X2[y==1], marker="+", color="lime", label="Target = +1")
plt.scatter(X1[y==-1], X2[y==-1], marker="o", color="blue", label="Target = -1")
plt.scatter(X1_test_SVM[(y_test_SVM == y_pred_One) & (y_test_SVM == 1)], X2_test_SVM[(y_test_SVM == y_pred_One) & (y_test_SVM == 1)],facecolors='none', edgecolors='gold', s=100, label="Predicted +1")
plt.scatter(X1_test_SVM[(y_test_SVM == y_pred_One) & (y_test_SVM == -1)], X2_test_SVM[(y_test_SVM == y_pred_One) & (y_test_SVM == -1)],facecolors='none', edgecolors='yellow', s=100, label="Predicted -1")
plt.plot(x_vals, y_vals_One, color='black', linewidth=1.5, label="Decision Boundary for Linear SVC Model where C=1")
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Question B - Part III Where C=1")
plt.legend(bbox_to_anchor=(1, 0.5))
plt.grid(False)
plt.tight_layout() 
plt.show()

# PLOT C=100
plt.figure(figsize=(12,12))
plt.scatter(X1[y==1], X2[y==1], marker="+", color="lime", label="Target = +1")
plt.scatter(X1[y==-1], X2[y==-1], marker="o", color="blue", label="Target = -1")
plt.scatter(X1_test_SVM[(y_test_SVM == y_pred_Hundred) & (y_test_SVM == 1)], X2_test_SVM[(y_test_SVM == y_pred_Hundred) & (y_test_SVM == 1)],facecolors='none', edgecolors='brown', s=100, label="Predicted +1")
plt.scatter(X1_test_SVM[(y_test_SVM == y_pred_Hundred) & (y_test_SVM == -1)], X2_test_SVM[(y_test_SVM == y_pred_Hundred) & (y_test_SVM == -1)],facecolors='none', edgecolors='fuchsia', s=100, label="Predicted -1")
plt.plot(x_vals, y_vals_Hundred, color='black', linewidth=1.5, label="Decision Boundary for Linear SVC Model where C=100")
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Question B - Part III Where C=100")
plt.legend(bbox_to_anchor=(1, 0.5))
plt.tight_layout() 
plt.grid(False)
plt.show()

# PART IV - SEE REPORT PDF

# QUESTION C

# PART I
X1Square=X1**2
X2Square=X2**2
XSquare=np.column_stack((X1,X2,X1Square,X2Square)) # Stack into 4D array
X_trainSquare,X_testSquare,y_trainSquare,y_testSquare = train_test_split(XSquare,y,test_size=0.2,random_state=0)
X1_test = X_testSquare[:, 0]
X2_test = X_testSquare[:, 1]

classifierSquare=LogisticRegression(random_state=0)
classifierSquare.fit(X_trainSquare,y_trainSquare)
y_predSquare = classifierSquare.predict(X_testSquare)

print("Squared Logistic Regression Model Parameters:")
print("Intercept (b0):", classifierSquare.intercept_[0])
print("Coefficients (b1, b2, b1^2, b2^2):", classifierSquare.coef_[0])

accuracySquare = accuracy_score(y_testSquare, y_predSquare)
print("Test Accuracy:", round(accuracySquare, 7))

# PART II
# Original Data - NOT squared
plt.figure(figsize=(12,12))
plt.scatter(X1[y==1], X2[y==1], marker="+", color="lime", label="Target = +1")
plt.scatter(X1[y==-1], X2[y==-1], marker="o", color="blue", label="Target = -1")

plt.scatter(X1_test[(y_testSquare == y_predSquare) & (y_testSquare == 1)], X2_test[(y_testSquare == y_predSquare) & (y_testSquare == 1)],facecolors='none', edgecolors='teal', s=100, label="Predicted +1")
plt.scatter(X1_test[(y_testSquare == y_predSquare) & (y_testSquare == -1)], X2_test[(y_testSquare == y_predSquare) & (y_testSquare == -1)],facecolors='none', edgecolors='aqua', s=100, label="Predicted -1")
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Question C - Part II - Sqaured Prediction Against Original Values")
plt.legend(bbox_to_anchor=(1, 0.5))
plt.tight_layout() 
plt.grid(False)
plt.show()

# PART III - Add Baseline Comparison
mostCommonClass = Counter(y_trainSquare).most_common(1)[0][0] # Most common class in training set

# Predict most common class
y_predBaseline = np.full_like(y_testSquare,fill_value=mostCommonClass)
baselineAccuracy = accuracy_score(y_testSquare, y_predBaseline)
print("Baseline Accuracy (majority class predictor):", round(baselineAccuracy, 2))
print("Squared Logistic Regression Accuracy:", round(accuracySquare, 2))

# PART IV - Create the decision boundary 
b0c = classifierSquare.intercept_[0]
b1c, b2c, b1csquared, b2csquared = classifierSquare.coef_[0]
x1_range = np.linspace(X1.min() - 1, X1.max() + 1, 200)
x2_range = np.linspace(X2.min() - 1, X2.max() + 1, 200)
X1_grid, X2_grid = np.meshgrid(x1_range, x2_range)


linearCombination = b0c + b1c*X1_grid + b2c*X2_grid + b1csquared*X1_grid**2 + b2csquared*X2_grid**2
prob = 1 / (1 + np.exp(-linearCombination))
plt.figure(figsize=(12,12))
plt.scatter(X1[y==1], X2[y==1], marker="+", color="lime", label="Target = +1")
plt.scatter(X1_test[(y_testSquare == y_predSquare) & (y_testSquare == 1)], X2_test[(y_testSquare == y_predSquare) & (y_testSquare == 1)],facecolors='none', edgecolors='teal', s=100, label="Predicted +1")
plt.scatter(X1_test[(y_testSquare == y_predSquare) & (y_testSquare == -1)], X2_test[(y_testSquare == y_predSquare) & (y_testSquare == -1)],facecolors='none', edgecolors='aqua', s=100, label="Predicted -1")
plt.scatter(X1[y==-1], X2[y==-1], marker="o", color="blue", label="Target = -1")
plt.contour(X1_grid, X2_grid, prob, levels=[0.5], colors='black', linewidths=2)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Question C - Part IV - Squared Logistic Regression Decision Boundary')
plt.legend(bbox_to_anchor=(1, 0.5))
plt.tight_layout() 
plt.grid(False)
plt.show()