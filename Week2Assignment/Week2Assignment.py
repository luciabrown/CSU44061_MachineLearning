# First line of the data file: # id:7-14-7 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC 

df = pd.read_csv("Week2Assignment/week2.php.csv",header=None,comment="#",sep=",",skipinitialspace=True)
print(df.head())
X1=df.iloc[:,0] # Col1
X2=df.iloc[:,1] # Col2
X=np.column_stack((X1,X2)) # Stack into 2D array
y=df.iloc[:,2]  # Target vals

# QUESTION A
# PART I 
plt.figure(figsize=(6,6))
plt.scatter(X1[y==1],X2[y==1],marker="+",color="lime",label="Target = +1")
plt.scatter(X1[y==-1],X2[y==-1],marker="o",color="blue",label="Target = -1")
#plt.xlabel("X1")
#plt.ylabel("X2")
#plt.title("Question A - Part I")
#plt.legend()
#plt.grid(False)
#plt.show()

# PART II - https://youtube.com/shorts/8it0yrJzQfU?si=1UoVpwNGf7flDZeh
# Test sizes from 0.1 to 0.95 in steps of 0.5
test_sizes = np.arange(0.1, 1, 0.05)
accuracy_dict = {}
for ts in test_sizes:
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=ts,random_state=0)

    classifier=LogisticRegression(random_state=0)
    classifier.fit(X_train,y_train)
    y_pred=classifier.predict(X_test)

    # Get model parameters
    print("\nTest size: ",round(ts,2))
    print("Intercept (b0):", classifier.intercept_[0])
    print("Coefficients (b1, b2):", classifier.coef_[0])

    for i, coef in enumerate(classifier.coef_[0]):
        effect = "increases" if coef > 0 else "decreases"
        print(f"Feature X{i+1} {effect} the probability of predicting +1 (coefficient={coef:.3f})")

    cm=confusion_matrix(y_test,y_pred)
    print("Confusion Matrix:\n", cm)
    print("Accuracy:", round(accuracy_score(y_test, y_pred),2))
    accuracy_dict[round(ts, 2)] = round(accuracy_score(y_test, y_pred),2)
print("\nAccuracy Dictionary",accuracy_dict)

# PART III - https://youtu.be/ZsM2z0pTbnk?si=iMnwQTl_QFWL7ani

# Use prediction on full X dataset, not just X_test
y_pred_full = classifier.predict(X)

#Add predictions to plot
plt.scatter(X1[(y == y_pred_full) & (y == 1)], X2[(y == y_pred_full) & (y == 1)], facecolors='none', edgecolors='magenta', s=100, label="Predicted +1 (correct)")
plt.scatter(X1[(y == y_pred_full) & (y == -1)], X2[(y == y_pred_full) & (y == -1)], facecolors='none', edgecolors='red', s=100, label="Predicted -1 (correct)")

# Add decision boundary as line on plot
b0 = classifier.intercept_[0]
b1, b2 = classifier.coef_[0]
x_vals = np.linspace(X1.min()-0.1, X1.max()+0.1, 200)
y_vals = -(b0 + b1 * x_vals) / b2
plt.plot(x_vals, y_vals, color='black', linewidth=1.5, label="Decision Boundary")

plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Question A")
plt.legend(bbox_to_anchor=(1, 0.5))
plt.tight_layout() 
plt.grid(False)
plt.show()

# PART IV - SEE REPORT PDF

# QUESTION B

# PART I - https://youtu.be/kPkwf1x7zpU?si=z3680NVTyFd0b_h4
svcScoreDictionary={}
cVals=[0.001,1,100]

for cVal in cVals:
    model = SVC(C=cVal)
    model.fit(X_train,y_train)
    svcScoreDictionary[round(cVal,3)] = round(model.score(X_test,y_test),2)
print("Scores for SVC:", svcScoreDictionary)