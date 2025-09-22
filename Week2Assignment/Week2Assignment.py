# First line of the data file: # id:7-14-7 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
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
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Question A - Part I")
plt.legend()
plt.grid(False)
plt.show()

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
# QUESTION B