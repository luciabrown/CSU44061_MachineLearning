# First line of the data file: # id:1--2-1 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("Week3Assignment/week3.php.csv",header=None,comment="#",sep=",",skipinitialspace=True)
X1=df.iloc[:,0] # Col1
X2=df.iloc[:,1] # Col2
X=np.column_stack((X1,X2)) # Stack into 2D array
y=df.iloc[:,2]  # Target vals

# QUESTION A
fig = plt.figure()
ax = fig.add_subplot(111,projection="3d")
ax.scatter(X[:,0],X[:,1],y, c=X[:,0], cmap='cool', marker='o')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Target (y)')
plt.show()

# QUESTION B - https://youtu.be/LmpBt0tenJE?si=zjtD7zzyK3WANI1c

# Add extra polynomial features
poly = PolynomialFeatures(degree=5, include_bias=False)
X_poly = poly.fit_transform(X)

#Train-test split
X_train,X_test,y_train,y_test = train_test_split(X_poly,y,test_size=0.2,random_state=1)

#Scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Choose from range of values for C/alpha
cVals = [1,10,100,1000]

for c in cVals:
    lasso=Lasso(alpha=1/c,random_state=1)
    lasso.fit(X_train,y_train) # fit lasso to training data
    y_pred = lasso.predict(X_test)

    #Metrics
    mae=mean_absolute_error(y_test,y_pred)
    mse=mean_squared_error(y_test,y_pred)
    r2=r2_score(y_test,y_pred)
    print("\nAlpha/C: ",c)
    print("Mean Absolute Error: ",round(mae,4))
    print("Mean Squared Error: ",round(mse,4))
    print("R2 Score: ",round(r2,4))

    # Model parameters
    intercept = lasso.intercept_
    coeffs = lasso.coef_
    feature_names = poly.get_feature_names_out(['X1', 'X2'])
    print("Intercept: ",round(intercept,4))
    for featureName, coefficient in zip(feature_names, coeffs):
        print(f"{featureName}: {coefficient:.4f}")

# QUESTION C