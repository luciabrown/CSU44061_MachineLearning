# First line of the data file: # id:1--2-1 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Lasso, Ridge
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
ax.set_title(f'Question A - Plotting of the Source Data')
plt.show()

# QUESTION B - https://youtu.be/LmpBt0tenJE?si=zjtD7zzyK3WANI1c

# Add extra polynomial features
poly = PolynomialFeatures(degree=5, include_bias=False)
X_poly = poly.fit_transform(X)

# Test train split
X_train_lasso, X_test_lasso, y_train_lasso, y_test_lasso = train_test_split(X_poly, y, test_size=0.2, random_state=0)

# Scale
scaler = StandardScaler()
X_train_lasso = scaler.fit_transform(X_train_lasso)
X_test_lasso = scaler.transform(X_test_lasso)

modelZeroLasso =Lasso(alpha=1/(2*0.001),random_state=1)
modelOneLasso=Lasso(alpha=1/(2*1),random_state=1)
modelHundredLasso =Lasso(alpha=1/(2*100),random_state=1)

# fit models
lasso_fitted_models = {
    0.001:modelZeroLasso.fit(X_train_lasso, y_train_lasso),
    1:modelOneLasso.fit(X_train_lasso, y_train_lasso),
    100:modelHundredLasso.fit(X_train_lasso, y_train_lasso)
}

# get predictions
y_pred_ZeroLasso=modelZeroLasso.predict(X_test_lasso)
y_pred_OneLasso=modelOneLasso.predict(X_test_lasso)
y_pred_HundredLasso=modelHundredLasso.predict(X_test_lasso)

lasso_predictions = {
    0.001: y_pred_ZeroLasso,
    1: y_pred_OneLasso,
    100: y_pred_HundredLasso
}

for c,prediction in lasso_predictions.items():
    #Metrics
    print("\n----- LASSO METRICS -----")
    mae=mean_absolute_error(y_test_lasso,prediction)
    mse=mean_squared_error(y_test_lasso,prediction)
    r2=r2_score(y_test_lasso,prediction)
    print("Metrics for Alpha/C: ",c)
    print("Mean Absolute Error: ",round(mae,4))
    print("Mean Squared Error: ",round(mse,4))
    print("R2 Score: ",round(r2,4))

for c,models in lasso_fitted_models.items():
    # Model parameters
    print("\n----- LASSO MODEL PARAMETERS -----")
    print("Model Parameters for Alpha/C: ",c)
    intercept = models.intercept_
    coeffs = models.coef_
    feature_names = poly.get_feature_names_out(['X1', 'X2'])
    print("Intercept: ",round(intercept,4))
    for featureName, coefficient in zip(feature_names, coeffs):
        print(f"{featureName}: {coefficient:.4f}")

# Plotting the fitted models/pedictions
grid_x1Lasso = np.linspace(-5, 5, 50)
grid_x2Lasso = np.linspace(-5, 5, 50)
Xtest_plottingLasso = []

for i in grid_x1Lasso:
    for j in grid_x2Lasso:
        Xtest_plottingLasso.append([i, j])
Xtest_plottingLasso = np.array(Xtest_plottingLasso)

# Add extra polynomial features as per the source data
Xtest_poly = poly.transform(Xtest_plottingLasso)
# Scale with the same scaler as training data
Xtest_poly_scaled = scaler.transform(Xtest_poly)

# grid for plotting
grid_predictionsLasso = {}
for C, model in lasso_fitted_models.items():
    y_grid_pred = model.predict(Xtest_poly_scaled)
    # Reshape to match grid for surface plotting
    grid_predictionsLasso[C] = y_grid_pred.reshape(len(grid_x1Lasso), len(grid_x2Lasso))

X1_gridLasso, X2_gridLasso = np.meshgrid(grid_x1Lasso, grid_x2Lasso)

fig = plt.figure(figsize=(18,5))

for idx, (C, y_pred_gridLasso) in enumerate(grid_predictionsLasso.items()):
    ax = fig.add_subplot(1, 3, idx+1, projection='3d')
    
    # Plot Lasso predictions as surface
    ax.plot_surface(X1_gridLasso, X2_gridLasso, y_pred_gridLasso, alpha=0.6, cmap='viridis')
    
    # Plot training data as scatter points
    ax.scatter(X[:,0], X[:,1], y, color='red', label='Training Data')
    
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('y')
    ax.set_title(f'Lasso predictions for C={C}')
    ax.legend()

plt.tight_layout()
plt.show()


# RIDGE REGRESSION MODEL
# train test split
X_train_ridge, X_test_ridge, y_train_ridge, y_test_ridge = train_test_split(X_poly, y, test_size=0.2, random_state=0)

# scale
X_train_ridge = scaler.fit_transform(X_train_ridge)
X_test_ridge = scaler.transform(X_test_ridge)

modelZeroRidge = Ridge(alpha=1/(2*0.001), random_state=1)
modelOneRidge = Ridge(alpha=1/(2*1), random_state=1)
modelHundredRidge = Ridge(alpha=1/(2*100), random_state=1)

# Fit models
ridge_fitted_models = {
    0.001: modelZeroRidge.fit(X_train_ridge, y_train_ridge),
    1: modelOneRidge.fit(X_train_ridge, y_train_ridge),
    100: modelHundredRidge.fit(X_train_ridge, y_train_ridge)
}

# Get predictions 
y_pred_Zero = modelZeroRidge.predict(X_test_ridge)
y_pred_One = modelOneRidge.predict(X_test_ridge)
y_pred_Hundred = modelHundredRidge.predict(X_test_ridge)

ridge_predictions = {
    0.001: y_pred_Zero,
    1: y_pred_One,
    100: y_pred_Hundred
}

# Metrics
for c, prediction in ridge_predictions.items():
    print("\n----- RIDGE METRICS -----")
    mae = mean_absolute_error(y_test_ridge, prediction)
    mse = mean_squared_error(y_test_ridge, prediction)
    r2 = r2_score(y_test_ridge, prediction)
    print("Metrics for Alpha/C:", c)
    print("Mean Absolute Error:", round(mae,4))
    print("Mean Squared Error:", round(mse,4))
    print("R2 Score:", round(r2,4))

# Model parameters
for c, model in ridge_fitted_models.items():
    print("\n----- RIDGE MODEL PARAMETERS -----")
    print("Model Parameters for Alpha/C:", c)
    intercept = model.intercept_
    coeffs = model.coef_
    feature_names = poly.get_feature_names_out(['X1','X2'])
    print("Intercept:", round(intercept,4))
    for fname, coef in zip(feature_names, coeffs):
        print(f"{fname}: {coef:.4f}")

# grid for plotting
grid_x1Ridge = np.linspace(-5,5,50)
grid_x2Ridge = np.linspace(-5,5,50)
Xtest_plottingRidge = np.array([[i,j] for i in grid_x1Ridge for j in grid_x2Ridge])

grid_predictionsRidge = {}
for c, model in ridge_fitted_models.items():
    y_grid_pred = model.predict(Xtest_poly_scaled)
    grid_predictionsRidge[c] = y_grid_pred.reshape(len(grid_x1Ridge), len(grid_x2Ridge))

X1_gridRidge, X2_gridRidge = np.meshgrid(grid_x1Ridge, grid_x2Ridge)

fig = plt.figure(figsize=(18,5))
for idx, (C, y_pred_gridRidge) in enumerate(grid_predictionsRidge.items()):
    ax = fig.add_subplot(1, 3, idx+1, projection='3d')
    ax.plot_surface(X1_gridRidge, X2_gridRidge, y_pred_gridRidge, alpha=0.6, cmap='viridis')
    ax.scatter(X[:,0], X[:,1], y, color='red', label='Training Data')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('y')
    ax.set_title(f'Ridge predictions for C={C}')
    ax.legend()

plt.tight_layout()
plt.show()