# In your assignments and projects, unless otherwise stated it
# is mandatory to present cross-validation analysis to support
# your choice of hyperparameter values.
# reasonable baseline too

# This assignment includes: 
# feature selection, model selection, model training and evaluation. Not all datasets
# are useful, e.g. sometimes the data measured fails to capture the important relationships
# or is just too noisy. You now have the tools to analyse the data to uncover such problems.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score,f1_score, classification_report, confusion_matrix, roc_curve, auc,ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier

# Preprocessing - splitting the datasets apart
# ID of dataset_1.csv: # id:23--46--23-0,,
# ID of dataset_2.csv: # id:23-23--23-0,,
with open("week4.php.txt", "r") as f:
    lines = f.readlines()
datasets = []
current = []
for line in lines:
    line = line.strip()
    if line.startswith("#"):
        if current:
            datasets.append(pd.DataFrame([x.split(",") for x in current]))
            current = []
        current.append(line)
    else:
        current.append(line)
if current:
    datasets.append(pd.DataFrame([x.split(",") for x in current]))
for i, df in enumerate(datasets):
    print(f"\nDataset {i+1}:")
    print(df)
    df.to_csv(f"dataset_{i+1}.csv", index=False, header=False)

# -------------------------------- QUESTION A -----------------------------------------------------#
# Scatter plot of the two classes
dataset1_df = pd.read_csv("dataset_1.csv",header=None,comment="#",sep=",",skipinitialspace=True)
dataset1_X1=dataset1_df.iloc[:,0] # Col1
dataset1_X2=dataset1_df.iloc[:,1] # Col2
dataset1_X=np.column_stack((dataset1_X1,dataset1_X2)) # Stack into 2D array
dataset1_y=dataset1_df.iloc[:,2]  # Target vals

dataset2_df = pd.read_csv("dataset_2.csv",header=None,comment="#",sep=",",skipinitialspace=True)
dataset2_X1=dataset2_df.iloc[:,0] # Col1
dataset2_X2=dataset2_df.iloc[:,1] # Col2
dataset2_X=np.column_stack((dataset2_X1,dataset2_X2)) # Stack into 2D array
dataset2_y=dataset2_df.iloc[:,2]  # Target vals

plt.figure(figsize=(12,12))
plt.scatter(dataset1_X1[dataset1_y==1],dataset1_X2[dataset1_y==1],marker="o",color="blue",label="Target = +1")
plt.scatter(dataset1_X1[dataset1_y==-1],dataset1_X2[dataset1_y==-1],marker="o",color="red",label="Target = -1")
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Question A - Source Data for Dataset 1')
plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0.)
plt.grid(False)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,12))
plt.scatter(dataset2_X1[dataset2_y==1],dataset2_X2[dataset2_y==1],marker="o",color="blue",label="Target = +1")
plt.scatter(dataset2_X1[dataset2_y==-1],dataset2_X2[dataset2_y==-1],marker="o",color="red",label="Target = -1")
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Question A - Source Data for Dataset 2')
plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0.)
plt.grid(False)
plt.tight_layout()
plt.show()

# Generate polynomial features & C values for both logistic models
fiveFoldPolynomialTest = KFold(n_splits=5, shuffle=True, random_state=1)

# Define grid of hyperparameters
degrees = list(range(1,7,1))  # test polynomial degrees
cVals = np.logspace(-3, 3, 10)  # test C values

# To store results
meanScores1 = np.zeros((len(degrees), len(cVals)))
stdScores1  = np.zeros((len(degrees), len(cVals)))
meanScores2 = np.zeros((len(degrees), len(cVals)))
stdScores2  = np.zeros((len(degrees), len(cVals)))

# Cross-validate
for i, degree in enumerate(degrees):
    for j, c in enumerate(cVals):
        # Build pipeline
        model = Pipeline([
            ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
            ('scaler', StandardScaler()),
            ('logreg', LogisticRegression(penalty='l2', C=c, solver='lbfgs', max_iter=10000))
        ])
        
        # Evaluate 5-fold for both datasets
        scores1 = cross_val_score(model, dataset1_X, dataset1_y, cv=fiveFoldPolynomialTest, scoring='accuracy')
        scores2 = cross_val_score(model, dataset2_X, dataset2_y, cv=fiveFoldPolynomialTest, scoring='accuracy')
        meanScores1[i, j] = scores1.mean()
        stdScores1[i, j] = scores1.std()
        meanScores2[i, j] = scores2.mean()
        stdScores2[i, j] = scores2.std()

# Find best degree and C for both datasets
bestIndex1 = np.unravel_index(np.argmax(meanScores1), meanScores1.shape)
bestDegree1 = degrees[bestIndex1[0]]
bestCVal1 = cVals[bestIndex1[1]]
print(f"Best polynomial degree for Dataset 1: {bestDegree1}")
print(f"Best C value for Dataset 1: {bestCVal1:.4f}")

bestIndex2 = np.unravel_index(np.argmax(meanScores2), meanScores2.shape)
bestDegree2 = degrees[bestIndex2[0]]
bestCVal2 = cVals[bestIndex2[1]]
print(f"Best polynomial degree for Dataset 2: {bestDegree2}")
print(f"Best C value for Dataset 2: {bestCVal2:.4f}\n")

# Plot for Dataset 1
plt.figure(figsize=(8,6))
for i, degree in enumerate(degrees):
    plt.errorbar(cVals, meanScores1[i], yerr=stdScores1[i],fmt='-o', capsize=4, label=f'Degree {degree}')
plt.xscale('log')
plt.xlabel('C')
plt.ylabel('Mean 5-Fold Accuracy')
plt.title('Question A - Cross-Validation for Dataset 1 - Maximum Degree Denoted as Red Circle with Black Border - Best C Val is X-coordinate of this Maximum')
plt.legend()
plt.grid(True, which='both', ls='--', lw=0.5)

# Mark the best point
plt.scatter(cVals[np.argmax(meanScores1[bestIndex1[0]])],meanScores1[bestIndex1], color='red', edgecolor='black', linewidth=2, s=80, zorder=5, label='Best Point')
plt.show()

# Plot for Dataset 2
plt.figure(figsize=(8,6))
for i, degree in enumerate(degrees):
    plt.errorbar(cVals, meanScores2[i], yerr=stdScores2[i],fmt='-o', capsize=4, label=f'Degree {degree}')
plt.xscale('log')
plt.xlabel('C')
plt.ylabel('Mean 5-Fold Accuracy')
plt.title('Question A - Cross-Validation for Dataset 2 - Maximum Degree Denoted as Red Circle with Black Border - Best C Val is X-coordinate of this Maximum')
plt.legend()
plt.grid(True, which='both', ls='--', lw=0.5)

# Mark the best point
plt.scatter(cVals[np.argmax(meanScores2[bestIndex2[0]])],meanScores2[bestIndex2], color='red', edgecolor='black', linewidth=2, s=80, zorder=5, label='Best Point')
plt.show()

# Train/test split
# Dataset 1
X_train1, X_test1, y_train1, y_test1 = train_test_split(dataset1_X, dataset1_y, test_size=0.2, random_state=42)
# Dataset 2
X_train2, X_test2, y_train2, y_test2 = train_test_split(dataset2_X, dataset2_y, test_size=0.2, random_state=42)

# Scaling & Logistical Regression model fits using the best degree and best cval
model1Logistical = Pipeline([
    ('poly', PolynomialFeatures(degree=bestDegree1, include_bias=False)),
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(penalty='l2', C=bestCVal1, solver='lbfgs', max_iter=10000))
])
model1Logistical.fit(X_train1, y_train1)

model2Logistical = Pipeline([
    ('poly', PolynomialFeatures(degree=bestDegree2, include_bias=False)),
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(penalty='l2', C=bestCVal2, solver='lbfgs', max_iter=10000))
])
model2Logistical.fit(X_train2, y_train2)

# Baselines

# for dataset 1 I am choosing the baseline of always selecting the modal result - this is because the -1's heavily outweight the +1's in this dataset
baseline1 = DummyClassifier(strategy='most_frequent')
baseline1.fit(X_train1, y_train1)
y_pred_baseline1 = baseline1.predict(X_test1)
baseline_acc1 = accuracy_score(y_test1, y_pred_baseline1)
print(f"\nDataset 1 — Baseline Accuracy (modal class): {baseline_acc1:.3f}")

# for dataset 2 the result of -1 and +1 seems approximately equal, irrespective of X1 and X2 - I am choosing the startified baseline which chooses randomly based on the training data
baseline2 = DummyClassifier(strategy='stratified', random_state=42)
baseline2.fit(X_train2, y_train2)
y_pred_baseline2 = baseline2.predict(X_test2)
baseline_acc2 = accuracy_score(y_test2, y_pred_baseline2)
print(f"Dataset 2 — Baseline Accuracy (stratified): {baseline_acc2:.3f}\n")

# Helper function for plotting models vs. their baseline
def modelVsBaselinePlot(model, baseline, X, y, title):
    # Meshgrid for decision boundary
    x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
    y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    # Predictions
    Z_model = model.predict(grid).reshape(xx.shape)
    Z_base  = baseline.predict(grid).reshape(xx.shape)    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14,6))
    
    # Left: Logistic Regression or KNN
    # Decision boundary
    axes[0].contour(xx, yy, Z_model, levels=[0], colors='black', linewidths=2)
    # Training data
    axes[0].scatter(X[y==1,0], X[y==1,1], color='blue', edgecolor='k', label='Class +1')
    axes[0].scatter(X[y==-1,0], X[y==-1,1], color='red', edgecolor='k', label='Class -1')
    axes[0].set_xlabel('X1')
    axes[0].set_ylabel('X2')
    axes[0].set_title("Machine Learning Model")
    # Legend
    decisionBoundary = Line2D([0], [0], color='black', lw=2, label='Decision Boundary')
    axes[0].legend(handles=[mpatches.Patch(color='blue', label='Class +1'),mpatches.Patch(color='red', label='Class -1'),decisionBoundary], loc='best')
    
    # Right: Baseline Classifier
    axes[1].contourf(xx, yy, Z_base, alpha=0.2, levels=np.linspace(-1, 1, 3), colors=["red", "blue"])
    # Training data
    axes[1].scatter(X[y==1,0], X[y==1,1], color='blue', edgecolor='k', label='Class +1')
    axes[1].scatter(X[y==-1,0], X[y==-1,1], color='red', edgecolor='k', label='Class -1')
    axes[1].set_xlabel('X1')
    axes[1].set_ylabel('X2')
    axes[1].set_title("Baseline Model")
    # Legend
    background = Line2D([0], [0], linestyle='None', label='Background = Baseline Prediction')
    axes[1].legend(handles=[mpatches.Patch(color='blue', label='Class +1'),mpatches.Patch(color='red', label='Class -1'),background], loc='best')
    
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

modelVsBaselinePlot(model1Logistical, baseline1, X_train1, y_train1,"Dataset 1: Logistic Regression vs Modal Baseline (Modal Result)")
modelVsBaselinePlot(model2Logistical, baseline2, X_train2, y_train2,"Dataset 2: Logistic Regression vs Modal Baseline (Randomised Result)")

# Helper function for the predictions
def predictAndMetrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    print(f"Accuracy: {accuracy:.3f}")
    print(f"F1_macro: {f1:.3f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    return y_pred

# Helper function for plotting predictions
def plotPredictions(model, X_train, y_train, X_test, y_test, title):
    X1_train, X2_train = X_train[:,0], X_train[:,1]
    X1_test, X2_test = X_test[:,0], X_test[:,1]
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Scatter plot of training data
    plt.figure(figsize=(12,12))
    plt.scatter(X1_train[y_train==1], X2_train[y_train==1], marker="o", color="blue", label="Target = +1")
    plt.scatter(X1_train[y_train==-1], X2_train[y_train==-1], marker="o", color="red", label="Target = -1")
    
    # Overlay predictions
    plt.scatter(X1_test[(y_test == y_pred) & (y_test == 1)], X2_test[(y_test == y_pred) & (y_test == 1)], facecolors='none', edgecolors='darkgreen', s=100, label="Predicted +1 Correct")
    plt.scatter(X1_test[(y_test == y_pred) & (y_test == -1)], X2_test[(y_test == y_pred) & (y_test == -1)], facecolors='none', edgecolors='fuchsia', s=100, label="Predicted -1 Correct")
    plt.scatter(X1_test[(y_test != y_pred) & (y_pred == 1)], X2_test[(y_test != y_pred) & (y_pred == 1)], marker="x", color='black', s=100, label="Predicted +1 Wrong")
    plt.scatter(X1_test[(y_test != y_pred) & (y_pred == -1)], X2_test[(y_test != y_pred) & (y_pred == -1)], marker="x", color='black', s=100, label="Predicted -1 Wrong")
    
    # Decision boundary
    x_min, x_max = X_train[:,0].min()-0.5, X_train[:,0].max()+0.5
    y_min, y_max = X_train[:,1].min()-0.5, X_train[:,1].max()+0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid).reshape(xx.shape)
    plt.contour(xx, yy, Z, levels=[0], colors='black', linewidths=1.5, label="Decision Boundary")
    
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title(title)
    plt.legend(bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.grid(False)
    plt.show()

# Dataset 1 prediction plot
plotPredictions(model1Logistical, X_train1, y_train1, X_test1, y_test1, "Dataset 1 - Logistic Regression Predictions with Decision Boundary")
# Dataset 2 prediction plot
plotPredictions(model2Logistical, X_train2, y_train2, X_test2, y_test2, "Dataset 2 - Logistic Regression Predictions with Decision Boundary")

print("Dataset 1 — Logistic Regression Performance:")
predictAndMetrics(model1Logistical, X_test1, y_test1)
print("Dataset 2 — Logistic Regression Performance:")
predictAndMetrics(model2Logistical, X_test2, y_test2)

# Print coefficients for the model that was chosen
def getParams(model,num):
    poly = model.named_steps['poly']
    params = model.named_steps['logreg']
    
    feature_names = poly.get_feature_names_out(['X1', 'X2'])
    coefs = params.coef_.flatten()
    intercept = params.intercept_[0]
    
    print(f"\nLOGISTIC REGRESSION PARAMETERS FOR DATASET {num} -  Best C={params.C} & Best Degree={poly.degree}")
    print(f"Intercept: {intercept:.6f}")
    for name, coef in zip(feature_names, coefs):
        print(f"{name}: {coef:.6f}")

# Print for both models
getParams(model1Logistical,"1")
getParams(model2Logistical,"2")

# -------------------------------- QUESTION B-----------------------------------------------------#
kVals=list(range(3,70,2)) # really high to avoid early convergence

# To store results
meanKScores1 = np.zeros(len(kVals))
stdKScores1  = np.zeros(len(kVals))
meanKScores2 = np.zeros(len(kVals))
stdKScores2  = np.zeros(len(kVals))

# Cross-validate
for i, k in enumerate(kVals):
    # Build pipeline
    knnPipeline = Pipeline([
        ('scaler',StandardScaler()),
        ('knn',KNeighborsClassifier(n_neighbors=k))
    ])

    # Evaluate 5-fold for both datasets
    kScores1 = cross_val_score(knnPipeline, dataset1_X, dataset1_y, cv=fiveFoldPolynomialTest, scoring='accuracy')
    kScores2 = cross_val_score(knnPipeline, dataset2_X, dataset2_y, cv=fiveFoldPolynomialTest, scoring='accuracy')
    meanKScores1[i] = kScores1.mean()
    stdKScores1[i] = kScores1.std()
    meanKScores2[i] = kScores2.mean()
    stdKScores2[i] = kScores2.std()

# Find best K both datasets
bestKVal1 = kVals[np.argmax(meanKScores1)]
print("\nBest K value for Dataset 1: ",bestKVal1)

bestKVal2 = kVals[np.argmax(meanKScores2)]
print("Best K value for Dataset 2: ",bestKVal2)

# Plot for Dataset 1
plt.figure(figsize=(8,6))
for i, k in enumerate(kVals):
    plt.errorbar(kVals, meanKScores1, yerr=stdKScores1,fmt='-o', capsize=4, label=f'K {k}')
plt.xlabel('K')
plt.ylabel('Mean 5-Fold Accuracy')
plt.title('Question B - Cross-Validation for Dataset 1 - Maximum KValue Denoted as Pink Circle with Green Border')
plt.grid(True, which='both', ls='--', lw=0.5)
# Mark the best point
bestKIndex1=np.argmax(meanKScores1)
plt.scatter(kVals[bestKIndex1],meanKScores1[bestKIndex1], color='pink', edgecolor='lime', linewidth=2, s=80, zorder=5, label='Best K Value')
plt.show()

# Plot for Dataset 2
plt.figure(figsize=(8,6))
for i, k in enumerate(kVals):
    plt.errorbar(kVals, meanKScores2, yerr=stdKScores2,fmt='-o', capsize=4, label=f'K {k}')
plt.xlabel('K')
plt.ylabel('Mean 5-Fold Accuracy')
plt.title('Question B - Cross-Validation for Dataset 2 - Maximum KValue Denoted as Pink Circle with Green Border')
plt.grid(True, which='both', ls='--', lw=0.5)
# Mark the best point
bestKIndex2=np.argmax(meanKScores2)
plt.scatter(kVals[bestKIndex2],meanKScores2[bestKIndex2], color='pink', edgecolor='lime', linewidth=2, s=80, zorder=5, label='Best K Value')
plt.show()

# Scaling & K Nearest Neighbours model fits using the best kval
model1K = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=bestKVal1))
])
model1K.fit(X_train1, y_train1)

model2K = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=bestKVal2))
])
model2K.fit(X_train2, y_train2)
modelVsBaselinePlot(model1K, baseline1, X_train1, y_train1,"Dataset 1: K Nearest Neighbours vs Modal Baseline (Modal Result)")
modelVsBaselinePlot(model2K, baseline2, X_train2, y_train2,"Dataset 2: K Nearest Neighbours vs Modal Baseline (Randomised Result)")
# Dataset 1 prediction plot
plotPredictions(model1K, X_train1, y_train1, X_test1, y_test1, "Dataset 1 - K Nearest Neighbours Predictions with Decision Boundary")
# Dataset 2 prediction plot
plotPredictions(model2K, X_train2, y_train2, X_test2, y_test2, "Dataset 2 - K Nearest Neighbours Predictions with Decision Boundary")
print("Dataset 1 — K Nearest Neighbours Performance:")
predictAndMetrics(model1K, X_test1, y_test1)
print("Dataset 2 — K Nearest Neighbours Performance:")
predictAndMetrics(model2K, X_test2, y_test2)

# -------------------------------- QUESTION C -----------------------------------------------------#
#https://youtu.be/4jRBRDbJemM?si=X4780ygozX_mBXRi
def confusionMatrix(model, X_test, y_test, name):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"\n{name} Performance:")
    print(f"Accuracy: {acc:.3f}")
    print(f"F1_macro: {f1:.3f}")
    
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(cmap="Blues", values_format="d", colorbar=False, ax=ax)

    labels = [["True Negative", "False Positive"],["False Negative", "True Positive"]]
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i + 0.25, labels[i][j], ha='center', va='top', fontsize=9, color='black')

    # Style tweaks
    ax.set_title(f"{name} — Confusion Matrix", pad=12)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    plt.show()

    return cm
    
    return cm

confusionMatrix(model1Logistical,X_test1,y_test1,"CONFUSION MATRIX - Logistical Regression - Trained on the Valid/Non-Noisy Dataset")
confusionMatrix(model1K,X_test1,y_test1,"CONFUSION MATRIX - K Nearest Neighbours - Trained on the Valid/Non-Noisy Dataset")
confusionMatrix(baseline1,X_test1,y_test1,"CONFUSION MATRIX - Baseline 1 - Always Select the Modal Class")

# -------------------------------- QUESTION C -----------------------------------------------------#
def rocCurve(models, X_test, y_test, title="ROC Curves Comparison"):
    plt.figure(figsize=(8, 6))

    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            y_scores = model.predict_proba(X_test)[:, 1]
        else:
            y_scores = model.decision_function(X_test)
            
        fpr, tpr, _ = roc_curve(y_test, y_scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")
    
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, ls='--', lw=0.5)
    plt.show()

models_dataset1 = {
    "Logistic Regression": model1Logistical,
    "K-Nearest Neighbours": model1K,
    "Baseline": baseline1
}
rocCurve(models_dataset1, X_test1, y_test1, title="Dataset 1 — ROC Curves")
models_dataset2 = {
    "Logistic Regression": model2Logistical,
    "K-Nearest Neighbours": model2K,
    "Baseline": baseline2
}
rocCurve(models_dataset2, X_test1, y_test1, title="Dataset 2 — ROC Curves")