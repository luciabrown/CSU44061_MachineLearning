# First line of the data file: # id:7-14-7 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
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


# QUESTION B