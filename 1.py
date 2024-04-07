import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.impute import SimpleImputer
dataset=pd.read_csv('Data/Data.csv')
x=dataset.iloc[:,:-1].values# its indicate [row,column]....[:,:-1]
y=dataset.iloc[:,-1].values
print("____________________x Axes______________")
print(x)
print("____________________Y Axes_______________")
print(y)
# now taking care of missing data... 
imputer = SimpleImputer(missing_values=np.NaN, strategy='mean')
imputer.fit(x[:,1:3])
x[:,1:3]=imputer.transform(x[:,1:3]) 
print(x)
#also using dataframe...use notebook jupyter
'''imputer = SimpleImputer(missing_values=np.NaN, strategy='mean')
x_values = x.values  # Convert DataFrame to numpy array
imputer.fit(x_values[:, 1:3])
x_values[:, 1:3] = imputer.transform(x_values[:, 1:3])
x = pd.DataFrame(x_values, columns=x.columns)  # Convert back to DataFrame
print(x)'''

# 3rd Encodding Start ..............convert to 10 code
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])], remainder='passthrough')
X=np.array(ct.fit_transform(x))
print("Encode independent var.......................")
print(X)
print("Encode dependent var.......................")
le=LabelEncoder()
y=le.fit_transform(y)
print(y)
#Spliting Dataset into training set and test set ------->feature scaling
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)
print(x_train)
print(x_test)
print(y_train)
print(y_test)
#              feature scaling
print(" feature scaling")
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train[:,3:]=sc.fit_transform(x_train[:,3:])
x_test[:,3:]=sc.transform(x_test[:,3:])
print(x_train)
print(x_test)
#            Data preprocessing 