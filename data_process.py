import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.impute import SimpleImputer
dataset=pd.read_csv('Data/Data.csv')
print(dataset)
# x=dataset.iloc[:,:-1].values
# y=dataset.iloc[:,-1].values
# print(y)
# take care of missing dataset
mean_age=dataset['Age'].mean()
dataset['Age']=dataset['Age'].fillna(mean_age)
# print(dataset)
mean_salary=dataset['Salary'].mean()
dataset['Salary']=dataset['Salary'].fillna(mean_salary)
# print(dataset) 
# ....................encoding data
# Define category labels
category_labels = {'France': 1, 'Germany': 2, 'Spain': 3}

# Map category labels to numerical values in the 'Country' column
dataset['Country'] = dataset['Country'].map(category_labels)
dataset['Purchased'] = dataset['Purchased'].map({'Yes':1,'No':0})

print(dataset)
# dataset['Country']=factor(dataset['Country'],levels=c('France','Spain','Germany'),labels=c(1,2,3))
# print(dataset)

#.................Now spliliting dataset for training and test set:::::::::::::::  
from sklearn.model_selection import StratifiedShuffleSplit
purchased_data = dataset['Purchased']
train_data, test_data = train_test_split(purchased_data, test_size=0.8, random_state=1)

# Filter the training set for purchases (True values) using boolean indexing
training_set = dataset[dataset['Purchased'] == True]
test_set = dataset[dataset['Purchased'] == False]