import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from apyori import apriori 

dataset=pd.read_csv('F:\git\Mechine_learning\\apriori\Market_Basket_Optimisation.csv',header=None)
transections=[]

for i in range(0,7501):
    transections.append([str(dataset.values[i,j]) for j in range(0,20)])
# print(transection)

# from apyori import apriori
rules = apriori(transactions = transections, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2, max_length = 2)
# print(results)
results = list(rules)

def inspect(results):
    lhs         =   [tuple(result[2][0][0])[0] for result in results]
    rhs         =   [tuple(result[2][0][1])[0] for result in results]
    support     =   [result[1] for result in results]
    confidence  =   [result[2][0][2] for result in results]
    lifts       =   [result[2][0][3] for result in results]
    return list(zip(lhs,rhs,support,confidence,lifts))
result_in_dataframe=pd.DataFrame(inspect(results), columns = ['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])
print(result_in_dataframe)