from sklearn import tree
import pandas as pd
import numpy as np

data=pd.read_csv("Q1/BreastTissue.csv")
d=data.as_matrix()
d=d[:,1:]

d1=pd.read_csv("Q3&Q4/magic_train.csv")
print ''
