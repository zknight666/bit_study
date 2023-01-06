import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import pandas as pd



#1. data
"""
pd.read_csv
"""

path='./_data/ddarung/'
train_csv=pd.read_csv(path+'train.csv',index_col=[0,9])
test_csv=pd.read_csv(path+'test.csv',index_col=0)
submission_csv=pd.read_csv(path+'submission.csv',index_col=0)

print(train_csv)
print(test_csv)
print(submission_csv)

print(train_csv.shape) # (1459, 9)
print(test_csv.shape) # (715, 9)
print(submission_csv.shape) # (715, 1)

print(train_csv.columns) # column 확인





