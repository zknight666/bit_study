import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


# data 전처리 ()
"""
파일 불러오기 : pd.read

"""
path= './_data/bike/'
train_csv=pd.read_csv(path+'train.csv',index_col=[0,9,10])
test_csv=pd.read_csv(path+'test.csv',index_col=0)
submission_csv=pd.read_csv(path+'submission.csv',index_col=0)

"""
열 확인
"""
print(train_csv)
print(test_csv)






