from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.metrics import r2_score, mean_squared_error, accuracy_score

import numpy as np
import pandas as pd


x=np.array([1,2,3]) 
y=np.array([1,2,3])

print(x) #[1 2 3]
print(y) #[1 2 3]
print(x.shape) #(3,)
print(y.shape) #(3,)







model=Sequential()
model.add(Dense(5,input_dim=1))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))




model.summary() # param value = dense 1 * dense2 + bias(*dense2)(좌우로 움직이는 값)
"""
Total params: 17,767
Trainable params: 17,767
Non-trainable params: 0
다른 사람것 모델 빌려쓸때 참고용으로 사용

"""


