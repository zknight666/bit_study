import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
import pandas as pd

#1. data 처리 (불러오기, 전처리, )

# x_train = np.array(range(1,11))
# y_train = np.array(range(1,11))
# x_test = np.array([11,12,13])
# y_test = np.array([11,12,13])
# x_validation = np.array([14,15,16])
# y_validation = np.array([14,15,16])

x=np.array(range(1,17))
y=np.array(range(1,17))
#실습 자르기

print(x.shape)
print(y.shape)
"""
x_val=x[:8]
y_val=y[:8]

print(x_val)
print(y_val)
"""

x_train,x_test,y_train,y_test=train_test_split(
    x,y,
    train_size=0.8,
    random_state=1    
)

print(x_train.shape,x_test.shape) # (12,) (4,)
print(y_train.shape,y_test.shape) # (12,) (4,)

# x_test,x_val,y_test,y_val=train_test_split(
#     x_test,y_test,
#     train_size=0.5,
#     random_state=1    
# )



#2. 모델 구성
model=Sequential()
model.add(Dense(50,input_dim=1))
model.add(Dense(550,activation='relu'))
model.add(Dense(1))




# 컴파일, 훈련


model.compile(
    loss='mse',
    optimizer='adam',
    metrics=['accuracy']
    )
"""
model.fit(
    x,y,
    epochs=5,
    validation_data=(x_val,y_val)
)
"""

model.fit(
    x_train,y_train,
    epochs=100,
    batch_size=4,
    validation_split=0.25)




#4. 평가, 예측
# loss=model.evaluate(x,y)
# print('loss : ',loss)

loss=model.evaluate(x_test,y_test)
print('loss : ',loss)

result=model.predict([17])
print('17 예측 값 : ',result)




