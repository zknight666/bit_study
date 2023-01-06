import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dropout


#1. data

dataset=fetch_california_housing()

x=dataset.data
y=dataset.target

print(x.shape) # (506, 13)
print(y.shape)



x_train,x_test,y_train,y_test=train_test_split(
    x,y,
    train_size=0.8,
    random_state=1,
    shuffle=True
)



#2. model

model=Sequential()
model.add(Dense(32,input_dim=8,activation='relu'))
model.add(Dense(512,activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(1))




#3. compile, training

model.compile(
    optimizer='nadam',
    loss='mae',
    metrics=['accuracy']
)

model.fit(
    x_train,y_train,
    batch_size=16,
    epochs=200,
    validation_split=0.2
)


#4. 평가, 예측

loss=model.evaluate(x_test,y_test)

y_predict=model.predict(x_test)

def RMSE(y_test,y_predict):
    return(np.sqrt(mean_squared_error(y_test,y_predict)))




print("loss",loss,
      'RMSE : ',RMSE(y_test,y_predict),
      'r2 : ',r2_score(y_test,y_predict)
      )


# R2 :  0.6782261757061483 이상 나올 것
"""
결과
loss [0.4580928683280945, 0.0026647287886589766] 
RMSE :  0.643215593730973 
r2 :  0.684585508811475
"""







