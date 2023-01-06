from sklearn.datasets import load_boston
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

#1. data

dataset = load_boston() # boston housing price
x=dataset.data
y=dataset.target

print(x.shape) #(506, 13)
print(y.shape) #(506, 13)


x_train,x_test,y_train,y_test=train_test_split(
    x,y,
    train_size=0.8,
    random_state=1,
    shuffle=True
)



#2. model


model=Sequential()
model.add(Dense(32,input_dim=13))
model.add(Dense(512,activation='relu'))
model.add(Dense(1))

#3. compile, training

model.compile(
    optimizer='adam',
    loss='mae',
    metrics=['accuracy']
)

model.fit(
    x_train,y_train,
    batch_size=8,
    epochs=150,
    validation_split=0.2
)


#4. 평가, 예측
"""
loss=model.evaluate
model.predict
sklearn, mae
def RMSE np sqrt
"""

loss=model.evaluate(x_test,y_test)

y_predict=model.predict(x_test)

def RMSE(y_test,y_predict):
    return(np.sqrt(mean_squared_error(y_test,y_predict)))

print("loss : ",loss)
print("RMSE : ",RMSE(y_test,y_predict))
print('r2 : ',r2_score(y_test,y_predict))

# R2값 0.8 이상

"""
결과
loss :  [2.902878761291504, 0.0]
RMSE :  4.0858269073324704
r2 :  0.8310794041312348
"""
