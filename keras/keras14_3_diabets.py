# 실습 r2기준 : 0.62
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dropout




from sklearn.datasets import load_diabetes

#1. data



datasets=load_diabetes()
x=datasets.data
y=datasets.target

print(x)
print(y)
print(x.shape)
print(y.shape)




x_train,x_test,y_train,y_test=train_test_split(
    x,y,
    train_size=0.7,
    random_state=1
)

#2. model


model=Sequential()
model.add(Dense(10,input_dim=10, activation='relu'))
# model.add(Dense(10))
# model.add(Dropout(rate=0.5))
# model.add(Dense(30))
# model.add(Dropout(rate=0.5))
# model.add(Dropout(rate=0.5))
# model.add(Dense(512))
# model.add(Dropout(rate=0.5))
# model.add(Dense(40))
# model.add(Dropout(rate=0.5))
# model.add(Dense(50))
# model.add(Dense(50))
# model.add(Dense(50))
model.add(Dense(1))

#3. compile, training

model.compile(
    loss='mae',
    optimizer='nadam',
    metrics=['mae']
    )

model.fit(
    x_train,y_train,
    batch_size=1,
    epochs=50
)


#4. 평가, 예측

loss=model.evaluate(x_test,y_test)
#뭘까
print('loss : ',loss)

y_predict=model.predict(x_test)
#뭘까

from sklearn.metrics import mean_squared_error
def RMSE(y_test,y_predict):
        return np.sqrt(mean_squared_error(y_test,y_predict))

print("RMSE : ",RMSE(y_test,y_predict))

from sklearn.metrics import r2_score
R2 = r2_score(y_test,y_predict)
print("R2 : ",R2)




