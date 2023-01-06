#실습 r2 0.55 ~ 0.6이상
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from tensorflow.keras.layers import Dropout

# 1. data

datasets=fetch_california_housing()
x=datasets.data
y=datasets.target

print(x)
print(x.shape)
print(y)
print(y.shape)
print(datasets.feature_names) # crim, tax, indus... 등
print(datasets.DESCR)


x_train,x_test,y_train,y_test=train_test_split(
    x,y,
    train_size=0.7,
    shuffle=True,
    random_state=1
)

#2. model

model=Sequential()
model.add(Dense(32,input_dim=8))
# model.add(Dropout(rate=0.5))
# model.add(Dropout(rate=0.5))
model.add(Dense(512, activation='relu'))
# model.add(Dropout(rate=0.5))
# model.add(Dropout(rate=0.5))
model.add(Dense(1))



#3. compile, training

model.compile(
    loss='mae',
    optimizer='nadam',
    metrics=['accuracy']
    )

model.fit(
    x_train,y_train,
    batch_size=32,
    epochs=300
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



"""
loss :  [0.5141951441764832, 0.002906507346779108]
RMSE :  0.7408207844308529
R2 :  0.5824876592696644 

loss :  [0.46025535464286804, 0.0027450346387922764]
RMSE :  0.6503600504422481
R2 :  0.6782261757061483

"""
