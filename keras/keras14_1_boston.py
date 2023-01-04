"""
 ***실습 평가지표 : R2,RMSE***
train 0.7, R2 : 0.8 이상, RMSE 사용



"""

from sklearn.datasets import load_boston
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dropout


# data

dataset = load_boston() # boston housing price
x=dataset.data
y=dataset.target

# print(x)
# print(y)
print(x.shape) #(506, 13)
print(y.shape) # (506,)
# print(dataset.feature_names) # crim, tax, indus... 등
# print(dataset.DESCR)


x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size=0.7,
    shuffle=True,
    random_state=3
    )








#2. 모델 구성

model=Sequential()
model.add(Dense(10,input_dim=13))
model.add(Dense(32))
model.add(Dropout(rate=0.5))
model.add(Dense(32))
model.add(Dropout(rate=0.5))
model.add(Dense(32))
model.add(Dropout(rate=0.5))
model.add(Dense(32))
model.add(Dense(1))



#3. 컴파일 훈련

model.compile(loss='mae',optimizer='adam',metrics=['mae'])
model.fit(x_train,y_train,batch_size=10,epochs=500)



#4. 평가 예측



loss=model.evaluate(x_test,y_test)

y_predict=model.predict(x_test)

# print("=====================")
# print(y_test)
# print('=====================')
# print(y_predict)
# print('=====================')


from sklearn.metrics import mean_squared_error
def RMSE(y_test,y_predict):
        return np.sqrt(mean_squared_error(y_test,y_predict))

print("RMSE : ",RMSE(y_test,y_predict))


from sklearn.metrics import r2_score
R2 = r2_score(y_test,y_predict)
print("R2 : ",R2)


