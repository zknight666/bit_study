import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

from sklearn.model_selection import train_test_split 

#1. data

x=np.array([range(1,11)]) #(1,10)
y=np.array([range(10)]) #(1,10)

x=x.T
y=y.T

x_train, x_test, y_train, y_test = train_test_split(
    x,y,train_size=0.7, 
    # test_size=0.3, 
    shuffle=True, 
    random_state=1 #사용하면 해당 난수 고정, 사용 안하면 완전 난수 랜덤
    )

"""
x_train=x[0:-3]
x_test=x[-3:]
y_train=y[:7]
y_test=y[7:]
"""


#실습 : 넘파이 리스트 슬라이싱 7:3으로 잘라라
#실습 : train과 test를 섞어서 7:3으로 만들기 사이킷런
# 전체 범위를 train data로 설정, 전체 범위 내 부분적으로 test data로 설정 (정확도 최대화, loss값 최소화)
#test data random하게 빼는게 좋음

print('x_train : ',x_train)
print('x_test : ',x_test)
print('y_train : ',y_train)
print('y_test : ',y_test)


print(x.shape) 
print(y.shape) 

#2. model

model=Sequential()
model.add(Dense(3,input_dim=1))
model.add(Dense(10))
model.add(Dense(15))
model.add(Dense(20))
model.add(Dense(1))

#3. compile, training

model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=90,batch_size=2)
# train용 data로 fit

#4. validation, predict

loss=model.evaluate(x_test,y_test)
# 평가용 data로 evaluate
print('loss :',loss)
result=model.predict([15])
print('predict :',result)

