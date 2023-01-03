import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

#1.데이터

x=np.array([range(10)])  #input 1개


y=np.array([[1,2,3,4,5,6,7,8,9,10], 
           [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4],
           [9,8,7,6,5,4,3,2,1,0]]) # , output 3개



print(x.shape) #(1,10)
print(y.shape) #(3,10)
x=x.T
y=y.T
print(x.shape) #(10,1)
print(y.shape) #(10,3)
"""
input 1개 / output 3개
행 무시, 열 우선
"""

model=Sequential()
model.add(Dense(5,input_dim=1)) #input 1개
model.add(Dense(10))
model.add(Dense(15))
model.add(Dense(20))
model.add(Dense(15))
model.add(Dense(3)) # output 3개

model.compile(loss='mse',optimizer='adam')
model.fit(x,y,epochs=70,batch_size=2)

""" 
metrics=['accuracy']
validation data=(x,y)
데이터 분리 -> 훈련(train set) & 평가(test set)


"""


loss=model.evaluate(x,y)
print('loss : ',loss)

result = model.predict([[9]])
print('predict : ',result)


"""
결과 :
predict :  [[9.850217   1.5213985  0.03505665]]
predict :  [[10.009439    1.6379923  -0.01246876]]
"""



