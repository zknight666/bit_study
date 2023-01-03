import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

#1. data

#x=np.array([range(1,11)]) #(1,10)
#y=np.array([range(10)]) #(1,10)

x_train=np.array([range(1,8)]) #(7,1)
x_test=np.array([[8,9,10]]) #(3,1)
y_train=np.array([range(1,8)]) #(7,1)
y_test=np.array([range(7,10)]) #(3,1)


print(x_train.shape)
print(x_test.shape) 
print(y_train.shape) 
print(y_test.shape) 

x_train=x_train.T
x_test=x_test.T
y_train=y_train.T
y_test=y_test.T

print(x_train.shape) #(1,7)
print(x_test.shape) #(1,3)
print(y_train.shape) #(1,7)
print(y_test.shape) #(1,3)



#2. model

model=Sequential()
model.add(Dense(3,input_dim=1))
model.add(Dense(10))
model.add(Dense(15))
model.add(Dense(20))
model.add(Dense(1))

#3. compile, training

model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=150,batch_size=2)
# train용 data로 fit

#4. validation, predict

loss=model.evaluate(x_test,y_test)
# 평가용 data로 evaluate
print('loss :',loss)
result=model.predict([15])
print('predict :',result)



