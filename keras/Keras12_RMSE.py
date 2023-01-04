import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. data

x=np.array(range(1,21)) # ()
y=np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20]) # ()



print(x.shape)
print(y.shape)



x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size=0.7,
    shuffle=True,
    random_state=1
    )

#2. model


model=Sequential()
model.add(Dense(10,input_dim=1))
model.add(Dense(10))
model.add(Dense(20))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))


#3.compile, training

model.compile(loss='mse',optimizer='adam',metrics=['mae'])
model.fit(x,y,batch_size=2,epochs=70)


#4. 평가, 예측

loss=model.evaluate(x_test,y_test)
print("loss : ",loss)

y_predict=model.predict(x_test)

print("=====================")
print(y_test)
print('=====================')
print(y_predict)
print('=====================')


from sklearn.metrics import mean_squared_error 
def RMSE(y_test,y_predict):
        return np.sqrt(mean_squared_error(y_test,y_predict))

print("RMSE : ",RMSE(y_test,y_predict))

#RMSE :  2.6588770478179784
#RMSE :  2.6367295354401885







