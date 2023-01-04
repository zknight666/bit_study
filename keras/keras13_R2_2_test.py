
#실습  R2를 음수가 아닌 0.5 이하로 줄이기
#2. 데이터 그대로
#3. 레이어 인풋 아웃풋 포함 7개
#4. batch size1, hiddne layer 10개~100개 사이, train70%, epochs 100번 이상, loss= mse, mae, 

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
#1. data

x=np.array(range(1,21)) # ()
y=np.array(range(1,21)) # ()


print(x.shape)
print(y.shape)



x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size=0.7,
    shuffle=False,
    random_state=2
    )



#2. model


model=Sequential()
model.add(Dense(100,input_dim=1))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))


#3.compile, training

model.compile(loss='mae',optimizer='adam',metrics=['mse'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=False)


model.fit(x_train,y_train,
          batch_size=1,
          epochs=100, 
          validation_split=0.2, 
          callbacks=[early_stopping]
          )

#4. 평가, 예측

loss=model.evaluate(x_test,y_test)
print("loss : ",loss)

y_predict=model.predict(x_test)



print("=====================")
print(y_test)
print('=====================')
print(y_predict)
print('=====================')



from sklearn.metrics import r2_score
R2 = r2_score(y_test,y_predict)


print("R2 : ",R2)



# from sklearn.metrics import mean_squared_error
# def RMSE(y_test,y_predict):
#         return np.sqrt(mean_squared_error(y_test,y_predict))

# print("RMSE : ",RMSE(y_test,y_predict))



# import matplotlib.pyplot as plt
# plt.scatter(x_train,y_train)
# plt.plot(x_test,y_predict, color='red')
# plt.show()

