import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
#1. data


dataset=load_diabetes()
x=dataset.data
y=dataset.target

print(x.shape) # (442, 10)
print(y.shape)


x_train,x_test,y_train,y_test=train_test_split(
    x,y,
    train_size=0.8,
    random_state=1,
    shuffle=True
)


#2. model
model=Sequential()
model.add(Dense(100,input_dim=10,activation='relu'))
model.add(Dense(500,activation='relu'))
# model.add(Dense(500,activation='relu'))
# model.add(Dense(100))
# model.add(Dense(500))
# model.add(Dense(100))
# model.add(Dense(64,activation='relu'))
model.add(Dense(1))
          
          
          
          

    
    
#3. compile, training
model.compile(
    optimizer='adam',
    loss='mae',
    metrics=['accuracy']
)

model.fit(
    x_train,y_train,
    batch_size=1,
    epochs=500,
    validation_split=0.2
)


#4. 평가 ,예측


loss=model.evaluate(x_test,y_test)

y_predict=model.predict(x_test)


def RMSE(y_test,y_predict):
    return(np.sqrt(mean_squared_error(y_test,y_predict)))





print('loss',loss)
print('RMSE',RMSE(y_test,y_predict))
print('r2',r2_score(y_test,y_predict))

# r2 값 0.62 이상














