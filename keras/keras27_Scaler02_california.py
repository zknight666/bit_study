from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dropout
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
#1. data

dataset=fetch_california_housing()

x=dataset.data
y=dataset.target

print(x.shape) # (506, 13)
print(y.shape)



x_train,x_test,y_train,y_test=train_test_split(
    x,y,
    train_size=0.8,
    random_state=1,
    shuffle=True
)


scaler_minmax=MinMaxScaler()
# scaler_standard=StandardScaler()
# x_train=scaler_standard.fit_transform(x_train)
# x_test=scaler_standard.transform(x_test)
x_train=scaler_minmax.fit_transform(x_train)
x_test=scaler_minmax.transform(x_test)







#2. model

model=Sequential()
model.add(Dense(32,input_dim=8,activation='selu'))
model.add(Dense(512,activation='selu'))
# model.add(Dropout(rate=0.2))
model.add(Dense(1))




#3. compile, training

model.compile(
    optimizer='nadam',
    loss='mae',
    metrics=['mse']
)

early_stopping=EarlyStopping(
    monitor='val_loss',
    patience=50,
    restore_best_weights=True,
    verbose=2
)


hist=model.fit(
    x_train,y_train,
    batch_size=32,
    epochs=1000,
    validation_split=0.2,
    verbose=2,
    callbacks=[early_stopping]
)





#4. 평가, 예측

loss=model.evaluate(x_test,y_test)

y_predict=model.predict(x_test)

def RMSE(y_test,y_predict):
    return(np.sqrt(mean_squared_error(y_test,y_predict)))




print("loss",loss,
      'RMSE : ',RMSE(y_test,y_predict),
      'r2 : ',r2_score(y_test,y_predict)
      )


print('hist : ',hist.history['loss'])

#5. 시각화

plt.figure(
    figsize=(9,6)
)
    

plt.plot(
    hist.history['loss'],
    c='red',
    marker='.',
    label='loss'
    )

plt.plot(
    hist.history['val_loss'],
    c='blue',
    marker='.',
    label='val_loss'
    )

plt.grid()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('california loss')
plt.legend(loc='upper right')
plt.show()












# R2 :  0.6782261757061483 이상 나올 것
"""
결과
loss [0.4580928683280945, 0.0026647287886589766] 
RMSE :  0.643215593730973 
r2 :  0.684585508811475
minmax scaler 사용했을때
loss [0.3463618755340576, 0.2919037342071533] 
RMSE :  0.5402811818786215 
r2 :  0.7774599426953992
"""







