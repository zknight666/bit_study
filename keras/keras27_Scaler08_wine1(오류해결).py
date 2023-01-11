import numpy as np
from sklearn.datasets import load_wine
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# import matplotlib.pyplot
# from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
# import pandas as pd


#1. data
datasets=load_wine()
x=datasets.data
y=datasets.target


print(x.shape) # (178, 13) 
print(y.shape) # (178,)

print(np.unique(y)) # y= [0 1 2] 3개 있다는 것 확인 -> 다중분류
print(np.unique(y,return_counts=True)) # (array([0, 1, 2]), array([59, 71, 48], dtype=int64)) -> 0,1,2가 각각 59개, 71개, 48개 있다는 것 확인 & data간 불균형이 약간 있으므로 train split에서 stratify y해줄 것


y=to_categorical(y)
print(x)
print(y)
print(y.shape) # (178, 3) 변환 확인 완료

"""
1) 에러 확인
2) input,output 수 확인
3) 차원 확인, 타입 확인
"""

x_train,x_test,y_train,y_test=train_test_split(
    x,y,
    random_state=1,
    shuffle=True,
    train_size=0.8,
    stratify=y
)

   
   # scaler_minmax=MinMaxScaler()
scaler_standard=StandardScaler()
x_train=scaler_standard.fit_transform(x_train)
x_test=scaler_standard.transform(x_test)
# x_train=scaler_minmax.fit_transform(x_train)
# x_test=scaler_minmax.transform(x_test)
   
   
   
    
#2. model
model=Sequential()
model.add(Dense(50,activation='relu',input_shape=(13,)))
model.add(Dense(400,activation='relu'))
model.add(Dense(3,activation='softmax'))





#3. compile, training
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
    
)

early_stopping=EarlyStopping(
    monitor='val_loss',
    patience=50,
    verbose=2,
    restore_best_weights=True
)



model.fit(
    x_train,y_train,
    epochs=200,
    batch_size=1,
    verbose=2,
    validation_split=0.2,
    callbacks=[early_stopping]
)
# early_stopping을 EarlyStopping으로 사용
# 당시 에러 메세지 TypeError: set_model() missing 1 required positional argument: 'model'





#4. 평가 , 예측
loss, accuracy=model.evaluate(x_test,y_test)
y_predict=model.predict(x_test)


print('loss : ',loss)
print('accuracy : ',accuracy)

print(y_test)
print(y_predict)


y_predict=np.argmax(y_predict,axis=1)
y_test=np.argmax(y_test,axis=1)

print('y_test : ',y_test)
print('y_predict : ',y_predict)




#5. 시각화


"""
안사용

scaler 사용


"""