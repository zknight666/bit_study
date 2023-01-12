import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt












#1. data
datasets=fetch_covtype()
x=datasets.data
y=datasets.target

print(x)
print(y)
print(x.shape) # (581012, 54)
print(y.shape) # (581012,)

print(np.unique(y,return_counts=True)) # (array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510], dtype=int64)) # columns 7개, 4번째 표본 매우 적음 확인
print(type(y)) # <class 'numpy.ndarray'>

y=pd.get_dummies(y)
y=y.values





x_train,x_test,y_train,y_test=train_test_split(
    x,y,
    train_size=0.8,
    stratify=y,
    random_state=1,
    shuffle=True
)

scaler_minmax=MinMaxScaler()
x_train=scaler_minmax.fit_transform(x_train)
x_test=scaler_minmax.transform(x_test) # fit 하면 여기에도 scaler 적용됨. train 만 적용시키면 됨.


#2. model (함수형)
input1=Input(shape=(54,))
dense1=Dense(400,activation='relu')(input1)
dropout1=Dropout(rate=0.2)(dense1)
dense2=Dense(50,activation='relu')(dropout1)
dense3=Dense(30,activation='relu')(dense2)
output1=Dense(7,activation='softmax')(dense3)
model=Model(inputs=input1,outputs=output1)

model.summary()


#3. compile, training
model.compile(
    optimizer='adam',
    metrics=['accuracy'],
    loss='categorical_crossentropy'
)

early_stopping=EarlyStopping(
    monitor='val_loss',
    patience=25,
    verbose=2,
    restore_best_weights=True
)

import datetime
date=datetime.datetime.now()
date=date.strftime("%m%d_%H%M")

model_checkpoint=ModelCheckpoint(
    filepath='c:/study/_save/MCP/' + 'K31_fetch_covtype_' + date + '_{epoch}-{val_loss:.4f}.h5',
    monitor='val_loss',
    verbose=2,
    save_best_only=True
)




hist=model.fit(
    x_train,y_train,
    batch_size=500,
    epochs=300,
    verbose=2,
    validation_split=0.2,
    callbacks=[early_stopping, model_checkpoint]
    
)


#4. 평가, 예측
loss, accuracy=model.evaluate(x_test,y_test)
y_predict=model.predict(x_test)

print(y_predict)
print(x_test)


print('loss',loss)
print('accuracy',accuracy)






