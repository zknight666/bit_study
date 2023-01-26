import numpy as np
from sklearn.datasets import load_wine
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
import matplotlib.pyplot as plt


#1. data
datasets=load_wine()
x=datasets.data
y=datasets['target']
print(x.shape,y.shape) # (178, 13) (178,) -> input 64개

print(np.unique(y,return_counts=True)) # (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180], dtype=int64)) -> output 10개

# plt.gray()
# plt.matshow(datasets.images[9])
# plt.show() # 1700(행) * (8 * 8)(열) * RGB 
# one hot encoding 10개

# 이미지 건들기

y=to_categorical(y)
print(x)
print(y)
print(y.shape) # (1797, 10) 변환 확인 완료






x_train,x_test,y_train,y_test=train_test_split(
    x,y,
    random_state=1,
    shuffle=True,
    train_size=0.8,
    stratify=y
)


# scaler_standard=StandardScaler()
# x_train=scaler_standard.fit_transform(x_train)
# x_test=scaler_standard.transform(x_test)
scaler_minmax=MinMaxScaler()
x_train=scaler_minmax.fit_transform(x_train)
x_test=scaler_minmax.transform(x_test)






#2. model
# model=Sequential()
# model.add(Dense(50,activation='relu',input_shape=(13,)))
# model.add(Dense(40,activation='relu'))
# model.add(Dense(30,activation='relu'))
# model.add(Dense(20,activation='relu'))
# model.add(Dense(10,activation='relu'))
# model.add(Dense(3,activation='softmax'))



#2.2 model (함수형)

input1=Input(shape=(13,))
dense1=Dense(32,activation='relu')(input1)
dense2=Dense(400,activation='relu')(dense1)
dropout1=Dropout(rate=0.2)(dense2)
dense3=Dense(10,activation='relu')(dense2)
dense4=Dense(10,activation='relu')(dense3)
output1=Dense(3,activation='softmax')(dense4)
model=Model(inputs=input1,outputs=output1)





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



import datetime
date=datetime.datetime.now()
date=date.strftime("%m%d_%H%M")

model_checkpoint=ModelCheckpoint(
    filepath='c:/study/_save/MCP/' + 'K31_wine_' + date + '_{epoch}-{val_loss:.4f}.h5',
    monitor='val_loss',
    verbose=2,
    save_best_only=True
)





hist=model.fit(
    x_train,y_train,
    epochs=500,
    batch_size=1,
    verbose=2,
    validation_split=0.2,
    callbacks=[early_stopping, model_checkpoint]
)





#4. 평가 , 예측
loss, accuracy=model.evaluate(x_test,y_test)
y_predict=model.predict(x_test)



print(y_test)
print(y_predict)



y_predict=np.argmax(y_predict,axis=1)
y_test=np.argmax(y_test,axis=1)

print('y_test : ',y_test)
print('y_predict : ',y_predict)
# print('hist : ',hist.history['loss'])

print('loss : ',loss)
print('accuracy : ',accuracy)



"""
안사용

min max 사용마자용
loss :  0.009075845591723919
accuracy :  1.0


scaler 사용
loss :  0.22858744859695435
accuracy :  0.9722222089767456

dropout + min max 사용
loss :  0.00034129497362300754
accuracy :  1.0






"""
