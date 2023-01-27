import numpy as np
from sklearn.datasets import load_digits
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.layers import Dense, Input, Dropout,LSTM
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
import matplotlib.pyplot as plt




#1. data
datasets=load_digits()
x=datasets.data
y=datasets['target']

print(x.shape,y.shape) # (1797, 64) (1797,) -> input 64개

print(np.unique(y,return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180], dtype=int64)) -> output 10개

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

#1.7) data 분류 후 train, test data 양 확인
print(x_train.shape) # (1437, 64)
print(x_test.shape) # (360, 64)

# scaler_standard=StandardScaler()
# x_train=scaler_standard.fit_transform(x_train)
# x_test=scaler_standard.transform(x_test)
scaler_minmax=MinMaxScaler()
x_train=scaler_minmax.fit_transform(x_train)
x_test=scaler_minmax.transform(x_test)


# reshape
x_train=x_train.reshape(1437,64,1)
x_test=x_test.reshape(360,64,1)

print(x_train.shape) # (1437, 64, 1)
print(x_test.shape) # (360, 64, 1)



#2.2 model (함수형)
model=Sequential()
model.add(LSTM(units=15,input_shape=(64,1), activation='relu',return_sequences=True))
model.add(LSTM(15))
model.add(Dense(300,activation='relu')) #
model.add(Dropout(rate=0.3))
model.add(Dense(10,activation='softmax'))

model.summary()



#3. compile, training
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
    
)



early_stopping=EarlyStopping(
    monitor='val_loss',
    patience=30,
    verbose=2,
    restore_best_weights=True
)


import datetime
date=datetime.datetime.now()
date=date.strftime("%m%d_%H%M")

model_checkpoint=ModelCheckpoint(
    filepath='c:/study/_save/MCP/' + 'K31_digits_' + date + '_{epoch}-{val_loss:.4f}.h5',
    monitor='val_loss',
    verbose=2,
    save_best_only=True
)



hist=model.fit(
    x_train,y_train,
    epochs=500,
    batch_size=16,
    verbose=2,
    validation_split=0.2,
    callbacks=[early_stopping]
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



#5. 시각화
# plt.figure(
#     figsize=(12,12)
#     )
# plt.plot(
#     hist.history['loss'],
#     c='red',
#     marker='.',
#     laber='loss'
# )
# # plt.plot(
# #     hist.history['accuracy'],
# #     c='blue',
# #     marker='.',
# #     laber='accuracy'
# # )
# plt.xlabel('epochs')
# plt.title('digits')
# plt.ylabel('loss')
# plt.legend(loc='upper right')
# plt.grid()
# plt.show()



"""
안사용
loss :  0.12287168204784393
accuracy :  0.9583333134651184

minmax 사용
loss :  0.18911316990852356
accuracy :  0.9444444179534912

scaler 사용
loss :  0.26374053955078125
accuracy :  0.9416666626930237


dropout + min max 사용
loss :  0.1230866014957428
accuracy :  0.9611111283302307



"""
