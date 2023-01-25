# DNN 모델로도 이미지 data 분석 가능하다 -> 하지만 이미지 기준 CNN이 더 좋음
# 36_1 복붙

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dense,Flatten, MaxPooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint


#1. data 
(x_train,y_train), (x_test,y_test) = mnist.load_data()
# 이미 train, test가 분리가 되어 있는 data임 

print(x_train.shape) # (60000, 28, 28) 6만장, 28x28 사이즈의 흑백
print(y_train.shape) # (60000,) 
# input shape 넣기 위해 reshape 함 (60000,28,28,1)이나 60000,28,28이나 같아서
print(x_test.shape) # (10000, 28, 28)
print(y_test.shape) # (10000,)
print(x_train[0]) # 28x28 
print(y_train[0]) # 5



#input shape는 3차원, 이미지는 4차원, 60000x28x28x1로 바꿈
# reshape 하거나 또는 flatten을 쓰거나
# x_train=x_train.reshape(60000,28,28,1)
# x_test=x_test.reshape(10000,28,28,1)
# x_train=x_train.reshape(60000,28*28)
# x_test=x_test.reshape(10000,28*28)

x_train=x_train/255.
x_test=x_test/255.

print(x_train.shape) # (60000, 28, 28, 1)  => (60000, 784) 784개 컬럼으로 바꿈
print(x_test.shape) # (10000, 28, 28, 1) => (10000, 784)
#CNN에 넣을 수 있는 4차원에 넣을 수 있는 data로 변환 완료



print(np.unique(y_train,return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],dtype=int64))
## 10개의 columns, input shape=









#2. model
model=Sequential()
model.add(Dense(40,input_shape=(28,28),activation='relu')) # 또는 input_shape=(28*28,)로 써도 된다.
# model.add(Dropout(0.2))
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.3))
# model.add(Dense(128,activation='relu'))
# model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(10,activation='softmax'))


# model.add(Conv2D(filters=128,kernel_size=(2,2),padding='same', input_shape=(28,28,1),activation='relu',strides=1))
# # kernel_size로 계속 자를 시 data의 가장자리 세부 data 사라짐. -> padding을 적용하여 input_shape 28*28 유지하여 data 보존 -> model.summary()로 확인 가능
# # padding 기본 옵션 : padding='vaild'
# # 현재 input_shape 28임 -> 일반 증명사진 사이즈 400*300 생각해야함. -> kernel size 2,2로 자르기에는 너무 크므로 그 이상으로 자른다.
# # strides = kenel_size 보폭. 기본값=1, maxpool기준 기본값=2
# model.add(MaxPooling2D(pool_size=(2,2),strides=1))
# model.add(Conv2D(filters=128,kernel_size=(2,2),padding='same',activation='relu')) # (27,27,128)
# model.add(Conv2D(filters=128,kernel_size=(2,2),padding='same',activation='relu')) # (26,26,64)
# model.add(Flatten()) # (25,25,64) -> # 40000
# model.add(Dense(32,activation='relu')) #inputshape = (60000,40000) -> 행무시 -> (40000,)로 표시
#                                        #inputshape = (batch_size, input_dim)
# model.add(Dense(10,activation='softmax'))


model.summary()






#3. compile, training
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['acc']
)

import time

start=time.time()




early_stoppong=EarlyStopping(
    monitor='val_loss',
    patience=20,
    verbose=2,
    restore_best_weights=True
)

import datetime

now_date=datetime.datetime.now()
now_date=now_date.strftime("%m%d_%H%M")

model_checkpoint=ModelCheckpoint(
    filepath='c:/study/_save/MCP/' + 'K34_mnist_' + now_date + '_{epoch}-{val_loss:.4f}.h5',
    monitor='val_loss',
    verbose=2,
    save_best_only=True
)


model.fit(
    x_train,y_train,
    epochs=200,
    batch_size=250,
    verbose=2,
    validation_split=0.2,
    callbacks=[early_stoppong] # 가중치 저장할거라면 modelcheckpoint 추가할 것
)

end=time.time()

#4. 평가, 예측


results=model.evaluate(x_test,y_test)



print('loss : ',results[0]) # loss와 acc 값 2개 나옴
print('acc : ',results[1]) 
print('걸린시간 : ',end-start)


#early stopping. model check point , validation split 적용



"""
GPU 사용 / 40ms/step

loss :  0.10192742943763733
acc :  0.9717000126838684

걸린시간 :  55.202141761779785
CPU 사용 /  579ms/step <15배 차이남..>



padding 적용시 성능
loss :  0.07824142277240753
acc :  0.9786999821662903
걸린시간 :  222.2018654346466

max pool 적용시 성능
loss :  0.06490149348974228
acc :  0.9814000129699707
걸린시간 :  101.31557774543762

maxpool strides=1 적용시 성능

DNN 적용시 성능
loss :  0.10050895065069199
acc :  0.9696999788284302
걸린시간 :  31.773829460144043

loss :  0.24879169464111328
acc :  0.9352999925613403
걸린시간 :  389.42547059059143


"""





