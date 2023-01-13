from tensorflow.keras.datasets import cifar10
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dense,Flatten, Dropout, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.model_selection import train_test_split
# from sklearn.utils import shuffle
from keras.utils import np_utils



#1. data

(x_train,y_train), (x_test,y_test) = cifar10.load_data()

#1) data 확인 
print(x_train.shape) # (50000, 32, 32, 3) # input shape가 3232 3 -> reshape 필요 없음
print(y_train.shape) # (50000, 1)
print(x_test.shape) # ((10000, 32, 32, 3)
print(y_test.shape) # (10000, 1)
print(x_train[0]) # (32,32)
print(y_train[0]) # (6)



#1) 정규화 (datasets 전처리)

#1)) data class 확인
print(np.unique(x_train,return_counts=True)) #dtype=uint8, 8비트의 부호 없는 정수형 배열 형태
print(type(x_train)) # <class 'numpy.ndarray'> 확인 안되네
print(type(x_test))
print(np.unique(y_train,return_counts=True)) # (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000], dtype=int64))


#2)) 실수형으로 변경
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0


#3)) 원-핫 인코딩
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]



#100개의 클래스 100,
#10개의 클래스 10,
# 10개 컬럼


#2. model
model=Sequential()
model.add(Conv2D(filters=64,kernel_size=(2,2),input_shape=(32,32,3),activation='relu', padding='same'))
model.add(Dropout(rate=0.2))
model.add(Conv2D(filters=64,kernel_size=(2,2),activation='relu', padding='same')) 
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(rate=0.2))
model.add(Conv2D(filters=64,kernel_size=(2,2),activation='relu', padding='same')) 
model.add(Dropout(rate=0.2))
model.add(Flatten()) 
model.add(Dense(512,activation='relu')) #inputshape = (batch_size, input_dim) -> 행무시 -> (40000,)로 표시
                                       #inputshape = (batch_size, input_dim)
model.add(Dropout(rate=0.2))
model.add(Dense(num_classes, activation='softmax'))







#3. compile, training
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['acc']
)

import time

start=time.time()




early_stoppong=EarlyStopping(
    monitor='val_loss',
    patience=10,
    verbose=2,
    restore_best_weights=True
)

import datetime

now_date=datetime.datetime.now()
now_date=now_date.strftime("%m%d_%H%M")

model_checkpoint=ModelCheckpoint(
    filepath='c:/study/_save/MCP/' + 'K34_cifar10_' + now_date + '_{epoch}-{val_loss:.4f}.h5',
    monitor='val_loss',
    verbose=2,
    save_best_only=True
)


model.fit(
    x_train,y_train,
    epochs=40,
    batch_size=250,
    verbose=2,
    validation_data=(x_test,y_test),
    # validation_split=0.6,
    callbacks=[early_stoppong],
    shuffle=True
)

end=time.time()





#4. 평가, 예측

results=model.evaluate(x_test,y_test)


print('loss : ',results[0]) # loss와 acc 값 2개 나옴
print('acc : ',results[1]) 
print('걸린시간 : ',end-start)



"""
loss :  2.302597761154175
acc :  0.10000000149011612
걸린시간 :  84.16299319267273

@dropout 사용했을 때
loss :  1.2130070924758911
acc :  0.5764999985694885
걸린시간 :  214.36703515052795

dropout + adadelta = x

@dropout + data 전처리 (x data, y data )
loss :  0.9850846529006958
acc :  0.6535999774932861
걸린시간 :  131.6443636417389

@dropout + data 전처리 (x data, y data ) + maxpooling 2d 추가 + padding 추가
loss :  0.8930463790893555
acc :  0.6942999958992004
걸린시간 :  124.70405101776123

@+model fit true
loss :  0.8629038333892822
acc :  0.7013999819755554
걸린시간 :  128.23426699638367

@dropout 0.25 변경/ dense 300으로 변경 = 변화 x


@validation split 0.5로 변경
loss :  0.8067587018013
acc :  0.7249000072479248
걸린시간 :  160.42562413215637

@+ dense 512, nadam 변경
loss :  0.7800161838531494
acc :  0.7304999828338623
걸린시간 :  200.289692401886


padding이란
maxpooling이란
shuffle 방법
model fit shuffle true 의미
데이터 증대(Data Augmentation) = 사진 resize (회전, 축소)
"""



# Conv2D(컨볼루션 필터의 수, 컨볼루션 커널(행,열) 사이즈, padding(valid(input image > output image), same(입력 = 출력), 
#        샘플 수를 제외한 입력 형태(행, 열 채널 수)), 입력 이미지 사이즈, 활성화 함수)
# MaxPooling은 풀링 사이즈에 맞춰 가장 큰 값을 추출함 (2,2)일 경우 입력 영상 크기에서 반으로 줄어듬.





