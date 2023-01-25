from tensorflow.keras.datasets import fashion_mnist
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dense,Flatten, Dropout, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint



(x_train,y_train), (x_test,y_test) = fashion_mnist.load_data()


print(x_train.shape) # (60000, 28, 28) data개수 6만장, 사이즈 28*28, 흑백사진
print(y_train.shape) # (60000,) 

print(x_test.shape) # (10000, 28, 28)
print(y_test.shape) # (10000,)

# import matplotlib.pyplot as plt
# plt.imshow(x_train[900],'gray')
# plt.show()

x_train=x_train.reshape(60000,28*28)
x_test=x_test.reshape(10000,28*28)

print(np.unique(x_train[:2], return_counts=True))  # dtype=int64
print(np.unique(y_train, return_counts=True))  # 클래스 10개 확인, 클래스별 데이터 개수 동일

x_train = x_train/255.
x_test = x_test/255.



#2. model
model=Sequential()
model.add(Dense(50,activation='relu', input_shape=(28*28,)))
model.add(Dense(512,activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(10,activation='softmax'))

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
    batch_size=128,
    verbose=2,
    validation_split=0.2,
    callbacks=[early_stoppong]
)

end=time.time()

#4. 평가, 예측


results=model.evaluate(x_test,y_test)



print('loss : ',results[0]) # loss와 acc 값 2개 나옴
print('acc : ',results[1]) 
print('걸린시간 : ',end-start)


"""

loss :  0.22543005645275116
acc :  0.9262999892234802
걸린시간 :  859.4679510593414

dnn 변환 후
loss :  0.5167781114578247
acc :  0.8116999864578247
걸린시간 :  41.0450222492218

dnn 변환 후 + scaler 추가
loss :  0.3454027771949768
acc :  0.8791000247001648
걸린시간 :  21.391445636749268



"""









