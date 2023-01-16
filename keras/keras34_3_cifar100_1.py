import datetime
import time
from tensorflow.keras.datasets import cifar100
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.initializers import RandomNormal, constant
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# 1. data
(x_train, y_train), (x_test, y_test) = cifar100.load_data()





# 1) data 확인
print(x_train.shape)  # (50000, 32, 32, 3)
print(y_train.shape)  # (50000, 1)
print(x_test.shape)  # (10000, 32, 32, 3)
print(y_test.shape)  # (10000, 1)
print(x_train[0])  # 32x32
print(y_train[0])  # [19]
print(np.unique(x_train[:2], return_counts=True))  # dtype=uint8
print(np.unique(y_train, return_counts=True))  # 클래스 100개 확인 완료





# 1) 정규화 (datasets 전처리)

# 1)) 실수형으로 변경
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train,
    train_size=0.8,
    random_state=93,
    shuffle=True,
    # y_train 훈련 (class)data 비율과 y_val 검증 (class)data 비율을 같게 함
    stratify=y_train
)




# 2. model
model = Sequential()
model.add(Conv2D(filters=256, kernel_size=(2, 2), input_shape=(32, 32, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(filters=256, kernel_size=(2, 2),activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.5))

model.add(Conv2D(filters=256, kernel_size=(2, 2),activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(filters=256, kernel_size=(2, 2),activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.5))

model.add(Conv2D(filters=256, kernel_size=(2, 2),activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(filters=256, kernel_size=(2, 2),activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.5))
model.add(Flatten())
# inputshape = (batch_size, input_dim) -> 행무시 -> (40000,)로 표시
model.add(Dense(1024, activation='relu'))
#inputshape = (batch_size, input_dim)
model.add(Dropout(rate=0.5))
model.add(BatchNormalization(
    momentum=0.95,
    epsilon= 0.005, #  epsilon: 분산이 0으로 계산되는 것을 방지하기 위해 분산에 추가되는 작은 실수(float) 값
    beta_initializer=RandomNormal(mean=0.0,stddev=0.05),
    gamma_initializer=constant(value=0.9)
))
model.add(Dense(100, activation='softmax'))

model.summary()









# 3. compile, training
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='RMSprop',
    metrics=['acc']
)


start = time.time()


early_stoppong = EarlyStopping(
    monitor='val_loss',
    patience=20,
    verbose=2,
    restore_best_weights=True
)


now_date = datetime.datetime.now()
now_date = now_date.strftime("%m%d_%H%M")

model_checkpoint = ModelCheckpoint(
    filepath='c:/study/_save/MCP/' + 'K34_cifar100_' +
    now_date + '_{epoch}-{val_loss:.4f}.h5',
    monitor='val_loss',
    verbose=2,
    save_best_only=True
)


model.fit(
    x_train, y_train,
    batch_size=200,
    epochs=120,
    # steps_per_epoch=100, # batch_size와 비슷한 개념, 
    verbose=2,
    validation_data=(x_val, y_val),
    callbacks=[early_stoppong]

)

end = time.time()


# 4. 평가, 예측

results = model.evaluate(x_test, y_test)


print('loss : ', results[0])  # loss와 acc 값 2개 나옴
print('acc : ', results[1])
print('걸린시간 : ', end-start)


"""
loss :  2.516951560974121
acc :  0.37560001015663147
걸린시간 :  92.03419232368469

drop rate 0.5로 변경
loss :  2.35160493850708
acc :  0.40470001101493835
걸린시간 :  222.61408758163452

loss :  2.312697410583496
acc :  0.424699991941452
걸린시간 :  229.69466423988342


ImageDataGenerator 추가
0.3 나옴
0.01 나옴

stratify 추가, dropout 0.2로 변경
0.3 나옴


BatchNormalization/dropout 0.5/image datagenerator 제거/ 추가하기
loss :  1.6159613132476807
acc :  0.5623000264167786 ★★★★
걸린시간 :  1316.0125260353088

model.compile(optimizer=optimizers.RMSprop(lr=1e-4) 추가해보기




"""
