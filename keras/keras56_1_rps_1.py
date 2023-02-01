import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
import os
from tensorflow.keras.utils import get_file
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense,Conv2D,Flatten, MaxPooling2D, BatchNormalization, Dropout
from tensorflow.keras.initializers import RandomNormal, constant

# data 증폭
train_datagen=ImageDataGenerator(
    rescale=1./255, # => min max 의미, 이미지 최소값-, 최대값 255
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7, # 전단
    fill_mode='nearest', # 비어있는 값에 가장 가까이 있는 수치를 채움.
)

test_datagn=ImageDataGenerator(
    rescale=1./255, # => 스칼라 min max 의미, 이미지 최소값-, 최대값 255
)

xy_train = train_datagen.flow_from_directory(
    directory='C:/_data/rps',
    target_size=(100,100),
    batch_size=32,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=True,
)


xy_test = train_datagen.flow_from_directory(
    'C:/_data/rps',
    target_size=(100,100),
    batch_size=32,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=True,
) # Found 120 images belonging to 2 classes.


# Found 2520 images belonging to 3 classes.
print(xy_train[0][0].shape) # x 값 (2520, 100, 100, 3)
print(xy_train[0][1]) # y 값


#2. model
model=Sequential()
model.add(Conv2D(16,(3,3),input_shape=(100,100,3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters=16, kernel_size=(3, 3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.5))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(rate=0.45))
model.add(BatchNormalization(
    momentum=0.9,
    epsilon= 0.005, #  epsilon: 분산이 0으로 계산되는 것을 방지하기 위해 분산에 추가되는 작은 실수(float) 값
    beta_initializer=RandomNormal(mean=0.0,stddev=0.05),
    gamma_initializer=constant(value=0.9)
))
model.add(Dense(3,activation='softmax')) # y=0,y=1이므로 sigmoid 사용해야함
model.summary()



# model.compile, training

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['acc']
)


early_stoppong = EarlyStopping(
    monitor='val_loss',
    patience=20,
    verbose=2,
    restore_best_weights=True
)

hist = model.fit(
    xy_train,
    steps_per_epoch=32, # 계산해줘야함. [총 사진 개수 / batch size = steps_per_epoch]
    epochs=10,
    validation_data=(xy_test[0][0],xy_test[0][1]),
    callbacks=[early_stoppong],
    verbose=2
    ) # x,y, batch size 이미 들어가있음, 


#4. 평가 , 예측

accuracy=hist.history['acc']
val_acc = hist.history['val_acc']
loss=hist.history['loss']
val_loss=hist.history['val_loss']

print('loss',loss[-1])
print('val_loss',val_loss[-1])
print('accuracy',accuracy[-1])
print('val_acc',val_acc[-1])






