from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense,Conv2D,Flatten



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
    rescale=1./255, # => min max 의미, 이미지 최소값-, 최대값 255
)
# train만 문제량 늘리는게 맞음. 시험 보러가서 문제를 10배로 늘리는 것은 의미 x, 학습량만 늘리는게 맞음.




# 디렉토리 = 폴더
 
xy_train = train_datagen.flow_from_directory(
    'C:/study/_data/brain/train/',
    target_size=(100,100),
    batch_size=1000,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True,
) # Found 160 images belonging to 2 classes.
# Found 160 images belonging to 2 classes.
# x = 160*150*150*1 (160장, 150*150 사이즈, 흑백)
# y = (160,) np.unique -> 0,1로 나옴 -> 0=80, 1=80 장씩 나옴.
# 사진 개수 정확히 모를경우 batch_size 많이 잡아놓으면 됨. -> 160,100,100 1 shape 확인할 수 있음 -> 사진 총 개수 확인 가능

# 사진 사이즈가 다른경우? -> target size를 사용하여 이미지 크기 증폭 or 축소

xy_test = train_datagen.flow_from_directory(
    'C:/study/_data/brain/test/',
    target_size=(100,100),
    batch_size=1000,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True,
) # Found 120 images belonging to 2 classes.


# print(xy_train) # <keras.preprocessing.image.DirectoryIterator object at 0x000002046C7AFCA0>
# print(xy_train[0]) # x,y, 두개 들어가 있음
# print(xy_train[0][0]) # 0의 0번째 
# print(xy_train[0][0].shape) # x 값 (10, 200, 200, 1)
# print(xy_train[0][1]) # y 값


# print(xy_train[15][0]) # 0의 1번째 = y [0. 1. 1. 1. 1. 1. 0. 0. 0. 1.]
# print(xy_train[15][1].shape) # (10,)



# print(type(xy_train)) # <class 'keras.preprocessing.image.DirectoryIterator'>
# print(type(xy_train[0])) # <class 'tuple'>
# print(type(xy_train[0][0])) # <class 'numpy.ndarray'>
# print(type(xy_train[0][1])) # <class 'numpy.ndarray'>




#2. model
model=Sequential()
model.add(Conv2D(64,(2,2),input_shape=(100,100,1)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(Flatten())
model.add(Dense(16,activation='relu'))
model.add(Dense(1,activation='sigmoid')) # y=0,y=1이므로 sigmoid 사용해야함

# model.compile, training

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['acc']
)

# hist = model.fit_generator(
#     xy_train,
#     steps_per_epoch=16,
#     epochs=10,
#     validation_data=xy_test,
#     validation_steps=4,
#     ) # x,y, batch size 이미 들어가있음, 


# ★★★ xy_train[0][0],xy_train[0][1], ★★★
hist = model.fit(
    # xy_train,
    xy_train[0][0],xy_train[0][1],
    epochs=10,
    batch_size=16, # 160개 data 16개씩 자름 -> 10번 움직임
    # validation_data=(xy_test[0][0],xy_test[0][1]),
    validation_split=0.2,
    ) # x,y, batch size 이미 들어가있음, 




accuracy=hist.history['acc']
val_acc = hist.history['val_acc']
loss=hist.history['val_loss']
val_loss=hist.history['val_loss']

print('loss',loss[-1])
print('val_loss',val_loss[-1])
print('accuracy',accuracy[-1])
print('val_acc',val_acc[-1])


# matplot 그림그리기



