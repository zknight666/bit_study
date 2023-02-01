import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
import os
from tensorflow.keras.utils import get_file
# ReduceLROnPlateau = 특정 지표가 개선을 멈췄을 때 옵티마이저의 학습률을 줄임.


train_datagen=ImageDataGenerator(rescale=1./255,)
test_datagn=ImageDataGenerator(rescale=1./255,)

xy_train = train_datagen.flow_from_directory(
    directory='C:/_data/cat_dog/train',
    target_size=(100,100),
    batch_size=100000,
    class_mode='binary',
    color_mode='rgb',
    shuffle=True,
)
# Found 25000 images belonging to 2 classes.


xy_test = train_datagen.flow_from_directory(
    directory='C:/_data/cat_dog/test1',
    target_size=(100,100),
    batch_size=100000,
    class_mode='binary',
    color_mode='rgb',
    shuffle=True,
) 
# Found 12500 images belonging to 1 classes. -> 지정 디렉토리에 바로 파일 있으면 적용 안됨. 폴더 안에 파일 있어야함

np.save('C:/study/_data/cat_dog/cat_dog_x_train.npy',arr=xy_train[0][0])
np.save('C:/study/_data/cat_dog/cat_dog_y_train.npy',arr=xy_train[0][1])
np.save('C:/study/_data/cat_dog/cat_dog_x_test.npy',arr=xy_test[0][0])
np.save('C:/study/_data/cat_dog/cat_dog_y_test.npy',arr=xy_test[0][1])


