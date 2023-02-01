from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.datasets import fashion_mnist



(x_train,y_train),(x_test,y_test)=fashion_mnist.load_data()
augument_size=100




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
 # 사이즈 28*28임
x_data = train_datagen.flow(
    np.tile(x_train[0].reshape(28*28),augument_size).reshape(-1,28,28,1), # 타일1개=이미지 한개 -1=전체를 의미 전체 개수 모를 때 사용, x_train 0번째가 100개가 생김. #x
    np.zeros(augument_size), #y
    batch_size=augument_size,
    shuffle=True
    )

print(x_data[0])
print(x_data[0][0].shape) # (100, 28, 28, 1)
print(x_data[0][1].shape) # (100,)

import matplotlib.pylab as plt

plt.figure(figsize=(7,7))
for i in range(49):
    plt.subplot(7,7,i+1)
    plt.axis('off')
    plt.imshow(x_data[0][0][i],cmap='gray')
plt.show()   




# Found 160 images belonging to 2 classes.
# x = 160*150*150*1 (160장, 150*150 사이즈, 흑백)
# y = (160,) np.unique -> 0,1로 나옴 -> 0=80, 1=80 장씩 나옴.

# 사진 사이즈가 다른경우? -> target size를 사용하여 이미지 크기 증폭 or 축소

# xy_test = train_datagen.flow_from_directory(
#     'C:/study/_data/brain/test/',
#     target_size=(200,200),
#     batch_size=10,
#     class_mode='binary',
#     color_mode='grayscale',
#     shuffle=True,
# ) # Found 120 images belonging to 2 classes.









