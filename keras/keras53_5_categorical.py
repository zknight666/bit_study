from keras.preprocessing.image import ImageDataGenerator
import numpy as np

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
    target_size=(200,200),
    batch_size=10,
    class_mode='categorical',
    color_mode='grayscale',
    shuffle=True,
) # Found 160 images belonging to 2 classes.
# Found 160 images belonging to 2 classes.
# x = 160*150*150*1 (160장, 150*150 사이즈, 흑백)
# y = (160,) np.unique -> 0,1로 나옴 -> 0=80, 1=80 장씩 나옴.

# 사진 사이즈가 다른경우? -> target size를 사용하여 이미지 크기 증폭 or 축소

xy_test = train_datagen.flow_from_directory(
    'C:/study/_data/brain/test/',
    target_size=(200,200),
    batch_size=10,
    class_mode='categorical',
    color_mode='grayscale',
    shuffle=True,
) # Found 120 images belonging to 2 classes.


# print(xy_train) # <keras.preprocessing.image.DirectoryIterator object at 0x000002046C7AFCA0>
# print(xy_train[0]) # x,y, 두개 들어가 있음
# print(xy_train[0][0]) # 0의 0번째 
print(xy_train[0][0].shape) # (10, 200, 200, 1) 얘는 똑같음
print(xy_train[0][1].shape) # binary로 했을 때 : (10, 200, 200, 1) class_mode='categorical'로 했을 때  -> (10, 2)
print(xy_train[0][1]) # binary로 했을 때 [1. 0. 0. 0. 0. 1. 1. 0. 0. 1.] /    class_mode='categorical'로 했을 때  -> onehot 됨 
#-> 양 많아지면 변환 시간 많아짐 -> 모델 돌릴때마다 변환하면 시간 낭비 -> numpy로 저장한 후 모델 돌림. 
"""
[[0. 1.]
 [0. 1.]
 [1. 0.]
 [0. 1.]
 [0. 1.]
 [0. 1.]
 [0. 1.]
 [0. 1.]
 [1. 0.]
 [0. 1.]]
"""

# print(xy_train[15][0]) # 0의 1번째 = y [0. 1. 1. 1. 1. 1. 0. 0. 0. 1.]
# print(xy_train[15][1].shape) # (10,)



# print(type(xy_train)) # <class 'keras.preprocessing.image.DirectoryIterator'>
# print(type(xy_train[0])) # <class 'tuple'>
# print(type(xy_train[0][0])) # <class 'numpy.ndarray'>
# print(type(xy_train[0][1])) # <class 'numpy.ndarray'>







