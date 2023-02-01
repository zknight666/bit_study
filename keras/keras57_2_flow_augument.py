from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.datasets import fashion_mnist



(x_train,y_train),(x_test,y_test)=fashion_mnist.load_data()
augument_size=40000
randidx=np.random.randint(x_train.shape[0],size=augument_size) # 60000개 중에 랜덤하게 뽑아낸다.
print(len(randidx)) # 6만개 중 랜덤하게 4만개 뽑음

# x_train.shape=60000,28,28 -> x_train.shape[0] = 60000

x_augument=x_train[randidx].copy() # 4만개 들어감, 데이터 양 많을 경우 .copy 사용하여 데이터 건드리지 않기
y_augument=y_train[randidx].copy() # y 4만개 생성

print(x_augument.shape) # (40000, 28, 28)
print(y_augument.shape) # (40000,)

x_augument=x_augument.reshape(40000,28,28,1)

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
x_augumented = train_datagen.flow(
    x_augument,
    y_augument,
    batch_size=augument_size,
    shuffle=True
    )


print(x_augumented[0][0].shape) # (40000, 28, 28, 1)
print(x_augumented[0][1].shape) # (40000,)



x_train=x_train.reshape(60000,28,28,1)


x_train=np.concatenate((x_train,x_augumented[0][0]))
y_train=np.concatenate((y_train,x_augumented[0][1]))

print(x_train.shape) # (100000, 28, 28, 1)
print(y_train.shape) # (100000,)



# 파일 가져와서 쓰는 것 flow from directory, 수치화 된 것 가져오는 것 = flow