from tensorflow.keras.datasets import fashion_mnist
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dropout

#1. data (data type 확인, data class 확인, data 차원 확인)

(x_train,y_train), (x_test,y_test) = fashion_mnist.load_data()

print (x_train.shape) # (60000, 28, 28) = data 개수 6만개, 사이즈 : 28*28, 흑백 사진
print (y_train.shape) # (60000,)
print (x_test.shape) # (10000, 28, 28)
print (y_test.shape) # (10000,)

# print(x_train[0]) # ?
# print(y_train[0]) # ?

print(np.unique(x_train[:2], return_counts=True)) # ? dtype=uint8
print(np.unique(y_train,return_counts=True)) # (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000], dtype=int64))
# 10개 클래스, dtype=uint8, 각 클래스 별 data 개수 동일








#2. model
model=Sequential()
model.add(Conv2D(filters=256,kernel_size=(2,2),strides=(1,1),padding='same',activation='relu',input_shape=(28,28,1)))


#3. fit, training


#4. 평가, 예측



