#분류    이미지=x,수치의 모음, 행렬 -> 결과 값=y,
# 픽셸~ 머신 이미지 인식 방법 한번에 인식 불가, 뭉치로 나눠서 인식 가운데부분 중첩해서 인식
#1번의 작업= 1layer -> 5x5 -> 4x4 
#복잡한 뉴런 네트웍(CNN)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

import numpy as np
from sklearn.model_selection import train_test_split


model=Sequential()

model.add(
    Conv2D
        (filters=10, # 2x2 짜리 커널(필터)을 10개로 늘리겠다 filters=parameter=커널            4x4x10
        kernel_size=(2,2), # 그 필터(커널)의 사이즈를 2x2짜리로 설정하겠다. ->  5x5의 4개 묶음이 4x4의 1개 layer됨 -> 한번 더 쓰면  3x3 => 2x2... (N,4,4,10)
        input_shape=(5,5,1) # 5x5짜리가 1개(5,5,1) /  color면 5x5짜리가 3개 = (5,5,3)
        )
    )
# model.add(Conv2D(filters=5,kernel_size=(2,2))) 
model.add(Conv2D(filters=5,kernel_size=(2,2))) # 4x4x10 -> 3x3x5 -> (n,3,3,5)dense 적용시킬려면 행렬로 바꿔야함. -> 펴줘야함 -> 3x3x5=45 -> (?,45) 컬럼 45개짜리 data
# 생략 가능 model.add(conv2d(5,(2,2))
# model.add(Conv2D(filters=5,kernel_size=(2,2))) 
#input= batch_size,row,columns,channels -> 가로,세로,색깔 -> 가로,세로,필터 / batch_size=훈련의 개수, = 행 / 
model.add(Flatten()) # 펴줌 (n,45)
model.add(Dense(10)) # (n,10) / # 
# model.add(Dense(units=10))
#인풋 (batch_size,input_dim) 열, 컬럼, 특성의 개수
#model.add(Dense(4,activation='relu')) # (n,1)
model.add(Dense(1)) # (n,1)

#1eopchs 이후 역전파되면서 weight값 갱신
#60000장 data 인풋은 (60000,5,5,1) (data 개수(장수),가로,세로,RGB) 총 4차원 = 4d shape

model.summary()
"""
=================================================================
 conv2d (Conv2D)             (None, 4, 4, 10)          50

 conv2d_1 (Conv2D)           (None, 3, 3, 5)           205 # filters 5

 flatten (Flatten)           (None, 45)                0 # flatten 적용되서 45됨

 dense (Dense)               (None, 10)                460

 dense_1 (Dense)             (None, 1)                 11

=================================================================
Total params: 726
Trainable params: 726
Non-trainable params: 0

summary conv2d, conv2d_1의 param 개수가 왜 저렇게 나온건지 설명하기 -> 필요한 이유 -> 내가 가진 메모리로 돌릴 수 있는지 여부 확인
input,outputshape 확인할 것, dense 확인할 것
input shape 조건 : 2가지 방식으로 표현됨 
    1) batch_shape + (channels, rows, cols) if data_format='channels_first' (RGB 수, 행, 열)
    2) batch_shape + (rows, cols, channels) if data_format='channels_last'. (행, 열, RGB 수)
output shape 조건 : 
    1) (filters, new_rows, new_cols) if data_format='channels_first' (필터 수, 새 행, 새 열)
    2) (new_rows, new_cols, filters) if data_format='channels_last'. (새 행, 새 열, 필터 수)
    
채널=RGB = 3 고정일듯




"""






 