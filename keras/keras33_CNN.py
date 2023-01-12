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
        (filters=10, # 2x2 짜리를 10개로 늘리겠다 filters=parameter 
        kernel_size=(2,2), # 2x2짜리 묶음으로 자르겠다. ->  5x5의 4개 묶음이 4x4의 1개 layer됨 -> 한번 더 쓰면  3x3 => 2x2...
        input_shape=(5,5,1) # 5x5짜리가 1개(5,5,1) /  color면 5x5짜리가 3개 = (5,5,3)
        )
    )

model.add(Conv2D(filters=5,kernel_size=(2,2))) # 3x3x5 -> dense 적용시킬려면 행렬로 바꿔야함. -> 펴줘야함 -> 3x3x5=45 -> (?,45) 컬럼 45개짜리 data
model.add(Flatten()) # 펴줌
model.add(Dense(10))
model.add(Dense(1))

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
"""






 