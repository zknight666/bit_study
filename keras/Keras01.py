import tensorflow as tf
print(tf.__version__)
import numpy as np

#1. data
x=np.array([1,2,3])
y=np.array([1,2,3])

#2. model
from tensorflow.keras.models import Sequential
#tensorflow 안에 keras 안에 models 안에 있는 Sequential import
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(1, input_dim=1))
#dim = 변수 축
#dense(y,x) / dense(output dim,input dim)

#3. compile, 훈련
model.compile(loss='mae', optimizer='adam')
#loss, error, cost
# weight 가중치, 최적

model.fit(x, y, epochs=3000)

#4. 평가, 예측
result = model.predict([4])
print('결과:', result)




# VS code = IDE 통합개발환경 (패키지 인클루딩, 문서 편집, 컴파일, 디버그, 원격 서버 액세스, 바이너리 배포)

