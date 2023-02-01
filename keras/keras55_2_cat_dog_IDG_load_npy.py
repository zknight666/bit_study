import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
import os
from tensorflow.keras.utils import get_file
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense,Conv2D,Flatten, MaxPooling2D, BatchNormalization, Dropout
from tensorflow.keras.initializers import RandomNormal, constant

x_train=np.load('C:/study/_data/cat_dog/cat_dog_x_train.npy')
y_train=np.load('C:/study/_data/cat_dog/cat_dog_y_train.npy')
x_test=np.load('C:/study/_data/cat_dog/cat_dog_x_test.npy')
y_test=np.load('C:/study/_data/cat_dog/cat_dog_y_test.npy')

print(x_train.shape) # (25000, 100, 100, 3)
print(y_train.shape) # (25000,)


#2. model

model=Sequential()
model.add(Conv2D(16,(3,3),input_shape=(100,100,3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters=16, kernel_size=(3, 3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(rate=0.5))
model.add(Flatten())
model.add(Dense(200, activation='relu'))
model.add(Dropout(rate=0.45))
model.add(BatchNormalization(
    momentum=0.9,
    epsilon= 0.005, #  epsilon: 분산이 0으로 계산되는 것을 방지하기 위해 분산에 추가되는 작은 실수(float) 값
    beta_initializer=RandomNormal(mean=0.0,stddev=0.05),
    gamma_initializer=constant(value=0.9)
))
model.add(Dense(1,activation='sigmoid')) # y=0,y=1이므로 sigmoid 사용해야함
# model.add(Dense(2,activation='softmax'))
model.summary()


# model.compile, training


model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['acc']
)



hist = model.fit(
    x_train,y_train,
    epochs=10,
    batch_size=250,
    validation_split=0.2,
    # validation_data=(xy_test[0][0],xy_test[0][1]),
    ) # x,y, batch size 이미 들어가있음, 

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

accuracy=hist.history['acc']
val_acc = hist.history['val_acc']
loss=hist.history['loss']
val_loss=hist.history['val_loss']

print('loss',loss[-1])
print('val_loss',val_loss[-1])
print('accuracy',accuracy[-1])
print('val_acc',val_acc[-1])


# # 5. 제출
sampleSubmission_csv = pd.read_csv('C:/study/_data/cat_dog/sampleSubmission.csv', index_col=0)
print(sampleSubmission_csv)  # 최종 data 확인
y_submit = model.predict(x_test)
y_submit = np.round(y_submit)
sampleSubmission_csv['label'] = y_submit
sampleSubmission_csv.to_csv('C:/study/_data/cat_dog/sampleSubmission.csv' + 'submission_0131_02.csv')



#5. 시각화
# import matplotlib.pyplot as plt
# from tensorflow.keras.preprocessing.image import array_to_img


# epochs = range(1, len(loss) + 1) #1부터 len(loss) + 1 범위까지를 epochs라 한다. / loss는 epochs 한번 당 1개씩 나옴.
# fig = plt.figure(figsize = (10, 5))





# # 훈련 및 검증 손실 그리기
# ax1 = fig.add_subplot(1, 2, 1)
# ax1.plot(epochs, loss, color = 'blue', label = 'train_loss')
# ax1.plot(epochs, val_loss, color = 'orange', label = 'val_loss')
# ax1.set_title('train_loss and val loss')
# ax1.set_xlabel('epochs')
# ax1.set_ylabel('loss')
# ax1.legend()

# # 훈련 및 검증 정확도 그리기
# ax2 = fig.add_subplot(1, 2, 2)
# ax2.plot(epochs, accuracy, color = 'blue', label = 'train_acc')
# ax2.plot(epochs, val_acc, color = 'orange', label = 'val_acc')
# ax2.set_title('train and val acc')
# ax2.set_xlabel('epochs')
# ax2.set_ylabel('loss')
# ax2.legend()

# plt.show()


"""
loss 0.5095224380493164
val_loss 0.5095224380493164
accuracy 0.9838500022888184
val_acc 0.7972000241279602
"""