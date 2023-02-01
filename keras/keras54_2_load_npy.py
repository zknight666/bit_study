import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense,Conv2D,Flatten

#1. data 전처리
# np.save('C:/study/_data/brain/brain_x_train.npy',arr=xy_train[0][0])
# np.save('C:/study/_data/brain/brain_y_train.npy',arr=xy_train[0][1])
# np.save('C:/study/_data/brain/brain_x_test.npy',arr=xy_test[0][0])
# np.save('C:/study/_data/brain/brain_y_test.npy',arr=xy_test[0][1])

x_train=np.load('C:/study/_data/brain/brain_x_train.npy')
y_train=np.load('C:/study/_data/brain/brain_y_train.npy')
x_test=np.load('C:/study/_data/brain/brain_x_test.npy')
y_test=np.load('C:/study/_data/brain/brain_y_test.npy')

print(x_train.shape) # (160, 200, 200, 1)
print(x_test.shape) # (120, 200, 200, 1)
print(y_train.shape) # (160,)
print(y_test.shape) # (120,)

# 모델 구성



#2. model

model=Sequential()
model.add(Conv2D(64,(2,2),input_shape=(200,200,1)))
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
    x_train,y_train,
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


