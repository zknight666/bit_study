from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense,Conv2D,Flatten,Dropout, MaxPooling2D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from tensorflow.keras.initializers import RandomNormal, constant




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
    rescale=1./255, # => 스칼라 min max 의미, 이미지 최소값-, 최대값 255
)
# train만 문제량 늘리는게 맞음. 시험 보러가서 문제를 10배로 늘리는 것은 의미 x, 학습량만 늘리는게 맞음.




# 디렉토리 = 폴더
 
xy_train = train_datagen.flow_from_directory(
    'C:/study/_data/brain/train/',
    target_size=(100,100),
    batch_size=5,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True,
) # Found 160 images belonging to 2 classes.
# Found 160 images belonging to 2 classes.
# x = 160*150*150*1 (160장, 150*150 사이즈, 흑백)
# y = (160,) np.unique -> 0,1로 나옴 -> 0=80, 1=80 장씩 나옴.

# 사진 사이즈가 다른경우? -> target size를 사용하여 이미지 크기 증폭 or 축소

xy_test = train_datagen.flow_from_directory(
    'C:/study/_data/brain/test/',
    target_size=(100,100),
    batch_size=5,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True,
) # Found 120 images belonging to 2 classes.


print(xy_train) # <keras.preprocessing.image.DirectoryIterator object at 0x000002046C7AFCA0>
print(xy_train[0]) # x,y, 두개 들어가 있음
print(xy_train[0][0]) # 0의 0번째 
print(xy_train[0][0].shape) # x 값 (10, 200, 200, 1)
print(xy_train[0][1]) # y 값


print(xy_train[15][0]) # 0의 1번째 = y [0. 1. 1. 1. 1. 1. 0. 0. 0. 1.]
print(xy_train[15][1].shape) # (10,)



print(type(xy_train)) # <class 'keras.preprocessing.image.DirectoryIterator'>
print(type(xy_train[0])) # <class 'tuple'> # ★   tuple형태 list의 차이
print(type(xy_train[0][0])) # <class 'numpy.ndarray'>
print(type(xy_train[0][1])) # <class 'numpy.ndarray'>
# ★   tuple형태 list의 차이




#2. model
model=Sequential()
model.add(Conv2D(64,(2,2),input_shape=(100,100,1), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters=64, kernel_size=(2, 2),activation='relu'))
model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(rate=0.5))
# model.add(Conv2D(filters=256, kernel_size=(2, 2),activation='relu', padding='same'))
# model.add(BatchNormalization())
# model.add(Conv2D(filters=256, kernel_size=(2, 2),activation='relu', padding='same'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(rate=0.5))
model.add(Flatten())
model.add(Dense(200, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(BatchNormalization(
    momentum=0.95,
    epsilon= 0.005, #  epsilon: 분산이 0으로 계산되는 것을 방지하기 위해 분산에 추가되는 작은 실수(float) 값
    beta_initializer=RandomNormal(mean=0.0,stddev=0.05),
    gamma_initializer=constant(value=0.9)
))
# model.add(Dense(1,activation='sigmoid')) # y=0,y=1이므로 sigmoid 사용해야함
model.add(Dense(2,activation='softmax'))
model.summary()
# Total params: 81,948,065
#Trainable params: 81,946,785 ???

# model.compile, training

model.compile(
    optimizer='adam',
    # loss='binary_crossentropy',
    loss='sparse_categorical_crossentropy',
    metrics=['acc']
)


early_stoppong = EarlyStopping(
    monitor='val_loss',
    patience=50,
    verbose=2,
    restore_best_weights=True
)

hist = model.fit_generator(
    xy_train,
    steps_per_epoch=12, # 계산해줘야함. [총 사진 개수 / batch size = steps_per_epoch]
    epochs=500,
    validation_data=(xy_test[0][0],xy_test[0][1]),
    # validation_steps=4,
    verbose=2,
    callbacks=[early_stoppong]
    ) # x,y, batch size 이미 들어가있음, 



accuracy=hist.history['acc']
val_acc = hist.history['val_acc']
loss=hist.history['loss']
val_loss=hist.history['val_loss']

print('loss',loss[-1])
print('val_loss',val_loss[-1])
print('accuracy',accuracy[-1])
print('val_acc',val_acc[-1])

"""
loss 0.6314132809638977
val_loss 0.3867977261543274
accuracy 0.6812499761581421
val_acc 0.824999988079071
"""




# matplot 그림그리기


#5. 시각화
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import array_to_img


epochs = range(1, len(loss) + 1) #1부터 len(loss) + 1 범위까지를 epochs라 한다. / loss는 epochs 한번 당 1개씩 나옴.
fig = plt.figure(figsize = (10, 5))

# 훈련 및 검증 손실 그리기
ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(epochs, loss, color = 'blue', label = 'train_loss')
ax1.plot(epochs, val_loss, color = 'orange', label = 'val_loss')
ax1.set_title('train_loss and val loss')
ax1.set_xlabel('epochs')
ax1.set_ylabel('loss')
ax1.legend()

# 훈련 및 검증 정확도 그리기
ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(epochs, accuracy, color = 'blue', label = 'train_acc')
ax2.plot(epochs, val_acc, color = 'orange', label = 'val_acc')
ax2.set_title('train and val acc')
ax2.set_xlabel('epochs')
ax2.set_ylabel('loss')
ax2.legend()

plt.show()

"""
# plt.figure(figsize=(5,5))
# for i in range(9):
#   data = next(xy_train)
#   arr = data[0][0]
#   plt.subplot(3,3,i+1)
#   plt.xticks([]) # 눈금 지우기
#   plt.yticks([]) # 눈금 지우기
#   img = array_to_img(arr).resize((128, 128))
#   plt.imshow(img)
# plt.show()
# # ImageDataGenerator 적용 확인 완료


from re import X
label_index = xy_train.class_indices

{'cats':0, 'dogs':1}
x,y = xy_train.next()
print(len(x))

for i in range(0,3):
  image = x[i]
  label = y[i].astype('int')
  plt.xticks([]) # 눈금 지우기
  plt.yticks([]) # 눈금 지우기
  print('사진: {}'.format([k for k, v in label_index.items() if v == label]))
  plt.imshow(image)
  plt.show()


label_index = ['ad', 'normal'] # 고양이 0번, 강아지 1번
x,y = xy_test.next()
for i in range(0,3):
  image = x[i]
  label = y[i].astype('int')

  y_prob = model.predict(image.reshape(1,100,100,1))
  y_prob_class = (model.predict(image.reshape(1,100,100,1)) > 0.5).astype('int')[0][0]
  print('정답: {}'.format(label_index[label]), 'label :', label)
  print('예측: {}'.format(label_index[y_prob_class]), 'predicted value:', y_prob)
  plt.imshow(image)
  plt.show()
  
"""