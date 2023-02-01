# 압축풀기
import os
import zipfile

local_zip = 'C:/_data/dogs-vs-cats.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r') # 'r' = 읽기
zip_ref.extractall('C:/_data')
zip_ref.close()

# 데이터 경로 설정
rootPath = 'C:/_data/dogs-vs-cats' # 기본 경로

# 훈련 및 검증 데이터 경로
train_dir = os.path.join(rootPath, 'train') 

# 운영체제에서 제공해주는 경로를 붙이는 함수 join -> /tmp/cats_and_dogs_filtered/train
val_dir = os.path.join(rootPath, 'validation') 
# -> /tmp/cats_and_dogs_filtered/validation

# 훈련 데이터 중 고양이 사진 경로
train_cats_dir = os.path.join(train_dir, 'cats') 
# -> /tmp/cats_and_dogs_filtered/train/cats
# 훈련 데이터 중 강아지 사진 경로
train_dogs_dir = os.path.join(train_dir, 'dogs') 
# -> /tmp/cats_and_dogs_filtered/train/dogs

# 검증 데이터 중 고양이 사진 경로
val_cats_dir = os.path.join(val_dir, 'cats') 
# -> /tmp/cats_and_dogs_filtered/validation/cats
# 검증 데이터 중 강아지 사진 경로
val_dogs_dir = os.path.join(val_dir, 'dogs') 
# -> /tmp/cats_and_dogs_filtered/validation/dogs

train_cat_fnames = os.listdir(train_cats_dir)
train_cat_fnames.sort()
train_dog_fnames = os.listdir(train_dogs_dir)
train_dog_fnames.sort()
val_cat_fnames = os.listdir(val_cats_dir)
val_cat_fnames.sort()
val_dog_fnames = os.listdir(val_dogs_dir)
val_dog_fnames.sort()

# print(train_dog_fnames[:10])

print('훈련 고양이 데이터 사진 수: ', len(os.listdir(train_cats_dir)))
print('훈련 강아지 데이터 사진 수: ', len(os.listdir(train_dogs_dir)))
print('검증 고양이 데이터 사진 수: ', len(os.listdir(val_cats_dir)))
print('검증 강아지 데이터 사진 수: ', len(os.listdir(val_dogs_dir)))


