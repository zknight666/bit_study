# 이메일 제목 고정: 한태희 00,000원
#첨부파일 소스, 가중치
# 삼성전자 월요일 시가

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dropout, MaxPooling2D, Dense, Flatten, LSTM
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pylab as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import time


#1. data 전처리

# 1.1) data 불러오기

sam01_csv=pd.read_csv('C:/study/_data/sam01/삼성전자 주가.csv',index_col=[0],encoding='cp949')
amo01_csv=pd.read_csv('C:/study/_data/sam01/아모레퍼시픽 주가.csv',index_col=[0],encoding='cp949')
# submission_csv=pd.read_csv('C:/study/_data/kospi200/submission.csv',index_col=[0])


#1.2) 총 data 개수, 컬럼 확인 & 클래스 확인 & Dtype 확인 & 결측치 확인
print(sam01_csv.shape) # (1980, 16)
print(amo01_csv.shape) # (2220, 16)
print(sam01_csv.info()) # <class 'pandas.core.frame.DataFrame'>, 16개 컬럼, 결측치 존재, dtypes: float64(3), object(13)
print(amo01_csv.info()) # <class 'pandas.core.frame.DataFrame'>, 16개 컬럼, 결측치 존재, dtypes: float64(3), object(13)
columns_to_change_sam01 = ['시가', '고가', '저가', '종가', '전일비', '거래량', '금액(백만)', '개인', '기관', '외인(수량)', '외국계', '프로그램']
columns_to_change_amo01 = ['시가', '고가', '저가', '종가', '전일비', '거래량', '금액(백만)', '개인', '기관', '외인(수량)', '외국계', '프로그램']
sam01_csv[columns_to_change_sam01] = sam01_csv[columns_to_change_sam01].astype(float)
sam01_csv.to_csv('C:/study/_data/sam01/삼성전자 주가.csv', index=False)
print(sam01_csv.info())




#1.3) object type -> float64로 변경
kospi_csv = kospi_csv.apply(pd.to_numeric())
samsung_csv = samsung_csv.apply(pd.to_numeric,error='coerce')
print(kospi_csv.info())
print(samsung_csv.info())

