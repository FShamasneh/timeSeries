# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 09:21:32 2019

@author: Nahir
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 

df = pd.read_csv('dataset/Miles_Traveled.csv',index_col = 'DATE', parse_dates = True)
df.index.freq = 'MS'
df.columns = ['Value']

df.plot(figsize = (12,5));

from statsmodels.tsa.seasonal import seasonal_decompose
components = seasonal_decompose(df['Value'])

components.observed.plot(figsize= (12,4))
components.seasonal.plot(figsize= (12,4))
components.trend.plot(figsize = (12,2))

train = df[:-12]
test = df[-12:]
len(test)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(train)
scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)

from keras.preprocessing.sequence import TimeseriesGenerator
n_input = 12 
n_features = 1 
generator = TimeseriesGenerator(scaled_train,scaled_train,length=n_input,batch_size=1)

from keras.models import Sequential
from keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(150, activation='relu', input_shape=(n_input,n_features)))
model.add(Dense(1))

model.compile(optimizer='adam',loss = 'mse')
model.summary()

model.fit_generator(generator,epochs = 10)

losses = model.history.history['loss']
plt.plot(range(len(losses)),losses)

first_ev_batch = scaled_train[-12:]
currentBatch = first_ev_batch.reshape((1,n_input,n_features))
listOfPreditions = []
for i in range(len(test)):
    pred = model.predict(currentBatch)[0]
    listOfPreditions.append(pred)
    currentBatch = np.append(currentBatch[:,1:,:],[[pred]],axis=1)
    
unscaledPredicitons = scaler.inverse_transform(listOfPreditions)
test['Predictions'] = unscaledPredicitons
test.plot(figsize=(12,6));
model.save('Model.h5')