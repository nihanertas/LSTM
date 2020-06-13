import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense 

data = pd.read_csv('xu100.csv')

dataset_total = data.iloc[:, 4:5].values #Close values 

print(type(dataset_total))
print(len(dataset_total))
print(dataset_total.shape)

sc = MinMaxScaler(feature_range=(0,1))
data_scaled = sc.fit_transform(dataset_total)

[print(i) for i in  data_scaled[0:5]]



len_dataset = len(dataset_total)
len_train = int(len_dataset*0.8)

training_set_scaled = data_scaled[:len_train]

real_stock_price = dataset_total[len_train:]
len_test = len(real_stock_price)

print(type(training_set_scaled))
print(len(training_set_scaled))
print(training_set_scaled.shape)


print(data.head())
[print(i) for i in  training_set_scaled[0:5]]



X_train = []
y_train = []

for i in range(60, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
    # print(X_train[-1][-1], y_train[-1])

X_train, y_train = np.array(X_train), np.array(y_train)
# print(len(X_train[0]))

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

model = Sequential()

model.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50))
model.add(Dropout(0.2))

model.add(Dense(units=1))

model.compile(optimizer='adam',loss='mean_squared_error')

model.fit(X_train,y_train,epochs=100,batch_size=32)

inputs = dataset_total[len(dataset_total) - len(real_stock_price) - 60:]
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

X_test = []
for i in range(60, 60+len_test):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

plt.figure()
plt.plot(real_stock_price, color = 'black', label = 'BIST100 Index Price')
plt.plot(predicted_stock_price, color = 'green', label = 'Predicted BIST100 Index Price')
plt.title('BIST100 Index Price Prediction')
plt.xlabel('Time')
plt.ylabel('BIST100 Index Price')
plt.legend()
plt.show()