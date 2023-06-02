import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('fast')
 
from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten,LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

output_folder = 'C:/Users/Alonso Nadeau/Documents/GitHub/tesisbkn/src/api/csv/'
output_path = output_folder + 'bitcoinYahoo.csv'
df = pd.read_csv(output_folder + 'bitcoinYahoo.csv')
print(df.head()) #7 columns, including the Date. 

# Separate dates for future plotting
train_dates = pd.to_datetime(df['Date'])
print(train_dates.tail(15)) #Check last few dates. 

# Variables para Entrenar
cols = list(df)[1:6]

df_for_training = df[cols].astype(float)


# Normalizador del Dataset
scaler= StandardScaler()
scaler =scaler.fit(df_for_training)
df_for_training_scaled = scaler.transform(df_for_training)


# Series de Tiempo
trainX = []
trainY = []

# Número de Dias que queremos usar para predeecir  
n_future = 1 
# Número de Dias que queremos usar como base para la predicción
n_past = 14 # 14 días

for i in range(n_past, len(df_for_training_scaled) - n_future +1):
    trainX.append(df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
    trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future, 0])

trainX, trainY = np.array(trainX), np.array(trainY)

print('trainX shape == {}.'.format(trainX.shape))
print('trainY shape == {}.'.format(trainY.shape))

# Definimos el modelo de autoencoder
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
model.add(LSTM(32, activation='relu', return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(trainY.shape[1]))

model.compile(optimizer='adam', loss='mse')
model.summary()

# Entrenamos el modelo

#
history = model.fit(trainX, trainY, epochs=10, batch_size=16, validation_split=0.1, verbose=1)

print("funcionó la volaita")
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
#plt.show()

# Predicción

n_future=90
forecast_period_dates = pd.date_range(list(train_dates)[-1], periods=n_future, freq='1d').tolist()

forecast = model.predict(trainX[-n_future:])
forecast_copies = np.repeat(forecast, df_for_training.shape[1], axis=-1)
y_pred_future = scaler.inverse_transform(forecast_copies)[:,0]

# Convertir las fechas a formato datetime
forecast_dates = []
for time_i in forecast_period_dates:
    forecast_dates.append(time_i.date())

df_forecast = pd.DataFrame({'Date':np.array(forecast_dates), 'Open':y_pred_future})
df_forecast['Date']=pd.to_datetime(df_forecast['Date'])

original = df[['Date', 'Open']]
original['Date']=pd.to_datetime(original['Date'])
original = original.loc[original['Date'] >= '2022-06-02']

sns.lineplot(data=original, x='Date', y='Open')
sns.lineplot(data=df_forecast, x='Date', y='Open')
plt.show()