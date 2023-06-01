import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

#Separate dates for future plotting
train_dates = pd.to_datetime(df['Date'])
print(train_dates.tail(15)) #Check last few dates. 
