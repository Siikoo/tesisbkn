import pandas as pd
import matplotlib.pyplot as plt
import requests
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.cryptocurrencies import CryptoCurrencies

# Obtén una clave de API de Alpha Vantage
API_KEY = '0LMC8AND1QTHQ3H60'

# Crea una instancia de TimeSeries y obtén los datos
cc = CryptoCurrencies(key=API_KEY, output_format='pandas')
data, meta_data = cc.get_digital_currency_daily(symbol='BTC', market='USD')
print(data.head())
print(meta_data)

# Extrae la fecha y el precio de cierre en variables separadas
fechas = data.index
precios_cierre = data['4a. close (USD)']

# Crea un diccionario con las columnas
datos = {'fecha': fechas, 'preciodecierre': precios_cierre}

# Crea un DataFrame a partir del diccionario
df = pd.DataFrame(datos)

# Guarda los datos en un archivo CSV
df.to_csv('bitcoinAlpha.csv', index=False)
print('Datos guardados exitosamente en bitcoinAlpha.csv')

