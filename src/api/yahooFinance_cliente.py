import yfinance as yf
import pandas as pd

# Configurar el símbolo del Bitcoin y el rango de fechas
symbol = 'BTC-USD'
start_date = '2010-01-01'
end_date = '2023-06-01'

# Obtener los datos históricos del Bitcoin desde Yahoo Finance
data = yf.download(symbol, start=start_date, end=end_date)

# Seleccionar solo la columna 'Close' (precio de cierre)
data = data['Close']

# Revertir el orden de los datos para tener la fecha más antigua primero
data = data.iloc[::-1]

# Guardar los datos en un archivo CSV
data.to_csv('bitcoin.csv', header=['Precio de Cierre'])

print('Datos guardados exitosamente en bitcoin.csv')