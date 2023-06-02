import yfinance as yf
import pandas as pd

# Configurar el símbolo del Bitcoin y el rango de fechas
symbol = 'BTC-USD'
start_date = '2022-06-02'
end_date = '2023-06-01'

# Obtener los datos históricos del Bitcoin desde Yahoo Finance
data = yf.download(symbol, start=start_date, end=end_date)

# Seleccionar las columnas deseadas
columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
data = data[columns]

# Formatear los valores en las columnas excepto 'Volume' a 6 decimales
decimal_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']
data[decimal_columns] = data[decimal_columns].applymap(lambda x: '{:.6f}'.format(x))

# Ruta de salida del archivo CSV
output_folder = 'C:/Users/Alonso Nadeau/Documents/GitHub/tesisbkn/src/api/csv/'
output_path = output_folder + 'bitcoinYahoo.csv'

# Guardar los datos en un archivo CSV
data.to_csv(output_path)

print('bitcoinYahoo.csv guardado exitosamente en ' + output_path)