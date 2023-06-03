import matplotlib.pyplot as plt
import seaborn as sns
import pandas_datareader.data as pdr
import pandas as pd
from datetime import datetime

output_folder = "C:/Users/Alonso Nadeau/Documents/GitHub/tesisbkn/src/api/csv/"  # Carpeta de salida)
output_path = output_folder + "bitcoinYahoo.csv"  # Ruta del archivo de salida
df = pd.read_csv(
    output_folder + "bitcoinYahoo.csv"
)  # Leer el archivo de datos como un DataFrame
print(df.head())

sns.set_style("darkgrid")
plt.figure(figsize=(8, 5), dpi=120)
sns.lineplot(x="Date", y="Open", data=df, label="BTC-USD")
plt.show()
