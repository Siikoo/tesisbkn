import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV


output_folder = "C:/Users/Alonso Nadeau/Documents/GitHub/tesisbkn/src/api/csv/"  # Carpeta de salida)
output_path = output_folder + "bitcoinYahoo.csv"  # Ruta del archivo de salida
df = pd.read_csv(
    output_folder + "bitcoinYahoo.csv"
)  # Leer el archivo de datos como un DataFrame

print(df.shape)  # Mostrar las primeras 5 filas del DataFrame
test_split = round(len(df) * 0.2)
df_for_training = df[:-test_split]
df_for_testing = df[-test_split:]
print(df_for_training.shape)
print(df_for_testing.shape)

# Eliminar la columna 'Date' del DataFrame
df_for_training = df_for_training.drop("Date", axis=1)
df_for_testing = df_for_testing.drop("Date", axis=1)

# Escalar los datos sin la columna 'Date'
scaler = MinMaxScaler(feature_range=(0, 1))
df_for_training_scaled = scaler.fit_transform(df_for_training)
df_for_testing_scaled = scaler.transform(df_for_testing)
print(df_for_training_scaled)


def createXY(dataset, n_past):
    print("Funcionando Create XY")
    dataX = []
    dataY = []
    for i in range(n_past, len(dataset)):
        dataX.append(dataset[i - n_past : i, 0 : dataset.shape[1]])
        dataY.append(dataset[i, 0])
    return np.array(dataX), np.array(dataY)


trainX, trainY = createXY(df_for_training_scaled, 30)
testX, testY = createXY(df_for_testing_scaled, 30)


def build_model(optimizer):
    print("Funcionando Build Model")
    grid_model = Sequential()
    grid_model.add(LSTM(50, return_sequences=True, input_shape=(30, 5)))
    grid_model.add(LSTM(50))
    grid_model.add(Dropout(0.2))
    grid_model.add(Dense(1))

    grid_model.compile(loss="mse", optimizer=optimizer)
    return grid_model


grid_model = KerasRegressor(
    build_fn=build_model, verbose=1, validation_data=(testX, testY)
)
parameters = {
    "batch_size": [16, 20],
    "epochs": [
        1,
        2,
    ],  # Debería reducir los ciclos si quiero hacer testeos mas rápídos 8, 10
    "optimizer": ["adam", "Adadelta"],
}

grid_search = GridSearchCV(estimator=grid_model, param_grid=parameters, cv=2)
grid_search = grid_search.fit(trainX, trainY)
print(grid_search.best_params_)
my_model = grid_search.best_estimator_.model
print(my_model)
prediction = my_model.predict(testX)
print("prediction\n", prediction)
print("\nPrediction Shape-", prediction.shape)
print(prediction.shape)
# Hasta acá llega y se corta
# scaler.inverse_transform(prediction)
prediction_copies_array = np.repeat(prediction, 5, axis=-1)
print(prediction_copies_array.shape)
print(prediction_copies_array)
pred = scaler.inverse_transform(
    np.reshape(prediction_copies_array, (len(prediction), 5))
)[:, 0]
original_copies_array = np.repeat(testY, 5, axis=-1)
print(original_copies_array.shape)
original = scaler.inverse_transform(np.reshape(original_copies_array, (len(testY), 5)))[
    :, 0
]

"""
prediction_copies_array = np.repeat(prediction, 5, axis=-1)
print(prediction_copies_array.shape)
print(prediction_copies_array)
pred = scaler.inverse_transform(
    np.reshape(prediction_copies_array, (len(prediction), 5))
)[:, 0]
original_copies_array = np.repeat(testY, 5, axis=-1)
print(original_copies_array.shape)
original = scaler.inverse_transform(np.reshape(original_copies_array, (len(testY), 5)))[
    :, 0
]
"""
print(pred)
print("Pred Values-- ", pred)
print("\nOriginal Values-- ", original)

plt.plot(original, color="red", label="Real  Stock Price")
plt.plot(pred, color="blue", label="Predicted  Stock Price")
plt.title(" Stock Price Prediction")
plt.xlabel("Time")
plt.ylabel(" Stock Price")
plt.legend()
plt.show()
