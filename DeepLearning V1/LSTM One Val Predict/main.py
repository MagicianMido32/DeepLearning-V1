import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler


def train_test_split(dataset, split_by=0.7):
    split_to = int(split_by * len(dataset))
    dataset_train = dataset[:split_to]
    dataset_test = dataset[split_to:]
    return dataset_train, dataset_test


def get_mape(y_true, y_predicted):
    y_true, y_predicted = np.array(y_true), np.array(y_predicted)
    mape = np.mean(np.abs((y_true - y_predicted) / y_true)) * 100
    return mape


def get_rmse(y_true, y_predicted):
    rmse = np.sqrt(np.mean(np.power((y_true - y_predicted), 2)))
    return rmse


def prepare_data(dataset, max_marker=60, lag_marker=50, stride=1):
    data_set = dataset.iloc[:, 1:2].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_set_scaled = scaler.fit_transform(data_set)
    x_data_set = []
    y_data_set = []
    for counter in range(max_marker, len(data_set), stride):
        # get a window from min_index to max_index
        x_data_set.append(data_set_scaled[counter - lag_marker:counter, 0])
        y_data_set.append(data_set_scaled[counter, 0])
    x_data_set, y_data_set = np.array(x_data_set), np.array(y_data_set)
    x_data_set = np.reshape(x_data_set, (x_data_set.shape[0], x_data_set.shape[1], 1))
    return x_data_set, y_data_set, scaler


def create_model(x_data_set, y_data_set, lag=50, epochs=50, batch_size=32):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(lag, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['MAPE', 'accuracy'])
    model.fit(x_data_set, y_data_set, epochs=epochs, batch_size=batch_size)
    return model


def predict(model, data_set, scaler):
    predicted = model.predict(data_set)
    predicted = scaler.inverse_transform(predicted)
    return predicted


def plot(real_df, predicted_values, max_marker=60, lag_marker=50, title="Real(blue) vs predicted(red)",
         x_label="occurtime", y_label="power", figsize_x=100, figsize_y=60):
    plt.figure(figsize=(figsize_x, figsize_y))
    plt.xticks(np.arange(0, len(real_df) - max_marker, lag_marker))
    plt.plot(real_df[x_label], real_df[y_label], color='blue', label='Real')
    plt.plot(real_df[x_label], predicted_values, color='red', label='predicted')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(["Real values", "Predicted values"], loc="upper left")

    return plt.show()


def predict_two_datasets(train_path, test_path, max_marker=60, lag=50, epochs=50, batch_size=32,
                         title="Real(blue) vs predicted(red)", x_label="occurtime", y_label="power"):
    df_train = pd.read_csv(train_path)
    x_train_data_set, y_train_data_set, scaler = prepare_data(df_train, max_marker, lag)

    model = create_model(x_train_data_set, y_train_data_set, lag=lag, epochs=epochs, batch_size=batch_size)

    df_test = pd.read_csv(test_path)
    x_test_data_set, y_test_data_set, scaler = prepare_data(df_test, max_marker, lag)

    y_predicted = predict(model, x_test_data_set, scaler).flatten()

    print('MAPE: ', get_mape(y_test_data_set, y_predicted))
    print('RMSE: ', get_rmse(y_test_data_set, y_predicted))

    predicted_values = y_predicted.flatten()
    real_df = df_test[:len(predicted_values)]

    plot(real_df, y_predicted, max_marker=max_marker, lag_marker=lag, title=title,
         x_label=x_label, y_label=y_label)


# consider receiving a dataframe instead of a path to be able to manipulate the data outside the logic
def predict_one_dataset(path1, split_by=0.7, max_marker=60, lag=50, epochs=50, batch_size=32,
                        title="Real(blue) vs predicted(red)", x_label="occurtime", y_label="power"):
    df_total = pd.read_csv(path1)
    df_train, df_test = train_test_split(df_total, split_by)
    x_train_data_set, y_train_data_set, scaler = prepare_data(df_train, max_marker, lag)

    model = create_model(x_train_data_set, y_train_data_set, lag=lag, epochs=epochs, batch_size=batch_size)

    x_test_data_set, y_test_data_set, scaler = prepare_data(df_test, max_marker, lag)
    y_predicted = predict(model, x_test_data_set, scaler).flatten()

    print('MAPE: ', get_mape(y_test_data_set, y_predicted))
    print('RMSE: ', get_rmse(y_test_data_set, y_predicted))

    predicted_values = y_predicted.flatten()
    real_df = df_test[:len(predicted_values)]

    plot(real_df, y_predicted, max_marker=max_marker, lag_marker=lag, title=title,
         x_label=x_label, y_label=y_label)
