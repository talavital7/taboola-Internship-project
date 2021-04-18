import os, glob
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from functools import reduce
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import metrics


def read_csv_to_df(data_path, core_path, matric):
    all_files = glob.glob(os.path.join(data_path + core_path +'/'+ matric, "*.csv"))
    all_csv = (pd.read_csv(f, sep=',') for f in all_files)
    data = pd.concat(all_csv, ignore_index=True)
    data.columns = ['dates', matric]
    return data

timesteps_to_the_future=1
cores_32 = '32 cores 125.6 GB'
cores_40 = '40 cores 187.35 GB'
cores_48 = '48 cores 187.19 GB'
data_path = 'Data/singleServer/AM/'
DATA=['avg_cpu_load',
      'cpu_user_util',
      'max_cpu_load',
      'p99_response_time',
      'reco_rate',
      'load_score_meter',
       # 'avg_memory',
      # 'avg_num_cores',
      # 'max_heap',
      # 'avg_heap',
      ]

csv_to_data = [read_csv_to_df(data_path, cores_32, matric) for matric in DATA]
data_per_cores = reduce(lambda left, right: pd.merge(left, right, on=['dates'], how='outer'), csv_to_data)
data_per_cores = data_per_cores.dropna()
dates_to_test = data_per_cores['dates']
data_without_dates = data_per_cores.drop('dates', 1)
data_without_dates = data_without_dates.drop('cpu_user_util', 1)

#TODO:put in function
sc = MinMaxScaler()
sc.fit(data_without_dates)
second_normalized_data_to_input = sc.fit_transform(data_without_dates)
data_to_predict_cpu_user_util = data_per_cores['cpu_user_util']
data_to_predict_cpu_user_util_reshape = data_to_predict_cpu_user_util.values.reshape(-1, 1)
cpu_user_util_to_input = sc.fit_transform(data_to_predict_cpu_user_util_reshape)


X_train, X_test, Y_train, Y_test = train_test_split(second_normalized_data_to_input, cpu_user_util_to_input, test_size = 0.25)
input_shape = (second_normalized_data_to_input.shape[0], second_normalized_data_to_input.shape[1])
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0]//timesteps_to_the_future, timesteps_to_the_future, X_test.shape[1]))
dates_of_predict = dates_to_test[X_train.shape[0]:]
dates_of_predict = dates_of_predict.values
dates_of_predict = dates_of_predict.reshape((dates_of_predict.shape[0],1))




model = Sequential()
model.add(LSTM(20, activation='relu', input_shape=(1,5), recurrent_activation='hard_sigmoid'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=[metrics.mae])
model.fit(X_train, Y_train, epochs=50, batch_size=1, verbose=2)
predict = model.predict(X_test, verbose=1)


# fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
# fig.add_trace(go.Scatter(x=dates_of_predict, y=Y_test, name='real'),row=1,col=1)
# fig.add_trace(go.Scatter(x=dates_of_predict, y=predict, name='predict'),row=1,col=1)
# fig.show()

Y_test = pd.DataFrame({'vals': Y_test[:, 0]})
predict = pd.DataFrame({'vals': predict[:, 0]})
dates_of_predict = pd.DataFrame({'vals': dates_of_predict[:, 0]})


Fig = go.Figure()
Fig.add_trace(go.Scatter(x=dates_of_predict['vals'], y=Y_test['vals'],
                             name = 'real data',
                             mode = 'markers+lines',
                             line=dict(shape='linear'),
                             connectgaps=True
                             ))
Fig.add_trace(go.Scatter(x=dates_of_predict['vals'], y=predict['vals'],
                             name = 'predicted data',
                             mode = 'markers+lines',
                             line=dict(shape='linear'),
                             connectgaps=True
                             ))
Fig.show()

plt.figure(1)
plt.scatter(Y_test, predict)
plt.title('CPU utilization for '+cores_32)
plt.show(block=False)

plt.figure(2)
Real, = plt.plot(Y_test)
Predict, = plt.plot(predict)
plt.title('CPU utilization for '+cores_32)
plt.legend([Predict, Real], ["Predicted", "Real"])
plt.show()

