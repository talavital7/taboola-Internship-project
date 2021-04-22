import os, glob
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
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


class MyModel:
    def __init__(self):
        self.DATA = ['avg_cpu_load',
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
        self.timesteps_to_the_future = 1

    def data_prep(self, data_path, cores_32):
        # global cores_32, data_per_cores, dates_to_test, data_without_dates
        # cores_32 = '32 cores 125.6 GB'
        # cores_40 = '40 cores 187.35 GB'
        # cores_48 = '48 cores 187.19 GB'
        # data_path = 'Data/singleServer/AM/'
        # DATA = ['avg_cpu_load',
        #         'cpu_user_util',
        #         'max_cpu_load',
        #         'p99_response_time',
        #         'reco_rate',
        #         'load_score_meter',
        #         # 'avg_memory',
        #         # 'avg_num_cores',
        #         # 'max_heap',
        #         # 'avg_heap',
        #         ]
        csv_to_data = [read_csv_to_df(data_path, cores_32, matric) for matric in self.DATA]
        data_per_cores = reduce(lambda left, right: pd.merge(left, right, on=['dates'], how='outer'), csv_to_data)
        self.data_per_cores = data_per_cores.dropna()
        self.dates_to_test = self.data_per_cores['dates']
        data_without_dates = self.data_per_cores.drop('dates', 1)
        self.data_without_dates = data_without_dates.drop('cpu_user_util', 1)

    # cores_32 = '32 cores 125.6 GB'
    # cores_40 = '40 cores 187.35 GB'
    # cores_48 = '48 cores 187.19 GB'
    # data_path = 'Data/singleServer/AM/'
    # data_prep(data_path, cores_32)


    def normalize(self):
        # global second_normalized_data_to_input, cpu_user_util_to_input
        sc = MinMaxScaler()
        sc.fit(self.data_without_dates)
        self.second_normalized_data_to_input = sc.fit_transform(self.data_without_dates)
        data_to_predict_cpu_user_util = self.data_per_cores['cpu_user_util']
        data_to_predict_cpu_user_util_reshape = data_to_predict_cpu_user_util.values.reshape(-1, 1)
        self.cpu_user_util_to_input = sc.fit_transform(data_to_predict_cpu_user_util_reshape)


    #TODO:put in function
    # normalize()


    def build_model(self):
        X_train, X_test, Y_train, Y_test = train_test_split(self.second_normalized_data_to_input, self.cpu_user_util_to_input,
                                                            test_size=0.25)
        self.Y_test = Y_test
        # input_shape = (second_normalized_data_to_input.shape[0], second_normalized_data_to_input.shape[1])
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test = X_test.reshape((X_test.shape[0] // self.timesteps_to_the_future, self.timesteps_to_the_future, X_test.shape[1]))
        dates_of_predict = self.dates_to_test[X_train.shape[0]:]
        dates_of_predict = dates_of_predict.values
        self.dates_of_predict = dates_of_predict.reshape((dates_of_predict.shape[0], 1))
        model = Sequential()
        model.add(LSTM(20, activation='relu', input_shape=(1, 5), recurrent_activation='hard_sigmoid'))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=[metrics.mae])
        model.fit(X_train, Y_train, epochs=10, batch_size=32, verbose=2)
        self.predict = model.predict(X_test, verbose=1)

    def add_multiply(self,dataset):
        feature_names1 = dataset.columns
        feature_names2 = dataset.columns
        for feature1 in feature_names1:
            feature_names2 = feature_names2[1:]
            for feature2 in feature_names2:
                if (feature1 != feature2 and feature1 != "dates" and feature2 != "dates" and feature1 != "cpu_user_util" and feature2 != "cpu_user_util"):
                    to_add = dataset[feature1] * dataset[feature2]
                    dataset[feature1 + " * " + feature2] = to_add
        return dataset

    def add_isWeekend_feature(self,dataset):
        dataset['is_weekend'] = dataset['dates'].str.split(' ', expand=True)[0]
        dataset['is_weekend'] = pd.to_datetime(dataset['is_weekend'], format='%Y-%m-%d')
        dataset['is_weekend'] = dataset['is_weekend'].dt.dayofweek
        is_weekend = dataset['is_weekend'].apply(lambda x: 1 if x >= 5.0 else 0)
        dataset['is_weekend'] = is_weekend
        return dataset

    def add_trend(self, dataset):
        feature_names = dataset.columns
        i = 0
        for feature in feature_names:
            i += 1
            x = dataset[feature]
            trend = [b - a for a, b in zip(x[::1], x[1::1])]
            trend.append(0)
            dataset["trend_" + feature] = trend
        return dataset

    def drop_low_corr_feature(self, dataset):
        # prediction = dataset["success_action"][args.time_steps:]
        # dataset_to_corr = dataset[:-args.time_steps]
        # dataset_to_corr["prediction"] = prediction
        corr = dataset.corr()["cpu_user_util"]
        corr = corr.abs()

        for name in dataset.columns:
            if (name != "time" and name != "date" and corr[name] < 0.8):
                dataset.drop(columns=[name], inplace=True)
        return dataset

    # build_model()
    def heatMapCorrelation(self):
        data = self.add_isWeekend_feature(self.data_per_cores)
        data_without_dates = data.drop('dates', 1)
        sc = MinMaxScaler()
        sc.fit(data_without_dates)
        dataToCheck = sc.fit_transform(data_without_dates)
        dataToCheckDf= pd.DataFrame(dataToCheck)
        dataToCheckDf.columns = self.DATA+['is_weekend']
        dataToHeatMap= self.add_multiply(dataToCheckDf)
        dataToHeatMap = self.drop_low_corr_feature(dataToHeatMap)
        # dataToHeatMap = self.add_trend(dataToCheckDf)
        # dataToHeatMap = self.add_multiply(dataToHeatMap)
        # dataToHeatMap = dataToCheckDf
        heatMap = sns.heatmap(dataToHeatMap.corr(), annot=True, cmap='coolwarm')
        heatMap.set_title('Correlation Heatmap', fontdict={'fontsize': 12}, pad=12)
        plt.show()
        Fig = go.Figure()
        trace = go.Heatmap(z=dataToCheckDf.corr().values,
                           x=dataToCheckDf.corr().index.values,
                           y=dataToCheckDf.corr().columns.values, colorscale="YlOrRd")
        Fig.add_trace(trace)
        Fig.show()





    def ploting(self, cores_32):
        Y_test = pd.DataFrame({'vals': self.Y_test[:, 0]})
        predict = pd.DataFrame({'vals': self.predict[:, 0]})
        dates_of_predict = pd.DataFrame({'vals': self.dates_of_predict[:, 0]})
        Fig = go.Figure()
        Fig.add_trace(go.Scatter(x=dates_of_predict['vals'], y=Y_test['vals'],
                                 name='real data',
                                 mode='markers+lines',
                                 line=dict(shape='linear'),
                                 connectgaps=True
                                 ))
        Fig.add_trace(go.Scatter(x=dates_of_predict['vals'], y=predict['vals'],
                                 name='predicted data',
                                 mode='markers+lines',
                                 line=dict(shape='linear'),
                                 connectgaps=True
                                 ))
        Fig.show()
        plt.figure(1)
        plt.scatter(Y_test, predict)
        plt.title('CPU utilization for ' + cores_32)
        plt.show(block=False)
        plt.figure(2)
        Real, = plt.plot(Y_test)
        Predict, = plt.plot(predict)
        plt.title('CPU utilization for ' + cores_32)
        plt.legend([Predict, Real], ["Predicted", "Real"])
        plt.show()


    # ploting()


cores_32 = '32 cores 125.6 GB'
cores_40 = '40 cores 187.35 GB'
cores_48 = '48 cores 187.19 GB'
data_path = 'Data/singleServer/AM/'
model = MyModel()
model.data_prep(data_path, cores_48)
model.heatMapCorrelation()
# model.normalize()
# model.build_model()
# model.ploting(cores_32)