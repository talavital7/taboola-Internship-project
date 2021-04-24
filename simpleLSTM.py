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
        csv_to_data = [read_csv_to_df(data_path, cores_32, matric) for matric in self.DATA]
        data_per_cores = reduce(lambda left, right: pd.merge(left, right, on=['dates'], how='outer'), csv_to_data)
        self.data_per_cores = data_per_cores.dropna()
        # extracting dates
        self.dates_to_test = self.data_per_cores['dates']
        # adding new features and pick the best for the model
        data_without_dates = self.add_features(self.data_per_cores)
        # dropping dates
        data_without_dates = data_without_dates.drop('dates', 1)
        # dropping cpu util
        self.data_without_dates = data_without_dates.drop('cpu_user_util', 1)



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
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.X_train = X_train
        self.X_test = X_test
        # input_shape = (second_normalized_data_to_input.shape[0], second_normalized_data_to_input.shape[1])
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test = X_test.reshape((X_test.shape[0] // self.timesteps_to_the_future, self.timesteps_to_the_future, X_test.shape[1]))
        dates_of_predict = self.dates_to_test[X_train.shape[0]:]
        dates_of_predict = dates_of_predict.values
        self.dates_of_predict = dates_of_predict.reshape((dates_of_predict.shape[0], 1))
        model = Sequential()
        model.add(LSTM(20, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), recurrent_activation='hard_sigmoid'))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=[metrics.mae])
        model.fit(X_train, Y_train, epochs=30, batch_size=32, verbose=2)
        self.predict = model.predict(X_test, verbose=1)
        self.model = model
        self.X_train = X_train
        self.Y_train = Y_train



    def add_multiply(self,dataset):
        feature_names1 = dataset.columns[1:]
        feature_names2 = dataset.columns[1:]
        for feature1 in feature_names1:
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
        for feature in feature_names[1:]:
            i += 1
            x = dataset[feature]
            trend = [b - a for a, b in zip(x[::1], x[1::1])]
            trend.append(0)
            dataset["trend_" + feature] = trend
        return dataset

    def drop_low_corr_feature(self, dataset):
        corr = dataset.corr()["cpu_user_util"]
        corr = corr.abs()
        print(corr)
        for name in dataset.columns:
            if (name != "dates" and corr[name] < 0.8):
                dataset.drop(columns=[name], inplace=True)
        return dataset

    def add_features(self, dataset):
        dataset = self.add_isWeekend_feature(dataset)
        dataset = self.add_trend(dataset)
        dataset = self.add_multiply(dataset)
        dataset = self.drop_low_corr_feature(dataset)
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

        #Cross validation plot
        Fig.show()
        plt.figure(1)
        plt.scatter(Y_test, predict)
        plt.title(' Cross validation on CPU utilization for ' + cores_32)
        plt.show(block=False)
        #predict VS real plot
        plt.figure(2)
        Real, = plt.plot(Y_test)
        Predict, = plt.plot(predict)
        plt.title('CPU utilization for ' + cores_32)
        plt.legend([Predict, Real], ["Predicted", "Real"])
        plt.show(block=False)

        # loss and accuracy plots:
        # Compile model
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # Fit the model
        # self.second_normalized_data_to_input, self.cpu_user_util_to_input,
        X = self.second_normalized_data_to_input.reshape((self.second_normalized_data_to_input.shape[0], 1, self.second_normalized_data_to_input.shape[1]))
        history = self.model.fit(X,self.cpu_user_util_to_input, validation_data=(self.X_test, self.Y_test),
                                 validation_split=0.33, epochs=50, batch_size=32, verbose=0)
        # summarize history for accuracy
        print(history.history.keys())
        plt.figure(3)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show(block=False)
        # summarize history for loss
        plt.figure(4)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()




cores_32 = '32 cores 125.6 GB'
cores_40 = '40 cores 187.35 GB'
cores_48 = '48 cores 187.19 GB'
data_path = 'Data/singleServer/AM/'
model = MyModel()
model.data_prep(data_path, cores_48)
# model.heatMapCorrelation()
model.normalize()
model.build_model()
model.ploting(cores_48)
