import os, glob
from datetime import time
from keras import backend
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
from functools import reduce
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Dropout, RepeatVector, TimeDistributed, Activation
from keras import metrics
import argparse
from sklearn.model_selection import GridSearchCV
from tensorflow import estimator
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor


def read_csv_to_df(data_path, core_path, matric):
    # sub_folders = [name for name in os.listdir(os.path.join(data_path + core_path))]
    all_files = glob.glob(os.path.join(data_path + core_path + '/' + matric, "*.csv"))
    all_csv = (pd.read_csv(f, sep=',') for f in all_files)
    data = pd.concat(all_csv, ignore_index=True)
    data.columns = ['dates', matric]
    return data


def merge_and_drop_dups(left, right):
    left = pd.merge(left, right, on=['dates'], how='inner')
    left.drop_duplicates(inplace=True)
    return left


def make_time_steps_data(values, n_time_steps):
    # split into input and outputs - the last column will be the target metric
    values_to_train = values[:len(values) - n_time_steps, :-1]
    # the future
    values_to_test = values[n_time_steps:, -1]
    return values_to_train, values_to_test


def drop_low_corr_feature(dataset):
    corr = dataset.corr()["cpu_user_util"]
    corr = corr.abs()
    print(corr)
    for name in dataset.columns:
        if name != "dates" and (corr[name] < 0.8 or np.isnan(corr[name])):
            dataset.drop(columns=[name], inplace=True)
    return dataset


def add_trend(dataset):
    feature_names = dataset.columns
    i = 0
    for feature in feature_names[1:]:
        i += 1
        x = dataset[feature]
        trend = [b - a for a, b in zip(x[::1], x[1::1])]
        trend.append(0)
        dataset["trend_" + feature] = trend
    return dataset


def add_multiply(dataset):
    feature_names1 = dataset.columns[1:]
    feature_names2 = dataset.columns[1:]
    for feature1 in feature_names1:
        for feature2 in feature_names2:
            if (feature1 != feature2 and feature1 != "dates" and feature2 != "dates"
                    and feature1 != "cpu_user_util" and feature2 != "cpu_user_util"):
                to_add = dataset[feature1] * dataset[feature2]
                dataset[feature1 + " * " + feature2] = to_add
    return dataset


def add_isWeekend_feature(dataset):
    dataset['is_weekend'] = dataset['dates'].str.split(' ', expand=True)[0]
    dataset['is_weekend'] = pd.to_datetime(dataset['is_weekend'], format='%Y-%m-%d')
    dataset['is_weekend'] = dataset['is_weekend'].dt.dayofweek
    is_weekend = dataset['is_weekend'].apply(lambda x: 1 if x >= 5.0 else 0)
    dataset['is_weekend'] = is_weekend
    return dataset


def split_train_test(n_time_steps, values, train_size):
    values_X, values_y = make_time_steps_data(values, n_time_steps)

    n_train_hours = int((len(values_X)) * train_size)
    train_X = values_X[:n_train_hours, :]
    train_y = values_y[:n_train_hours]

    test_X = values_X[n_train_hours:, :]
    test_y = values_y[n_train_hours:]

    # reshape input to be 3D [samples, timesteps, features]

    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

    return train_X, train_y, test_X, test_y


class MyModel:
    def __init__(self, timesteps_to_the_future):
        # self.DATA = ['avg_cpu_load',
        # 		'max_cpu_load',
        # 		'p99_response_time',
        # 		'reco_rate',
        # 		'load_score_meter',
        # 		'cpu_user_util',
        # 		# 'avg_memory',
        # 		# 'avg_num_cores',
        # 		# 'max_heap',
        # 		# 'avg_heap',
        # 		]
        self.timesteps_to_the_future = timesteps_to_the_future

    def data_prep(self, data_path, cores):
        self.DATA = [name for name in os.listdir(os.path.join(data_path + cores))]
        csv_data_cores = [read_csv_to_df(data_path, cores, metric) for metric in self.DATA]
        data_per_cores = reduce(lambda left, right: merge_and_drop_dups(left, right), csv_data_cores)
        data_per_cores.dropna(inplace=True)
        data_per_cores.drop_duplicates(subset=['dates'], inplace=True)
        data_per_cores.set_index('dates', inplace=True)
        data_per_cores = data_per_cores.sort_values(by=['dates'])
        data_per_cores.reset_index(inplace=True)
        self.data_per_cores = data_per_cores
        # extracting dates
        self.dates_to_test = self.data_per_cores['dates']
        # adding new features and pick the best for the model
        data_without_dates = self.add_features(self.data_per_cores)
        corr = data_without_dates.corr()["cpu_user_util"]
        # dropping dates
        data_without_dates = data_without_dates.drop('dates', 1)
        c_u_u = data_without_dates['cpu_user_util']
        data_without_dates.drop('cpu_user_util', 1, inplace=True)
        data_without_dates.insert(data_without_dates.shape[1], 'cpu_user_util', c_u_u)
        # dropping cpu util
        self.data_without_dates = data_without_dates  # .drop('cpu_user_util', 1)

    def normalize(self):
        # global second_normalized_data_to_input, cpu_user_util_to_input
        self.sc = MinMaxScaler()
        self.sc.fit(self.data_without_dates)
        self.second_normalized_data_to_input = self.sc.fit_transform(self.data_without_dates)
        data_to_predict_cpu_user_util = self.data_per_cores['cpu_user_util']
        data_to_predict_cpu_user_util_reshape = data_to_predict_cpu_user_util.values.reshape(-1, 1)
        self.cpu_user_util_to_input = self.sc.fit_transform(data_to_predict_cpu_user_util_reshape)

    # TODO:put in function
    # normalize()

    def build_model(self):
        self.X_train, self.Y_train, self.X_test, self.Y_test = split_train_test(self.timesteps_to_the_future,
                                                                                self.second_normalized_data_to_input,
                                                                                0.75)

        # self.Y_train.tofile('Y_train.csv',sep=',')
        # self.X_train.tofile('X_train.csv',sep=',')
        # self.X_test.tofile('X_test.csv',sep=',')
        # self.Y_test.tofile('Y_test.csv',sep=',')
        dates_of_predict = self.dates_to_test[self.X_train.shape[0]:]
        dates_of_predict = dates_of_predict.values
        self.dates_of_predict = dates_of_predict.reshape((dates_of_predict.shape[0], 1))
        model = Sequential()
        # newdf = self.X_train
        # newdf['Pred'] = self.Y_train
        #num_of_features = self.X_train.shape[2]
        #model.add(LSTM(20, activation='relu', input_shape=(self.X_train.shape[1], num_of_features),
        #               recurrent_activation='hard_sigmoid'))
        #model.add(Dense(1))
        model.add(LSTM(units=128, input_shape=(self.X_train.shape[1], self.X_train.shape[2])))
        model.add(Dropout(rate=0.2))
        model.add(RepeatVector(n=self.X_train.shape[1]))
        model.add(LSTM(units=128, return_sequences=True))
        model.add(Dropout(rate=0.2))
        model.add(TimeDistributed(Dense(self.X_train.shape[2])))
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=[metrics.mae, 'accuracy'])
        # TODO: consider adding early stopping.
        model.fit(self.X_train, self.Y_train, epochs=30, batch_size=64, verbose=2)
        self.predict = model.predict(self.X_test)
        self.model = model

    def show_future_train_test(self):
        # make predictions
        trainPredict = self.model.predict(self.X_train)
        testPredict = self.model.predict(self.X_test)
        # invert predictions
        trainPredict = self.sc.inverse_transform(trainPredict)
        trainY = self.sc.inverse_transform([self.Y_train])
        testPredict = self.sc.inverse_transform(testPredict)
        testY = self.sc.inverse_transform([self.Y_test])
        # calculate root mean squared error
        # trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
        # print('Train Score: %.2f RMSE' % (trainScore))
        # testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
        # print('Test Score: %.2f RMSE' % (testScore))
        # shift train predictions for plotting
        trainPredictPlot = np.empty_like(self.cpu_user_util_to_input)
        trainPredictPlot[:, :] = np.nan
        trainPredictPlot[self.timesteps_to_the_future:len(trainPredict) + self.timesteps_to_the_future,
        :] = trainPredict
        # shift test predictions for plotting
        testPredictPlot = np.empty_like(self.cpu_user_util_to_input)
        testPredictPlot[:, :] = np.nan
        testPredictPlot[len(trainPredict) + (self.timesteps_to_the_future):, :] = testPredict
        # plot baseline and predictions
        raw_cpu_util = self.sc.inverse_transform(self.cpu_user_util_to_input)
        plt.plot(raw_cpu_util)
        plt.plot(trainPredictPlot)
        plt.plot(testPredictPlot)
        plt.show()
        Fig = go.Figure()

        dates_of_predict = pd.DataFrame({'vals': self.dates_to_test})
        Fig.add_trace(go.Scatter(x=dates_of_predict['vals'], y=trainPredictPlot.reshape(-1),
                                 name='train predict',
                                 mode='markers+lines',
                                 line=dict(shape='linear'),
                                 connectgaps=False
                                 ))
        Fig.add_trace(go.Scatter(x=dates_of_predict['vals'], y=testPredictPlot.reshape(-1),
                                 name='test preditct',
                                 mode='markers+lines',
                                 line=dict(shape='linear'),
                                 connectgaps=False
                                 ))
        Fig.add_trace(go.Scatter(x=dates_of_predict['vals'], y=raw_cpu_util.reshape(-1),
                                 name='real data',
                                 mode='markers+lines',
                                 line=dict(shape='linear'),
                                 connectgaps=False
                                 ))

        Fig.show()

    def add_features(self, dataset):
        dataset = add_isWeekend_feature(dataset)
        dataset = add_trend(dataset)
        dataset = add_multiply(dataset)
        dataset = drop_low_corr_feature(dataset)
        return dataset

    # build_model()
    def heat_map_correlation(self):
        data = add_isWeekend_feature(self.data_per_cores)
        data_without_dates = data.drop('dates', 1)
        sc = MinMaxScaler()
        sc.fit(data_without_dates)
        dataToCheck = sc.fit_transform(data_without_dates)
        dataToCheckDf = pd.DataFrame(dataToCheck)
        dataToCheckDf.columns = self.DATA + ['is_weekend']
        dataToHeatMap = add_multiply(dataToCheckDf)
        dataToHeatMap = drop_low_corr_feature(dataToHeatMap)
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

    def plot(self, title):
        # Y_test = pd.DataFrame({'vals': self.Y_test[:, 0]})
        # predict = pd.DataFrame({'vals': self.predict[:, 0]})
        dates_of_predict = pd.DataFrame({'vals': self.dates_of_predict[:, 0]})
        Fig = go.Figure()
        Fig.add_trace(go.Scatter(x=dates_of_predict['vals'], y=self.Y_test.reshape(-1),
                                 name='real data',
                                 mode='markers+lines',
                                 line=dict(shape='linear'),
                                 connectgaps=False
                                 ))
        Fig.add_trace(go.Scatter(x=dates_of_predict['vals'], y=self.predict.reshape(-1),
                                 name='predicted data',
                                 mode='markers+lines',
                                 line=dict(shape='linear'),
                                 connectgaps=False
                                 ))
        # Fig.show()
        # Cross validation plot
        Fig.show()
        plt.figure(1)
        plt.scatter(self.Y_test, self.predict)
        plt.title(' Cross validation on CPU utilization for ' + title)
        plt.show(block=False)
        # predict VS real plot
        plt.figure(2)
        Real, = plt.plot(self.Y_test)
        Predict, = plt.plot(self.predict)
        plt.title('CPU utilization for ' + title)
        plt.legend([Predict, Real], ["Predicted", "Real"])
        plt.show(block=False)

    # TODO: Fix the dimensions in the reshape
    # accuracy and loss
    # self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy', metrics.mae])
    # # Fit the model
    # X = self.second_normalized_data_to_input.reshape((self.second_normalized_data_to_input.shape[0], 1, self.second_normalized_data_to_input.shape[1]))
    # history = self.model.fit(self.X_train, self.cpu_user_util_to_input, validation_split=0.25, epochs=20, batch_size=128, verbose=0)
    # # list all data in history
    # print(history.history.keys())
    # # summarize history for accuracy
    # plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    # # summarize history for loss
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model accuracy and loss')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()

    def grids(self, time_steps):
        # LSTM  Input Shape
        time_steps = 1  # number of time-steps you are feeding a sequence (?)
        inputs_numb = self.X_train.shape[1]  # number of inputs
        input_shape = [time_steps, inputs_numb]
        # ,input_shape=input_shape
        model = KerasRegressor(build_fn=create_model, verbose=1)

        # GridSearch code
        start = time()
        optimizers = ['adam', 'Adamax', 'Nadam']
        epochs = [30, 50, 100]
        hl1_nodes = [10, 20, 50]
        btcsz = [32, 64, 128]
        # learning_rate= [0.001, 0.01, 0.0001]

        param_grid = dict(optimizer=optimizers, hl1_nodes=hl1_nodes, nb_epoch=epochs, batch_size=btcsz)
        scoring = make_scorer(accuracy_score)  # in order to use a metric as a scorer
        # scoring = estimator.score(self.X_test,self.Y_test)
        grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_absolute_error', n_jobs=-1,
                            verbose=3)
        grid_result = grid.fit(self.X_train, self.Y_train)

        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        # for params, mean_score, scores in grid_result.cv_results_:
        # 	print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))
        for x in grid_result.cv_results_:
            print(x)
        print("total time:", time() - start)


def create_model(optimizer, hl1_nodes):
    # creation of the NN - Electric Load
    # LSTM layers followed by other LSTM layer must have the parameter "return_sequences" set at True
    model = Sequential()
    model.add(LSTM(units=hl1_nodes, input_shape=(1, 43), return_sequences=False))
    model.add(Dense(1))  # output layer
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])
    model.summary()
    return model


def main(arguments):
    # cores_32 = '32 cores 125.6 GB'
    cores_40 = '40 cores 187.35 GB'
    # cores_48 = '48 cores 187.19 GB'

    model = MyModel(arguments.timesteps_to_the_future)
    model.data_prep(arguments.path, cores_40)
    # model.heat_map_correlation()
    model.normalize()
    model.build_model()
    # model.grids(arguments.timesteps_to_the_future)
    model.show_future_train_test()
    model.plot(cores_40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LSTM supervised')
    parser.add_argument('--path', dest='path', type=str, required=True, help='path to files')
    parser.add_argument('--timesteps_to_the_future', dest='timesteps_to_the_future', type=int, required=True,
                        help='timesteps to predict', default=6)
    args = parser.parse_args()
    main(args)
