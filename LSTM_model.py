import os, glob
from functools import reduce
import pandas as pd
import tensorflow
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import metrics
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import argparse
import plotly.graph_objects as go
from keras import backend
import random
import numpy as np

# cross dc


avg_cpu_load_DC = '/avg(node_load15{hostname=~_^water._}) by (domain)'
avg_heap_DC = '/avg_heap'
avg_memory_Dc = '/avg(avg(node_memory_MemTotal_bytes{hostname=~_^water._})) by (hostname)'
avg_num_cores_Dc = '/avg(count (node_cpu_seconds_total{mode=_idle_,hostname=~_^water._,job=~_node_exporter_}) by (hostname))'
max_cpu_load_Dc = '/max(node_load15{hostname=~_^water._}) by (domain)'
max_heap_Dc = '/max_heap'
p99_response_time_Dc = '/trc_requests_timer_p99_weighted_dc'
reco_rate_Dc = '/recommendation_requests_5m_rate_dc'

paths_cross_dc = [[avg_cpu_load_DC, 'avg_cpu_load'], [avg_heap_DC, 'avg_heap'], [avg_memory_Dc, 'avg_memory']
    , [avg_num_cores_Dc, 'avg_num_cores'], [max_cpu_load_Dc, 'cpu_user_util'],
                  [max_cpu_load_Dc, 'max_cpu_load'], [max_heap_Dc, 'max_heap']
    , [p99_response_time_Dc, 'p99_response_time'], [reco_rate_Dc, 'reco_rate']]

# Data/Single servers/AM/40 cores 187.35 GB
data_path_servers = 'Data/Single servers'
data_path_cross_Dc = 'Data/Cross DC'
cores_32_path = '32 cores 125.6 GB'
cores_32_path = '32 cores 125.64 GB'
cores_40_path = '40 cores 187.35 GB'
cores_48_path = '48 cores 187.19 GB'
cores_72_path = '72 cores 251.63GB'
cores_40_path_copy = '40 cores 187.35 GB - Copy'
country_AM = '/AM/'
country_IL = '/IL/'
country_LA = '/LA/'
country_US = '/US/'

# gets dataframe with no dates and predicted matric name
# returns cartesian multiplication of metrics
def add_multiply(dataset, predict_metric_name):
    feature_names1 = dataset.columns
    feature_names2 = dataset.columns
    for feature1 in feature_names1:
        feature_names2 = feature_names2[1:]
        for feature2 in feature_names2:
            if feature1 != feature2 and feature1 != predict_metric_name and feature2 != predict_metric_name:
                to_add = dataset[feature1] * dataset[feature2]
                dataset[feature1 + " * " + feature2] = to_add
    return dataset


def mask_under_threshold(dataset, predict_metric_name, threshold):
    correlated_features = set()
    correlation_matrix = dataset.corr()
    j = 0
    for i in correlation_matrix[predict_metric_name]:
        if i < threshold:
            correlated_features.add(correlation_matrix[predict_metric_name].keys()[j])
        j = j + 1
    dataset.drop(labels=correlated_features, axis=1, inplace=True)
    return dataset


def add_trend(dataset):
    feature_names = ["p99_response_time"]
    i = 0
    for feature in feature_names:
        i += 1
        x = dataset[feature]
        trend = [b - a for a, b in zip(x[::1], x[1::1])]
        trend.append(0)
        dataset["trend_" + feature] = trend
    return dataset


def add_isWeekend_feature(dataset):
    dataset['is_weekend'] = dataset['dates'].str.split(' ', expand=True)[0]
    dataset['is_weekend'] = pd.to_datetime(dataset['is_weekend'], format='%Y-%m-%d')
    dataset['is_weekend'] = dataset['is_weekend'].dt.dayofweek
    is_weekend = dataset['is_weekend'].apply(lambda x: 1 if x >= 5.0 else 0)
    dataset['is_weekend'] = is_weekend
    return dataset


def getCsv(data_path, path, metric_path, name_of_metric,day):
    if data_path == data_path_cross_Dc:
        all_files = glob.glob(os.path.join(path + metric_path, "*.csv"))
    else:
        all_files = glob.glob(os.path.join(path + metric_path, "*.csv"))
    all_csv = (avg5minToDay(f,day) for f in all_files)
    new_csv = pd.concat(all_csv, ignore_index=True)
    new_csv.columns = ['dates', name_of_metric]
    return new_csv

def avg5minToDay(f,day):
    dateLength = 10
    date_and_time = 19
    df = pd.read_csv(f, sep=',')
    date = df['ds'][0]
    date = date[:dateLength]
    # mean = df['y'].mean()
    mean = df['y'].quantile(0.99)
    mead_df = pd.DataFrame({'ds':[date],'y':[mean]})
    if (day > 1):
        num_of_points = len(df['ds'])
        value_of_points = []
        dates_of_points = []
        for i in range(day):
            value_of_points.append(float(df['y'][(num_of_points // day)* i]))
            dates_of_points.append(df['ds'][(num_of_points // day) * i])
        points_df = pd.DataFrame({'ds': dates_of_points, 'y': value_of_points})
        # points_df['y'] = points_df['y'].str.replace('%', '').astype(np.float64)
        return points_df
    if (day == -1):
        return mead_df
    if (day == 0):
        return df


# reading metrics and pushing the predicted metric to the end
def get_paths(path, predict_metric_name):
    dirlist = [item for item in os.listdir(path) if os.path.isdir(os.path.join(path, item))]
    a = dirlist.index(predict_metric_name)
    b = len(dirlist) - 1
    dirlist[b], dirlist[a] = dirlist[a], dirlist[b]  # push the predict_metric_name to the end
    dirlist = [['/' + item, item] for item in dirlist]
    return dirlist


def getDataSet(predict_metric_name, path_org, data_path,day):
    paths = get_paths(path_org, predict_metric_name)
    csv_data_cores = [getCsv(data_path, path_org, path[0], path[1],day) for path in paths]
    csv_data_cores = reduce(lambda left, right: merge_and_drop_dups(left, right), csv_data_cores)
    none_tech_metrics = ['avg_memory', 'avg_num_cores','load_score_meter','max_heap'
                         ,'disk_space','avg_cpu_load']
    # none_tech_metrics = ['avg_memory', 'avg_num_cores','reco_rate','load_score_meter','avg_heap'
                         # ,'disk_space','avg_cpu_load','gc_time','disk_space','max_cpu_load','qps','p99_response_time']
    csv_data_cores.drop(none_tech_metrics, axis='columns', inplace=True)
    csv_data_cores.dropna(inplace=True)
    csv_data_cores.drop_duplicates(subset=['dates'], inplace=True)
    csv_data_cores.set_index('dates', inplace=True)
    csv_data_cores = csv_data_cores.sort_values(by=['dates'])
    csv_data_cores.reset_index(inplace=True)
    # print(len(csv_data_cores))
    # csv_data_cores = add_isWeekend_feature(csv_data_cores)
    # csv_data_cores = add_trend(csv_data_cores)
    # drop date
    data_witout_dates = csv_data_cores.drop('dates', 1)
    return data_witout_dates, csv_data_cores


def merge_and_drop_dups(left, right):
    left = pd.merge(left, right, on=['dates'], how='inner')
    left.drop_duplicates(inplace=True)
    return left


def scale(data_to_scale, predicted_metric):
    sc = MinMaxScaler()
    sc.fit(data_to_scale)
    data_to_scale = sc.fit_transform(data_to_scale)
    predicted_metric_reshape = predicted_metric.values.reshape(-1, 1)
    predicted_metric = sc.fit_transform(predicted_metric_reshape)
    return data_to_scale, predicted_metric


def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred-y_true), axis=1))

def model_settings(X_train, Y_train, arguments):
    model = Sequential()
    num_of_features = X_train.shape[2]
    model.add(LSTM(arguments.number_of_nodes, activation='tanh', input_shape=(1, num_of_features),
                   recurrent_activation='hard_sigmoid'))
    model.add(Dense(1))
    # model.compile(loss='mean_squared_error', optimizer='adam', metrics=[metrics.mae, 'accuracy'])
    model.compile(loss=rmse, optimizer='adam', metrics=[metrics.mae])
    model.fit(X_train, Y_train, epochs=arguments.epochs, batch_size=arguments.batch_size, verbose=2)
    return model

def create_model(optimizer='adam',loss=rmse,activation='relu',hl1_nodes=30):
    model = Sequential()
    # num_of_features = X_train.shape[2]
    model.add(LSTM(hl1_nodes, activation=activation, input_shape=(1, 13),
                   recurrent_activation='hard_sigmoid'))
    model.add(Dense(1))
    # model.compile(loss='mean_squared_error', optimizer='adam', metrics=[metrics.mae, 'accuracy'])
    model.compile(loss=loss, optimizer=optimizer, metrics=[metrics.mae])
    return model
def random_grid_search(X,Y,model_fn=create_model):
    kears_estimator = KerasRegressor(build_fn=create_model, verbose=1)
    optimizers = ['adam']  # , 'Adamax', 'Nadam']
    epochs = [30, 50, 100]
    hl1_nodes = [10, 20, 50]
    btcsz = [32, 64, 128]
    loss = [rmse, 'mean_squared_error']
    activation = ['relu', 'tanh', 'sigmoid']

    # learning_rate= [0.001, 0.01, 0.0001]

    param_grid = dict(optimizer=optimizers, hl1_nodes=hl1_nodes,
                      nb_epoch=epochs, batch_size=btcsz, activation=activation, loss=loss)
    n_iter_search = 80  # Number of parameter settings that are sampled.
    random_search = RandomizedSearchCV(estimator=kears_estimator,
                                       param_distributions=param_grid,
                                       n_iter=n_iter_search,scoring='neg_mean_absolute_error', n_jobs=-1,
                        verbose=3)
    random_search.fit(X, Y)

    # Show the results
    print("Best: %f using %s" % (random_search.best_score_, random_search.best_params_))
    means = random_search.cv_results_['mean_test_score']
    stds = random_search.cv_results_['std_test_score']
    params = random_search.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

def keras_grid_search(X,Y,model_fn=create_model):

    # kears_estimator = KerasClassifier(build_fn=model_fn, verbose=1)
    kears_estimator = KerasRegressor(build_fn=create_model, verbose=1)
    optimizers = ['adam','sgd']#, 'Adamax', 'Nadam']
    epochs = [30, 50, 100]
    hl1_nodes = [10,20, 50]
    btcsz = [32, 64, 128]
    loss=[rmse,'mean_squared_error']
    activation=['relu','tanh','sigmoid']

    # learning_rate= [0.001, 0.01, 0.0001]

    param_grid = dict(optimizer=optimizers, hl1_nodes=hl1_nodes,
                      nb_epoch=epochs, batch_size=btcsz,activation=activation,loss=loss)

    grid = GridSearchCV(estimator=kears_estimator, param_grid=param_grid,
                        scoring='neg_mean_absolute_error', n_jobs=-1,
                        verbose=3)

    grid_result = grid.fit(X, Y)

    # Show the results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

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


def make_time_steps_data(values, n_time_steps):
    # split into input and outputs - the last column will be the target metric
    values_to_train = values[:len(values) - n_time_steps, :-1]
    values_to_test = values[n_time_steps:, -1]
    return values_to_train, values_to_test






def add_moving_avg(data_set,arguments):
    feature_names1 = data_set.columns
    time_res = 288
    for feature in feature_names1:
        data_set[feature+"_p75"] = data_set[feature].rolling(window=time_res * 21 ).quantile(0.75)
        data_set[feature + "_p75"] = data_set[feature].rolling(window=time_res * 14).quantile(0.75)
        data_set[feature + "_p75"] = data_set[feature].rolling(window=time_res * 7).quantile(0.75)
    data_set.dropna(inplace=True)
    return data_set




def main(arguments):
    random.seed(100)
    np.random.seed(100)
    tensorflow.random.set_seed(100)
    # get_data with no date, and all data in csv
    # new_path = data_path_servers + country_AM + cores_40_path
    new_path = data_path_servers + country_US + cores_32_path
    data_to_scale_no_dates, csv_data_with_dates = getDataSet(arguments.predict_metric_name, new_path, data_path_servers,arguments.day)
    # save predicted metric
    cpu_user_util_csv = csv_data_with_dates['cpu_user_util']
    save_dates = csv_data_with_dates['dates']
    data_to_scale_no_dates = add_moving_avg(data_to_scale_no_dates, arguments)

    # scale data
    lst_of_features = list(data_to_scale_no_dates)
    scaler = MinMaxScaler()
    scaler.fit(data_to_scale_no_dates)
    data_to_scale_no_dates[lst_of_features] = scaler.fit_transform(data_to_scale_no_dates[lst_of_features])
    col_list = list(data_to_scale_no_dates)
    last_element = len(col_list) - 1
    col_list[data_to_scale_no_dates.columns.get_loc(arguments.predict_metric_name)], col_list[last_element] = \
        col_list[last_element], col_list[data_to_scale_no_dates.columns.get_loc(arguments.predict_metric_name)]
    data_to_scale_no_dates = data_to_scale_no_dates.reindex(columns=col_list)
    if (arguments.cartesian_multiplication):
        data_to_scale_no_dates = add_multiply(data_to_scale_no_dates, arguments.predict_metric_name)
        data_to_scale_no_dates = mask_under_threshold(data_to_scale_no_dates, arguments.predict_metric_name,
                                                      arguments.threshold)
        col_list = list(data_to_scale_no_dates)
        last_element = len(col_list) - 1
        col_list[data_to_scale_no_dates.columns.get_loc(arguments.predict_metric_name)], col_list[last_element] = \
        col_list[last_element], col_list[data_to_scale_no_dates.columns.get_loc(arguments.predict_metric_name)]
        data_to_scale_no_dates = data_to_scale_no_dates.reindex(columns=col_list)

    data_scaled_no_dates = data_to_scale_no_dates.to_numpy()

    # split into test & train
    X_train, Y_train, X_test, Y_test = split_train_test(arguments.timesteps_to_the_future, data_scaled_no_dates, 0.80)

    # create the lstm model
    # lstm_model = model_settings(X_train, Y_train, arguments)
    # predict = lstm_model.predict(X_test)
    #
    # #keras grid search
    # Y_train=Y_train.astype(np.float32)
    # keras_grid_search(X_train, Y_train, model_fn=create_model)
    # #random grid search
    random_grid_search(X_train, Y_train, model_fn=create_model)


    # fig = go.Figure([
    #
    #     go.Scatter(
    #         name='Real',
    #         x=save_dates.values[Y_train.shape[0]:].reshape(-1),
    #         y=Y_test.reshape(-1),
    #         mode='markers+lines',
    #         marker=dict(color='red', size=1),
    #         showlegend=True,
    #         connectgaps=False
    #
    #     ),
    #     go.Scatter(
    #         name='Predict - test',
    #         x=save_dates.values[Y_train.shape[0]:].reshape(-1),
    #         y=predict.reshape(-1),
    #         mode='lines',
    #         marker=dict(color="#444"),
    #         line=dict(width=1),
    #         showlegend=True,
    #         connectgaps=False
    #     )
    # ])
    # fig.update_layout(
    #     title=new_path + "\n" + "**predicted metric = " + arguments.predict_metric_name +
    #           ",multiply= " + str(arguments.cartesian_multiplication) + ",threshold= " + str(
    #         arguments.threshold) + " ,epohcs = " + str(arguments.epochs) +" epochs = "+ str(arguments.epochs) +" batch_size = "+ str(arguments.batch_size) +
    #           " day_res = " + str(arguments.day)+ ",!!! time steps = " + str(
    #         arguments.timesteps_to_the_future)+"!!! **",
    #     xaxis_title="dates",
    #     yaxis_title="vals",
    #     legend_title="Legend Title",
    # )
    #
    # fig.show()
    # pass


# Real, = plt.plot(Y_test)
# Predict, = plt.plot(predict)
# plt.title(country_AM + cores_40_path)
# plt.legend([Predict, Real], ["Predicted Data - CPU Util", "Real Data - CPU Util "])
# plt.show()
#
# plt.figure(2)
# plt.scatter(Y_test, predict)
# plt.show(block=False)
#
#
# Real, = plt.plot(save_dates.values[:Y_test.shape[0]],Y_test)
# Predict, = plt.plot(save_dates.values[:Y_test.shape[0]],predict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LSTM supervised')
    parser.add_argument('--timesteps_to_the_future', dest='timesteps_to_the_future', type=int, required=False,
                        help='timesteps to predict', default=288*7)
    parser.add_argument('--batch_size', dest='batch_size', type=int, required=False, help='batch size', default=32)
    parser.add_argument('--epochs', dest='epochs', type=int, required=False, help='epochs', default=50)
    parser.add_argument('--number_of_nodes', dest='number_of_nodes', type=int, required=False, help='number of nodes',
                        default=50)
    parser.add_argument('--predict_metric_name', dest='predict_metric_name', type=str, required=False,
                        help='predict metric name',
                        default='max_cpu_load')
    parser.add_argument('--cartesian_multiplication', dest='cartesian_multiplication', type=bool, required=False,
                        help='cartesian multiplication flag',
                        default=False)
    parser.add_argument('--threshold', dest='threshold', type=int, required=False,
                        help='threshold for cartesian multiplication corr with predicted metric ',
                        default=0.9)
    parser.add_argument('--day', dest='day', type=int, required=False,
                        help='-1  == p99, 0 == 5min, day > 1 == num of points  ',
                        default=0)
    args = parser.parse_args()
    main(args)