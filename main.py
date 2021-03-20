import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import argparse
from sklearn.preprocessing import MinMaxScaler

class Reader:
    # path is the path to the Main dataset folder
    def __init__(self, path):
        self.path = path

    # set the path again if you want to use the same reader to read another dataset
    def set_path(self, path):
        self.path = path

    # read all the csv file in self.path
    # return Data Frame containing all of the data
    # TODO add read from date to date (in a range) another method
    def read(self):
        all_filenames = glob.glob(self.path + "/*.csv")
        data_types = {'ds': str, 'y': float}
        parse_dates = ['ds']
        frame = pd.concat(
            (
                pd.read_csv(filename, dtype=data_types, parse_dates=parse_dates, date_parser=pd.to_datetime,
                            index_col=None,
                            header=0) for filename in all_filenames[0:1]),
            axis=0, ignore_index=True)
        return frame

class MyModel:

    # TODO get DATA as parameter and index for "target" data
    # where target is the data you want to predict
    def __init__(self):
        self.DATA = ('avgCPUloadin15mwindow',
                     # 'avgheap-oldgen20mwindow',
                     #'avgMemoryinGB',
                     #'avgnumberofcores',
                     'maxCPUloadin15mwindow',
                     # 'maxheap-oldgen20mwindow',
                      'Recommendationratein5minwindow',
                      'The99thPercentileOfResponseTimeToArecommenationRequest')

    # fetching the data from path,
    # where path is the main folder containing the data.
    def fetch_data(self, path):
        reader = Reader(path=path + '//' + self.DATA[0])
        # self.target = np.array([reader.read().loc[:, 'y']])
        self.feature = []
        for i in range(0, len(self.DATA)):
            reader.set_path(path=path + '//' + self.DATA[i])
            self.feature.append(np.array([reader.read().loc[:, 'y']]))

    # showing the raw data without interpretation
    # TODO find out how to make it in loop
    def present_raw_data(self):
        plt.figure(1)
        #T, = plt.plot(self.Y[0, :])
        F1, = plt.plot(self.X[:,0])
        F2, = plt.plot(self.X[:,1])
        F3, = plt.plot(self.X[:,2])
        F4, = plt.plot(self.X[:,3])
        # F5, = plt.plot(self.X[:,4])
        # F6, = plt.plot(self.X[:, 5])
        plt.legend([F1, F2, F3, F4], (self.DATA))
        plt.show()

    # preparing the data (Min max normalization and shape of the data)
    def concatenate_data(self):
        self.X = np.concatenate(self.feature)
        self.X = np.transpose(self.X)

        # self.Y = self.target
        # self.Y = np.transpose(self.Y)

    def normalize(self):
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(self.X)
        self.X = scaler.transform(self.X)

        # scaler1 = MinMaxScaler(feature_range=(0, 1))
        # scaler1.fit(self.Y)
        # self.Y = scaler1.transform(self.Y)


    # def prep_data(self):
    #     self.concatenate_data()
    #     self.normalize()


# reader = Reader(path='Data/crossServer/US/The99thPercentileOfResponseTimeToArecommenationRequest')
# target = np.array([reader.read().loc[:, 'y']])
# plt.figure(1)
# T, = plt.plot(target[0, :])
# plt.legend([T], ['The99thPercentileOfResponseTimeToArecommenationRequest'])
# plt.show()
# t='talavital'
model = MyModel()
model.fetch_data('Data/crossServer/US')
model.concatenate_data()
model.normalize()
model.present_raw_data()
name='talavital'
