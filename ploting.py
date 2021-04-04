import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly
import plotly.graph_objs as go

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
                            header=0) for filename in all_filenames),
            axis=0, ignore_index=True)
        return frame

class MyModel:


    # where target is the data you want to predict
    def __init__(self):
        self.DATA = ('avgCPUloadin15mwindow',
                     'avgheap-oldgen20mwindow',
                     'avgMemoryinGB',
                     'avgnumberofcores',
                     'maxCPUloadin15mwindow',
                     'maxheap-oldgen20mwindow',
                      'Recommendationratein5minwindow',
                      'The99thPercentileOfResponseTimeToArecommenationRequest')

    # fetching the data from path,
    # where path is the main folder containing the data.
    def fetch_data(self, path):
        reader = Reader(path=path + '//' + self.DATA[0])
        # self.target = np.array([reader.read().loc[:, 'y']])
        self.feature = []
        self.xlx =[]
        for i in range(0, len(self.DATA)):
            reader.set_path(path=path + '//' + self.DATA[i])
            self.feature.append(reader.read())
    def present_raw_data_plotly(self):
        trace0 = go.Scatter(x=self.newfeature[0]['ds'],y=self.newfeature[0][0],name=self.DATA[0])
        trace1 = go.Scatter(x=self.newfeature[1]['ds'], y=self.newfeature[1][0], name=self.DATA[1])
        trace2 = go.Scatter(x=self.newfeature[2]['ds'], y=self.newfeature[2][0], name=self.DATA[2])
        trace3 = go.Scatter(x=self.newfeature[3]['ds'], y=self.newfeature[3][0], name=self.DATA[3])
        trace4 = go.Scatter(x=self.newfeature[4]['ds'], y=self.newfeature[4][0], name=self.DATA[4])
        trace5 = go.Scatter(x=self.newfeature[5]['ds'], y=self.newfeature[5][0], name=self.DATA[5])
        trace6 = go.Scatter(x=self.newfeature[6]['ds'], y=self.newfeature[6][0], name=self.DATA[6])
        trace7 = go.Scatter(x=self.newfeature[7]['ds'], y=self.newfeature[7][0], name=self.DATA[7])
        data_tr = [trace0, trace1, trace2 ,trace3, trace4, trace5, trace6, trace7]

        fig=go.Figure(data_tr,layout={'title': 'matrics to Evaluate'})
        iplot(fig,show_link=False)




        # sns.lineplot(x='ds',y=0,label=self.DATA[0],data=self.newfeature[0]).set(xlabel='dates-5minjumps ', ylabel='values', title='metrics ploting')
        # sns.lineplot(x='ds', y=0,label=self.DATA[1], data=self.newfeature[1])
        # sns.lineplot(x='ds', y=0,label=self.DATA[2], data=self.newfeature[2])
        # sns.lineplot(x='ds', y=0,label=self.DATA[3], data=self.newfeature[3])
        # sns.lineplot(x='ds', y=0,label=self.DATA[4], data=self.newfeature[4])
        # sns.lineplot(x='ds', y=0,label=self.DATA[5], data=self.newfeature[5])
        # sns.lineplot(x='ds', y=0, label=self.DATA[6], data=self.newfeature[6])
        # sns.lineplot(x='ds', y=0, label=self.DATA[7], data=self.newfeature[7])
        # plt.show()

    # showing the raw data without interpretation
    # TODO find out how to make it in loop
    def present_raw_data(self):

        sns.lineplot(x='ds',y=0,label=self.DATA[0],data=self.newfeature[0]).set(xlabel='dates-5minjumps ', ylabel='values', title='metrics ploting')
        sns.lineplot(x='ds', y=0,label=self.DATA[1], data=self.newfeature[1])
        sns.lineplot(x='ds', y=0,label=self.DATA[2], data=self.newfeature[2])
        sns.lineplot(x='ds', y=0,label=self.DATA[3], data=self.newfeature[3])
        sns.lineplot(x='ds', y=0,label=self.DATA[4], data=self.newfeature[4])
        sns.lineplot(x='ds', y=0,label=self.DATA[5], data=self.newfeature[5])
        sns.lineplot(x='ds', y=0, label=self.DATA[6], data=self.newfeature[6])
        sns.lineplot(x='ds', y=0, label=self.DATA[7], data=self.newfeature[7])
        plt.show()
        # plt.figure(1)
        # #T, = plt.plot(self.Y[0, :])
        # F1, = plt.plot(self.X[:,0])
        # F2, = plt.plot(self.X[:,1])
        # F3, = plt.plot(self.X[:,2])
        # F4, = plt.plot(self.X[:,3])
        # F5, = plt.plot(self.X[:,4])
        # F6, = plt.plot(self.X[:, 5])
        # plt.legend([F1, F2, F3, F4, F5, F6], (self.DATA))
        # plt.show()

    # preparing the data (Min max normalization and shape of the data)
    def concatenate_data(self):
        self.X = np.concatenate(self.feature)
        self.X = np.transpose(self.X)


        # self.Y = self.target
        # self.Y = np.transpose(self.Y)

    def normalize(self):
        self.newfeature=[]
        for f in self.feature:
            scaler = MinMaxScaler(feature_range=(0, 1))
            y=f[['y']]
            scaler.fit(y)
            y = scaler.transform(y)
            data=pd.DataFrame(y)
            data['ds']=f['ds']
            self.newfeature.append(data)


model = MyModel()
model.fetch_data('Data/crossServer/US')
# model.fetch_data('Data/singleServer/US/32 cores 125.64 GB')

# model.concatenate_data()
model.normalize()
# model.present_raw_data()
model.present_raw_data_plotly()