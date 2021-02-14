import numpy as np
import numpy.linalg as linalg
from numpy import transpose as tt
from numpy import concatenate as cc
import matplotlib.pyplot as plt
from sklearn import linear_model
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Dense
import keras


#geometric brownian motion class to create a MGBM path and sample MGBM terminal values
class GeometricBrownianMotion:
    def __init__(self, s0, means, stdev, corr_matrix):
        if np.any(np.linalg.eigvals(corr_matrix) <= 0):
            print('Error: correlation matrix must be positive definite')
        self.corr_matrix = corr_matrix
        self.means = means
        self.stdev = stdev
        self.chol_dec_matrix = linalg.cholesky(corr_matrix)
        self.n_assets = len(means)
        self.s0 = s0
        self.s_time_series = []
        self.s_samples = []
        self.epsilon_samples  = []

    def generate_timeseries(self, maturity=1, n_t=1000, set_seed=42):
        dt = maturity / n_t
        s = self.s0
        s_time_series = []
        for i in range(n_t):
            np.random.seed(set_seed+i)
            epsilon = np.random.standard_normal(size=self.n_assets)
            dw = dt**0.5*self.chol_dec_matrix.dot(epsilon)
            s = s + s*(self.means)*dt + s*dw*self.stdev
            s_time_series.append(list(s))
        self.s_time_series = tt(np.array(s_time_series))

    def generate_samples(self, maturity = 1, n_samples = 1000, set_seed = 0):
      deterministic = (self.s0*np.exp((self.means-self.stdev**2/2)*maturity))[..., None]*np.ones(n_samples,)
      np.random.seed(set_seed)
      self.epsilon_samples = np.random.standard_normal(size=(self.n_assets, n_samples))
      dw = maturity**0.5*self.chol_dec_matrix.dot(self.epsilon_samples)
      stochastic = np.exp((self.stdev[..., None]*np.ones(n_samples,))*dw)
      self.s_samples =  deterministic*stochastic

#Data class tp transform a time series path into features and labels 
#used for supervised learning. Also performs train and test split
class DataClass:
    def __init__(self, data, n_features = [], x = [], y = []):
        self.data = data
        self.n_features = n_features
        self.prediction_lag = []
        self.x = x
        self.x_cv = []
        self.x_train = []
        self.x_test = []
        self.y = y
        self.y_cv = []
        self.y_train = []
        self.y_test = []
        if len(np.shape(self.data))>1:
            self.dimension = len(self.data) 
        else:
            self.dimension = 1

    def time_series_ml_transform(self, prediction_lag):
        self.prediction_lag = prediction_lag
        if self.dimension == 1:
            self.n_samples = len(self.data)-prediction_lag-self.n_features +1 
            for j in range(np.shape(self.data)[0] - self.n_features - self.prediction_lag+1):
                self.x.append(list(self.data[j:j + self.n_features]))
                self.y.append(self.data[j + self.n_features + self.prediction_lag - 1])
            self.x = np.array(self.x)
            self.y = np.array(self.y)
        else:
            self.n_samples = np.shape(self.data)[1]-prediction_lag-self.n_features +1 
            for j in range(np.shape(self.data)[1] - self.n_features - self.prediction_lag+1):
                self.x.append(list(tt(self.data[:, j:j + self.n_features].reshape(-1, 1))[0]))
                self.y.append(list(tt(self.data[:, j + self.n_features + self.prediction_lag - 1].reshape(-1, 1))[0]))
            self.x = np.array(self.x)
            self.y = np.array(self.y)

    def train_cv_test_split(self, train_split = 0.7, cv_split = 0.9, shuffle = False, set_seed = 0):
        self.train_split = train_split
        self.cv_split = cv_split
        if shuffle == False:
            n_train = int(self.x.shape[0] * train_split / 5) * 5
            n_cv = int(self.x.shape[0] * cv_split / 5) * 5
            self.n_cv = int(self.x.shape[0] * cv_split / 5) * 5
            self.x_train = self.x[0:n_train]
            self.y_train = self.y[0:n_train]
            self.x_cv = self.x[n_train:n_cv]
            self.y_cv = self.y[n_train:n_cv]
            self.x_test = self.x[n_cv:]
            self.y_test = self.y[n_cv:]
        else:
            np.random.seed(set_seed)
            rv = np.random.uniform(size=(self.n_samples,))
            self.x_train = self.x[np.where(rv<train_split)]
            self.y_train = self.y[np.where(rv<train_split)]
            self.x_cv = self.x[np.where((rv>train_split)*(rv < cv_split))]
            self.y_cv = self.y[np.where((rv>train_split)*(rv < cv_split))]
            self.x_test = self.x[np.where(rv>cv_split)]
            self.y_test = self.y[np.where(rv>cv_split)]
          
#Class that contains different models used 
class ModelClass:
    def __init__(self, dataclass):
        self.n_features = dataclass.n_features
        self.prediction_lag = dataclass.prediction_lag
        self.dimension = dataclass.dimension
        self.x = dataclass.x
        self.x_cv = dataclass.x_cv
        self.x_train = dataclass.x_train
        self.x_test = dataclass.x_test
        self.y = dataclass.y
        self.y_cv = dataclass.y_cv
        self.y_train = dataclass.y_train
        self.y_test = dataclass.y_test
        self.x_train_lstm = np.reshape(self.x_train, (self.x_train.shape[0], 1, self.x_train.shape[1]))
        self.x_test_lstm = np.reshape(self.x_test, (self.x_test.shape[0], 1, self.x_test.shape[1]))
        self.x_cv_lstm = np.reshape(self.x_cv, (self.x_cv.shape[0], 1, self.x_cv.shape[1]))

    def RidgeCV(self):
        params_Ridge = {'alpha': [1, 0.1, 0.01, 0.001, 0.0001, 0], "fit_intercept": [True, False], "solver": ['svd', 'cholesky', 'lsqr']}
        err = np.infty
        model_best = linear_model.Ridge()
        for alpha_val in params_Ridge['alpha']:
            for ic_val in  params_Ridge['fit_intercept']:
                for solv_val in params_Ridge['solver']:
                    model = linear_model.Ridge(alpha=alpha_val, fit_intercept=ic_val, solver=solv_val)
                    model.fit(self.x_train, self.y_train)
                    y_hat = model.predict(self.x_cv)
                    err_new = np.mean(np.mean((y_hat-self.y_cv)**2))
                    if err_new < err:
                        model_best = model
                        err = err_new
                        print('Ridge CV loss: ', err)
        self.model_ridge = model_best
        return self.model_ridge

    def DeepLearningCV(self, layers = [100], activation_fun = [None], loss_fun = 'mse', n_epochs = 200, use_lstm = False):
        self.model_DL = Sequential()
        for i, val in enumerate(layers):
            if i==0:
                if use_lstm:
                    self.model_DL.add(LSTM(val, input_shape=(1, int(max(np.shape(self.x_train_lstm[1])))), activation = activation_fun[i]))
                else: 
                    self.model_DL.add(Dense(val, input_shape=(1, np.shape(self.x_train)[1]), activation = activation_fun[i]))
            else:
                self.model_DL.add(Dense(val,activation = activation_fun[i]))
        self.model_DL.add(Dense(self.dimension))
        self.model_DL.compile(loss=loss_fun, optimizer='adam')
        if use_lstm==False:
            self.model_DL.fit(self.x_train, self.y_train, validation_data=(self.x_cv, self.y_cv), epochs = n_epochs, verbose = 2)
        else:
            self.model_DL.fit(self.x_train_lstm, self.y_train, epochs = n_epochs, validation_data=(self.x_cv_lstm, self.y_cv), verbose = 2)
        return self.model_DL

#Function to evaluate model performances           
def model_performance(y_hat, y_test, pred_plot = False):
    mrae = np.mean(np.mean(np.abs(y_hat/y_test-1)))
    mse = np.mean(np.mean((y_hat-y_test)**2))
    corr_mae = np.mean(np.mean(np.abs(np.corrcoef(tt(y_hat)) - np.corrcoef(tt(y_test)))))
    print('ma relative  error: ', mrae, ' mse: ',mse,  '  corr_mae:', corr_mae)
    line_colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    if pred_plot == True:
        plt.figure(figsize=(20, 10))
        for i, y_hat_val in enumerate(tt(y_hat)):
            plt.plot(y_hat_val,'--', c = line_colors[i])
            plt.plot(tt(y_test)[i], c = line_colors[i])
    return mrae, mse, corr_mae

