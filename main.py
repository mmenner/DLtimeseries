#Importing
from dlts import DataClass
from dlts import ModelClass
from dlts import GeometricBrownianMotion
from dlts import model_performance
from numpy import transpose as tt
import matplotlib.pyplot as plt
import numpy as np


#Step 1: Define Input data, MGBM sampling, and time series modeling

#number of time steps for MGBM time series for 1st approac
n_t = 10000

#number of time series lags used as a feature
n_features = int(n_t * 0.05)

#number of time series lags to predict future values
prediction_lag = int(n_t * 0.05)

#use shuffling within time series approach (True: interpolation, False: extrapolation)
use_shuffle = True

#Number of MGBM samples for 2nd approach
n_samples = 10000

#define multivariate GBM parameters
corr_matrix = np.array([[1.0, 0.6,  0.4],
                        [0.6, 1.0,  0.5],
                        [0.4, 0.5,  1.0]])

means = np.array([0.1, 0.07, 0.08])
stdev = np.array([0.5, 0.3, 0.4])
s0 = np.ones(3,)

#Create a MGBM path and MGBM samples
gbm = GeometricBrownianMotion(s0, means, stdev, corr_matrix)
gbm.generate_samples(n_samples=n_samples)
gbm.generate_timeseries(n_t=n_t)
data_gbm_samples = gbm.s_samples
data_timeseries = gbm.s_time_series

#time series appraoch
dcts = DataClass(data_timeseries, n_features=n_features)
dcts.time_series_ml_transform(prediction_lag)
dcts.train_cv_test_split(train_split=0.6, cv_split=0.9, shuffle=use_shuffle)

#sampling approach
dc = DataClass(data_gbm_samples, x=tt(gbm.epsilon_samples), y=tt(data_gbm_samples))
dc.train_cv_test_split(train_split=0.6, cv_split=0.9)


#Step 2: define and train models

#deep learning parameters
nn_layers = [128]*3
activation = ['relu']*3
epochs = 100
use_lstm_ts = True
use_lstm_sample = True


#time series approach modeling
modeling_ts = ModelClass(dcts)
print('Ridge regression - time series')
print(' ')
model_sklearn_ts = modeling_ts.RidgeCV()
print(' ')
print('Deep Learning - time series')
print(' ')
print('')
model_DL_ts = modeling_ts.DeepLearningCV(layers=nn_layers, activation_fun=activation, n_epochs=epochs, use_lstm=use_lstm_ts)

if use_lstm_ts:
    x_train_ts = modeling_ts.x_train_lstm
    x_test_ts = modeling_ts.x_test_lstm
    x_cv_ts = modeling_ts.x_cv_lstm
else:
    x_train_ts = dcts.x_train
    x_test_ts = dcts.x_test
    x_cv_ts = dcts.x_cv


#sampling approach modeling (learn the sampling of a multivariate GBM without correlations given)
modeling_sample = ModelClass(dc)
print('Ridge regression - sampling')
print(' ')
model_sklearn_sample = modeling_sample.RidgeCV()
print(' ')
print('Deep Learning - sampling')
print(' ')
print(' ')
model_DL_sample = modeling_sample.DeepLearningCV(layers=nn_layers, activation_fun=activation, n_epochs=epochs, use_lstm=use_lstm_sample)

if use_lstm_sample:
    x_train_s = modeling_sample.x_train_lstm
    x_test_s = modeling_sample.x_test_lstm
    x_cv_s = modeling_sample.x_cv_lstm
else:
    x_train_s = dc.x_train
    x_test_s = dc.x_test
    x_cv_s = dc.x_cv


#Step 3: prediction and model evaluation
#Time series
print('Evaluation, Time Series')
print(' ')
print(' ')
y_hat_DL_ts = model_DL_ts.predict(x_test_ts)
y_hat_sklearn_ts = model_sklearn_ts.predict(dcts.x_test)
print('-------------------------------------------------------')
print('Deep Learning Performance:')
print('')
model_performance(y_hat_DL_ts, dcts.y_test, pred_plot=True)
print('')
print('')
print('')
print('-------------------------------------------------------')
print('Sklearn Performance:')
print('')
plt.figure()
model_performance(y_hat_sklearn_ts, dcts.y_test, pred_plot=True)
print('')
print('')
print('Plot above: Deep Learning', '     Plot below: Sklearn', '     solid line is data, dashed line is prediction')
print('-----------------------------------------------------')
print(' ')
print(' ')


#Sampling
y_hat_DL_sample = model_DL_sample.predict(x_test_s)
y_hat_sklearn_sample = model_sklearn_sample.predict(dc.x_test)
print('Evaluation, Sampling')
print(' ')
print(' ')
print('-------------------------------------------------------')
print('Deep Learning Performance:')
print('')
model_performance(y_hat_DL_sample, dc.y_test, pred_plot = False)
plt.figure(figsize=(20, 10))
fig1 = plt.hist(y_hat_DL_sample, bins = 50, density = True, cumulative = False, histtype = 'step', color=['r','g','b'])
fig2 = plt.hist(dc.y_test, bins = 50, density = True, cumulative = False, histtype = 'step',color=['r','g','b'])
print('')
print('')
print('')
print('-------------------------------------------------------')
print('Sklearn Performance:')
print('')
plt.figure(figsize=(20, 10))
model_performance(y_hat_sklearn_sample, dc.y_test, pred_plot = False)
fig1 = plt.hist(y_hat_sklearn_sample, bins = 50, density = True, cumulative = False, histtype = 'step', color=['r','g','b'])
fig2 = plt.hist(dc.y_test, bins = 50, density = True, cumulative = False, histtype = 'step',color=['r','g','b'])
print('')
print('')
print('Plot above: Deep Learning', '     Plot below: Sklearn', '     shows distribution of multivariate samples (observed vs. model, in same color respectively)')
print('-----------------------------------------------------')
