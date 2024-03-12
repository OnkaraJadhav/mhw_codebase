#%% Statistical downscaling model that predicts SST of the regional model (ROMS) by relating global and local variables

"""
Name: runscript

Requirement:
    numpy, xarray, interpolationroutine, os, time, readfiles, trainingtestingdata
    nnmodel, postprocessing

Inputs:
    Global climate model data
    Local climate model data
    Variable of interest (what you want to predict? - SST)

Output:
    predicted SST using a feedforward NN
    
Variables:
    Global climate model variables: 
        SST, Salt
    Local climate model variables:
        SST

"""
#%% Import necessary libraries
import numpy as np

import xarray as xr
from interpolationroutine import interpolator, plots, westernAustraliaGlobal, westernAustraliaLocal
import os
import time 
from readfiles import readfiles
from trainingtestingdata import trainingdata, testingdata
from nnmodel import neuralnet
from sklearn import metrics

#%% Western Australia and other details:
depth = 0
T = 0
latmin = -34.3265 
latmax = -22.5763
lonmin = 108.511
lonmax = 116.284
days = 5
var_local = 'temp'
var_global_sst = 'sst'
var_global_salt = 'salt'
#%% SST interpolation
pathplots = 'C:/Users/00113324/Documents/Onkar/SurfaceTemp/Plots/'
interpolatedlist_SST = []

ds = xr.open_dataset('C:/Users/00113324/Documents/Onkar/SurfaceTemp/Access-s2_data/SST/do_sst_2021.nc') # GCM data of year 2021
ds_local = xr.open_dataset('C:/Users/00113324/Documents/Onkar/SurfaceTemp/Data/Jan2021/cwa_20210101_12__avg.nc') # just for the grid

for T in range(0, days):
    print(T)
    interpolationresults_SST = interpolator(ds, ds_local, var_global_sst, var_local, 
                                        T, depth, latmin, latmax, lonmin, lonmax)
    
    interpolationresults_SST = np.nan_to_num(interpolationresults_SST)
    interpolatedlist_SST.append(interpolationresults_SST.ravel())
    # plots(ds, ds_local, var_global_sst, var_local, T, depth, latmin, latmax, lonmin, lonmax, T, pathplots)

print("interpolation of SST is done")
#%% Salt interpolation
interpolatedlist_Salt = []

ds = xr.open_dataset('C:/Users/00113324/Documents/Onkar/SurfaceTemp/Access-s2_data/salt/do_salt_2021.nc')
ds_local = xr.open_dataset('C:/Users/00113324/Documents/Onkar/SurfaceTemp/Data/Jan2021/cwa_20210101_12__avg.nc') # just for the grid

for T in range(0, days):
    print(T)
    interpolationresults_salt = interpolator(ds, ds_local, var_global_salt, var_local, 
                                        T, depth, latmin, latmax, lonmin, lonmax)
    
    interpolationresults_salt = np.nan_to_num(interpolationresults_salt)
    interpolatedlist_Salt.append(interpolationresults_salt.ravel())
    # plots(ds, ds_local, var_global_salt, var_local, T, depth, latmin, latmax, lonmin, lonmax, T, pathplots)

print("interpolation of Salt is done")
#%% Read local climate model data for SST:

path = 'C:/Users/00113324/Documents/Onkar/SurfaceTemp/Data/Jan2021/'

filesnames = os.listdir(path)
filesnames = [f for f in filesnames if (f.startswith("cwa") and f.lower().endswith(".nc"))]
pds_local = readfiles(filesnames, path, var_local)

print("Local SST data has been read")
#%% read local climate model data for salt:

var_local_salt = 'salt'

path = 'C:/Users/00113324/Documents/Onkar/SurfaceTemp/Data/Jan2021/'

filesnames = os.listdir(path)
filesnames = [f for f in filesnames if (f.startswith("cwa") and f.lower().endswith(".nc"))]
pds_local_salt = readfiles(filesnames, path, var_local_salt)

print("Local Salt data has been read")
#%% Lets get some training and testing data

index = 4 # the day you want to predict

X_train, y_train1 = trainingdata(interpolatedlist_SST, interpolatedlist_Salt, pds_local, pds_local_salt, index, days)
X_test, y_test1 = testingdata(interpolatedlist_SST, interpolatedlist_Salt, pds_local, pds_local_salt, index, days)

print("Training and testing data computed")
#%% Some data manipulation (NN does not like those nans)

y_train1[y_train1 == 0] = 'nan'
nan_mask = np.isnan(y_train1)
y_train = y_train1[~nan_mask]
X_train = X_train[~nan_mask]

y_test1[y_test1 == 0] = 'nan'
nan_mask1 = np.isnan(y_test1)
y_test = y_test1[~nan_mask1]
X_test = X_test[~nan_mask1]
#%% Train the neural net
st = time.time()

model, X_test, X_train = neuralnet(X_train, y_train, X_test)

ed = time.time()

print("time to train model is", ed-st)

#%% Predict result and check r2, RMSE

y_pred = model.predict(X_test)

from sklearn.metrics import r2_score
print("Coefficient of determination (R2):",r2_score(y_test, y_pred))
print('RMSE_ang:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
#%% Plots!!!

y_test_array = np.empty(y_test1.shape)
y_test_array[~nan_mask1] = y_test
y_pred_array = np.empty(y_test1.shape)
y_pred_array[~nan_mask1] = y_pred.ravel()

from postprocessing import postprocessing

y_test = y_test_array
y_pred = y_pred_array
postprocessing(y_test, y_pred, ds_local, var_local, T, depth, index, pathplots)
