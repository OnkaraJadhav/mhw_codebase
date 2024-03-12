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
from interpolationroutine import interpolator
from interpolationroutine_u import interpolator_u
from interpolationroutine_v import interpolator_v
import time 
from readfiles import readfiles, getfilenames
from trainingtestingdatagenerator import trainingdata, testingdata
from nnmodel import neuralnet
from sklearn import metrics

#%% Western Australia and other details:
depth = 0
T = 0
latmin = -34.3265 
latmax = -22.5763
lonmin = 108.511
lonmax = 116.284
days = 4
var_local = 'temp'
var_global_sst = 'sst'
var_global_salt = 'salt'
var_global_u = 'u'
var_global_v = 'v'
var_global_temp = 'temp'
#%% SST interpolation
interpolatedlist_SST = []
# result_list = []
ds = xr.open_dataset('C:/Users/00113324/Documents/Onkar/SurfaceTemp/Access-s2_data/SST/do_sst_2021.nc') # GCM data of year 2021
ds_local = xr.open_dataset('C:/Users/00113324/Documents/Onkar/SurfaceTemp/Data/Jan2021/cwa_20210101_12__avg.nc') # just for the grid

for T in range(0, days):
    print(T)
    interpolationresults_SST = interpolator(ds, ds_local, var_global_sst, var_local, 
                                        T, depth, latmin, latmax, lonmin, lonmax)
    
    interpolationresults_SST = np.nan_to_num(interpolationresults_SST)
    interpolatedlist_SST.append(interpolationresults_SST.ravel())

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

print("interpolation of Salt is done")

#%% u interpolation
interpolatedlist_u = []

ds = xr.open_dataset('C:/Users/00113324/Documents/Onkar/SurfaceTemp/Access-s2_data/u/do_u_2021.nc')
ds_local = xr.open_dataset('C:/Users/00113324/Documents/Onkar/SurfaceTemp/Data/Jan2021/cwa_20210101_12__avg.nc') # just for the grid

for T in range(0, days):
    print(T)
    interpolationresults_u = interpolator_u(ds, ds_local, var_global_u, var_local, 
                                        T, depth, latmin, latmax, lonmin, lonmax)
    
    interpolationresults_u = np.nan_to_num(interpolationresults_u)
    interpolatedlist_u.append(interpolationresults_u.ravel())

print("interpolation of u is done")

#%% v interpolation
interpolatedlist_v = []

ds = xr.open_dataset('C:/Users/00113324/Documents/Onkar/SurfaceTemp/Access-s2_data/v/do_v_2021.nc')
ds_local = xr.open_dataset('C:/Users/00113324/Documents/Onkar/SurfaceTemp/Data/Jan2021/cwa_20210101_12__avg.nc') # just for the grid

for T in range(0, days):
    print(T)
    interpolationresults_v = interpolator_v(ds, ds_local, var_global_v, var_local, 
                                        T, depth, latmin, latmax, lonmin, lonmax)
    
    interpolationresults_v = np.nan_to_num(interpolationresults_v)
    interpolatedlist_v.append(interpolationresults_v.ravel())

print("interpolation of v is done")

#%% temp interpolation
interpolatedlist_temp = []

ds = xr.open_dataset('C:/Users/00113324/Documents/Onkar/SurfaceTemp/Access-s2_data/ts/do_temp_2021.nc')
ds_local = xr.open_dataset('C:/Users/00113324/Documents/Onkar/SurfaceTemp/Data/Jan2021/cwa_20210101_12__avg.nc') # just for the grid

for T in range(0, days):
    print(T)
    interpolationresults_temp = interpolator(ds, ds_local, var_global_temp, var_local, 
                                        T, depth, latmin, latmax, lonmin, lonmax)
    
    interpolationresults_temp = np.nan_to_num(interpolationresults_temp)
    interpolatedlist_temp.append(interpolationresults_temp.ravel())

print("interpolation of temp is done")

#%% Read local climate model data for SST:

path = 'C:/Users/00113324/Documents/Onkar/SurfaceTemp/Data/2021/'

monthstart = 1
monthend = 2
year = 2021

daysinmonth, filesnames = getfilenames(path, monthstart, monthend, year)
filesnames = [f for f in filesnames if (f.startswith("cwa") and f.lower().endswith(".nc"))]
pds_local = readfiles(filesnames, path, var_local)

print("Local SST data has been read")
#%% read local climate model data for salt:

var_local_salt = 'salt'

pds_local_salt = readfiles(filesnames, path, var_local_salt)

print("Local Salt data has been read")
#%% Lets get some training and testing data

days_testS = 1 # the day you want to predict
days_testE = 2

X_train, y_train1 = trainingdata(interpolatedlist_SST, interpolatedlist_Salt, interpolatedlist_u, interpolatedlist_v, interpolatedlist_temp, pds_local, pds_local_salt, days, days_testS, days_testE)
X_test, y_test1 = testingdata(interpolatedlist_SST, interpolatedlist_Salt, interpolatedlist_u, interpolatedlist_v, interpolatedlist_temp, pds_local, pds_local_salt, days, days_testS, days_testE)

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
