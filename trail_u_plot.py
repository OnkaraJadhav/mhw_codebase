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
from interpolationroutine import interpolator, plots, westernAustraliaGlobal, westernAustraliaLocal, padding
import os
import time 
from readfiles import readfiles
from trainingtestingdatarange import trainingdata, testingdata
from nnmodel import neuralnet
from sklearn import metrics

#%% Western Australia and other details:
depth = 0
T = 0
latmin = -34.3265 
latmax = -22.5763
lonmin = 108.511
lonmax = 116.284
days = 1
var_local = 'temp'
var_global_sst = 'temp'
var_global_salt = 'salt'
#%% SST interpolation
pathplots = 'C:/Users/00113324/Documents/Onkar/SurfaceTemp/Plots/'
interpolatedlist_SST = []
# result_list = []
ds = xr.open_dataset('C:/Users/00113324/Documents/Onkar/SurfaceTemp/Access-s2_data/v/do_v_2021.nc') # GCM data of year 2021
#%%
ds_local = xr.open_dataset('C:/Users/00113324/Documents/Onkar/SurfaceTemp/Data/Jan2021/cwa_20210101_12__avg.nc') # just for the grid

for T in range(0, days):
    print(T)
    interpolationresults_SST = interpolator(ds, ds_local, var_global_sst, var_local, 
                                        T, depth, latmin, latmax, lonmin, lonmax)
    
    interpolationresults_SST = np.nan_to_num(interpolationresults_SST)
    interpolatedlist_SST.append(interpolationresults_SST.ravel())
    plots(ds, ds_local, var_global_sst, var_local, T, depth, latmin, latmax, lonmin, lonmax, T, pathplots)
    # result = MatComp(ds, ds_local, var_global_sst, var_local, 
                      # T, depth, latmin, latmax, lonmin, lonmax)
    # result_list.append(result.ravel())
    # ds_QoI_np = padding()

print("interpolation of SST is done")