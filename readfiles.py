"""
Name: Onkar Jadhav
read files and extract features
"""

import xarray as xr
import numpy as np
import pandas as pd
import calendar
import os


def getfilenames(path, monthstart, monthend, year):
    filesnames = []
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith('avg.nc'):
                filesnames.append(f)
                
    daysinmonth = []
    for k in range(monthstart, monthend):
        print(k)
        daysinmonthvar = calendar.monthrange(year, k)[1]
        daysinmonth.append(daysinmonthvar)
        
    totaldays = sum(daysinmonth)

    filesnames = filesnames[0:totaldays]
    return daysinmonth, filesnames

def readfiles(filesnames, path, var):
    """
    Open files and read them
    
    Input: 
        nc files local climate data
    Output: 
        list of data having the variable of interest defined with var
        
    """
    
    pds_day = []
    for file in filesnames:
        # open files
        ds_local = xr.open_dataset(path + file)
        # variable of interest
        
        ds_sstloc = ds_local[var].isel(s_rho=24)
        ds_sstloc_np = ds_sstloc.to_numpy()
        ds_sstloc_mean_np = ds_sstloc_np[0]
        
        ds_sstloc_mean_np_2d = np.nan_to_num(ds_sstloc_mean_np)
        
        # then flatten it to 1d
        ds_sstloc_mean_np_1d = ds_sstloc_mean_np_2d.ravel()  
        
        # append daily data into a list
        pds_day.append(ds_sstloc_mean_np_1d)
        
    return pds_day