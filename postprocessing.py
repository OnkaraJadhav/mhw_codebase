"""
Name: postprocessing
plot function

Requirement:
    interpolationroutine, numpy, matplotlib

Inputs:
    predicted results
    true results

Output:
    error, predicted, true plots

"""

from interpolationroutine import westernAustraliaLocal
import numpy as np
import matplotlib.pyplot as plt


def postprocessing(y_test, y_pred, ds_local, var_local, T, depth, index, path):
    
    ypred = y_pred.reshape(640,480)

    Latlocal_1, Lonlocal_1, Lat_np, Lon_np, ds_sstloc_mean_np = westernAustraliaLocal(ds_local, var_local, T, depth)

    idx0 = np.argwhere(np.isnan(ds_sstloc_mean_np))
    idx0 = np.asarray(idx0)

    ypred[idx0[:,0],idx0[:,1]] = 0
    ypred[ypred == 0] = 'nan'

    ds_local['Predicted'] = (('eta_rho', 'xi_rho'), ypred)
    
    ytest = y_test.reshape(640,480)
    ytest[ytest==0] = 'nan'
    ds_local['ROMS'] = (('eta_rho', 'xi_rho'), ytest)
    
    Diff = (ytest - ypred)/ytest
    ds_local['Error'] = (('eta_rho', 'xi_rho'), Diff)
    
    section = ds_local.Predicted
    section.plot(x="lon_rho", y="lat_rho", figsize=(15, 7), clim=(25, 35))
    plt.savefig(str(path) + "predicted_" + str(var_local) + str(index) + ".png", format="png", dpi=300)
    
    section = ds_local.ROMS
    section.plot(x="lon_rho", y="lat_rho", figsize=(15, 7), clim=(25, 35))
    plt.savefig(str(path) + "ROMS_" + str(var_local) + str(index) + ".png", format="png", dpi=300)

    section = ds_local['Predicted'].isel(eta_rho=slice(380, 500) , xi_rho=slice(350, 460))
    section.plot(x="lon_rho", y="lat_rho", figsize=(7, 3), clim=(25, 35), vmin=24, vmax=30)
    plt.savefig(str(path) + "predicted_sharkbay_" + str(var_local) + str(index) + ".png", format="png", dpi=300)
    
    section = ds_local['ROMS'].isel(eta_rho=slice(380, 500) , xi_rho=slice(350, 460))
    section.plot(x="lon_rho", y="lat_rho", figsize=(7, 3), clim=(25, 35), vmin=24, vmax=30)
    plt.savefig(str(path) + "ROMS_sharkbay_" + str(var_local) + str(index) + ".png", format="png", dpi=300)

    section = ds_local.Error
    section.plot(x="lon_rho", y="lat_rho", figsize=(15, 7), clim=(25, 35))
    plt.savefig(str(path) + "Error_" + str(var_local) + str(index) + ".png", format="png", dpi=300)
    
    section = ds_local['Error'].isel(eta_rho=slice(380, 500) , xi_rho=slice(350, 460))
    section.plot(x="lon_rho", y="lat_rho", figsize=(15, 7), clim=(25, 35))
    plt.savefig(str(path) + "Error_sharkbay_" + str(var_local) + str(index) + ".png", format="png", dpi=300)
