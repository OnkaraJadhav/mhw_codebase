"""
Name: trainingtestingdata

Extract training and testing datasets

Requirement:
    numpy

Inputs:
    list of local climate data for a particular variable
    List of interpolated global climate data for a particular variable

Output:
    X_train, y_train, X_test, y_test

"""
import numpy as np

def noised(pds_local, interpolatedlist, days):
    
    # noise
    interpolatedlist_noised = []
    for i in range(0, days):
        noise = pds_local[i] - interpolatedlist[i]
        noise = noise*0.7 + interpolatedlist[i]
        interpolatedlist_noised.append(noise.ravel())
        
    return interpolatedlist_noised


def trainingdata(interpolatedlist_SST, interpolatedlist_Salt, pds_local, pds_local_salt, days, days_testS, days_testE):
    #%% Training data
    """
    Extract training dataset
    new features are also created based on the SST, and salt datasets
    All combined features are stored in the X_train variable
    """
    interpolatedlist_SST_noised = noised(pds_local, interpolatedlist_SST, days)
    interpolatedlist_Salt_noised = noised(pds_local_salt, interpolatedlist_Salt, days)

    interpolatedlist_SST_train = interpolatedlist_SST_noised[:days_testS] + interpolatedlist_SST_noised[days_testE+1 :days]
    interpolatedlist_Salt_train = interpolatedlist_Salt_noised[:days_testS] + interpolatedlist_Salt_noised[days_testE+1 :days]
    pds_local_train = pds_local[:days_testS] + pds_local[days_testE+1 :days]

    X_sst_global = np.concatenate(interpolatedlist_SST_train)
    X_salt_global = np.concatenate(interpolatedlist_Salt_train)

    # Lets create some new features
    X_feature_1 = X_sst_global + X_salt_global
    X_feature_2 = X_sst_global * X_salt_global
    X_feature_3 = np.divide(X_salt_global, X_sst_global, out=np.zeros_like(X_salt_global), where=X_sst_global!=0)
    daysrangeS = np.arange(0,(days_testS))
    daysrangeE = np.arange(days_testE+1,days)
    daysrange = np.r_[daysrangeS, daysrangeE]
    X_feature_4 = np.repeat(daysrange, np.size(interpolatedlist_SST[0]))
    X_train = np.concatenate((X_sst_global.reshape(-1,1), X_salt_global.reshape(-1,1), X_feature_1.reshape(-1,1)
                              , X_feature_2.reshape(-1,1), X_feature_3.reshape(-1,1), X_feature_4.reshape(-1,1)), axis =1)

    y_train1 = np.concatenate(pds_local_train)
    
    return X_train, y_train1


def testingdata(interpolatedlist_SST, interpolatedlist_Salt, pds_local, pds_local_salt, days, days_testS, days_testE):
    
    interpolatedlist_SST_noised = noised(pds_local, interpolatedlist_SST, days)
    interpolatedlist_Salt_noised = noised(pds_local_salt, interpolatedlist_Salt, days)
    
    X_sst_global_test = np.concatenate(interpolatedlist_SST_noised[days_testS:days_testE+1])
    X_salt_global_test = np.concatenate(interpolatedlist_Salt_noised[days_testS:days_testE+1])

    X_feature_1_test = X_sst_global_test + X_salt_global_test
    X_feature_2_test = X_sst_global_test * X_salt_global_test
    X_feature_3_test = np.divide(X_salt_global_test, X_sst_global_test, out=np.zeros_like(X_salt_global_test), where=X_sst_global_test!=0)
    daysrange = np.arange(days_testS,(days_testE+1))
    X_feature_4_test = np.repeat(daysrange, np.size(interpolatedlist_SST[0]))
    X_test = np.concatenate((X_sst_global_test.reshape(-1,1), X_salt_global_test.reshape(-1,1), X_feature_1_test.reshape(-1,1)
                              , X_feature_2_test.reshape(-1,1), X_feature_3_test.reshape(-1,1), X_feature_4_test.reshape(-1,1)), axis =1)

    y_test1 = np.concatenate(pds_local[days_testS:days_testE+1])

    return X_test, y_test1