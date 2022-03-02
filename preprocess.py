import os
import itertools
import numpy as np
import pandas as pd
import scipy as sp
from scipy.signal import medfilt
from scipy.stats import kurtosis, skew
from scipy.interpolate import interp1d


InfantTrunkMid = ("00B45AF6", "tMid")
InfantTrunkLeft = ("00B45AFD", "tL")
InfantRightLeg = ("00B45AFC", "lR")
InfantLeftLeg = ("00B45ADA", "lL")
InfantHead = ("00B45ADB", "h")

PATHS = ["Freeplay Sensors/"+d+"/sensors/" for d in os.listdir("Freeplay Sensors") if ".D" not in d]
SENSORS = [InfantTrunkMid, InfantRightLeg, InfantLeftLeg]


def loadData(filepath, sensor, sep="\t"):
    
    '''
    Input:
        filepath: path to the file containing data
        sensor: defines from which sensor to extract data
        sep: column separator in the file
    Output:
        extracted data in a form of pandas dataframe
    '''
    
    file = [f for f in os.listdir(filepath) if sensor[0] in f]
    
    
    ## Count the rows to skip that contain description of the file
    i = 0
    with open(filepath+file[0],"r") as f:
        for line in f:
            if line.startswith("/"):
                i += 1
            else:
                break
    
    return pd.read_csv(filepath+file[0], sep=sep, skiprows=i)


def selectSensorColumns(df, cols=['Acc_X', 'Acc_Y', 'Acc_Z', 'Roll', 'Pitch', 'Yaw']):

    '''
    Select columns from which to extract features.
    
    Input:
        df: DataFrame with sensor data
        cols: names of columns to extract
    '''
    
    return df[cols]


def flattenList(L):

    '''
    Flattens out a list.
    Especially useful with strings and other
    non-numerical objects.
    '''
    
    flat_L = []
    
    for el in L:
        if isinstance(el,list):
            flat_L.extend(flattenList(el))
        else:
            flat_L.append(el)

    return flat_L


def countStatistics(df: pd.DataFrame, sensor_name:str, window:int=240, step:int=60) -> pd.DataFrame:
    
    '''
    Counts statistics for the data
    based on a specified time window and in given time steps.
    It calculates a series of statistics based on extracted windows.
    
    Following Franchak et al. (2021) we will calculate:
    - sum
    - mean
    - median
    - kurtosis
    - skew
    - standard deviation
    - minimum
    - maximum
    - 25th quantile
    - 75th quantile
    
    Input:
        df: DataFrame with sensor data
        sensor_name: Takes the name of given sensor
        window: the size of window to extract data from
        step: the size of time step to move the window
    Output:
        stats: DataFrame with extracted features
    '''
    
    stats = []
    columns = df.columns.to_list()
            
    ## Remove columns filled with NaNs and 0
    for col in columns:
        if (df[col].isna()-1).sum() == 0:
            print(col)
            df = df.drop(col, axis=1)
            columns.remove(col)
        elif (df[col].abs().sum() == 0):
            df = df.drop(col, axis=1)
            columns.remove(col)
    
    new_columns = []
    measures = ["sum","mean","std","median","skew","kurtosis","max","min","Q25","Q75"]

    ## Extract windows and calculate statistics
    ## TO DO use this loop to actually count all the statistics
    #-- take as input the three loaded sensor dataframes
    #-- count the singular statistics as well as the pairs and cross sensor & axis stats
    #-- how do you calculate difference score? (it is calculated within a window)
    #-- how do you calculate correlation? Pearsons r? something else? (as well calculated within a window)
    #-- check out franchacks github, casue there he does the difference score!
    for iWindow in range(0,len(df),step):
        temp_s = []
        
        na_sum = np.nansum(df.iloc[iWindow:iWindow+window], axis=0)
        na_mean = np.nanmean(df.iloc[iWindow:iWindow+window], axis=0)
        na_sd = np.nanstd(df.iloc[iWindow:iWindow+window], axis=0)
        na_med = np.nanmedian(df.iloc[iWindow:iWindow+window], axis=0)
        na_skew = skew(df.iloc[iWindow:iWindow+window], axis=0, nan_policy="omit")
        na_kurtosis = kurtosis(df.iloc[iWindow:iWindow+window], axis=0, nan_policy="omit")
        na_max = np.nanmax(df.iloc[iWindow:iWindow+window], axis=0)
        na_min = np.nanmin(df.iloc[iWindow:iWindow+window], axis=0)
        na_25 = np.nanquantile(df.iloc[iWindow:iWindow+window], .25, axis=0)
        na_75 = np.nanquantile(df.iloc[iWindow:iWindow+window], .75, axis=0)
            
        *temp_l, = [l for l in [na_sum, na_mean, na_sd, na_med, na_skew, na_kurtosis, na_max, na_min, na_25, na_75] if len(l) > 1]
        for el in temp_l:
            temp_s.extend([*el])
            
        stats.append(temp_s)
        
    new_columns.append(["{}_{}_{}".format(y,sensor_name,x) for x in measures for y in columns])
    new_columns = flattenList(new_columns)
    
    return pd.DataFrame(stats,columns=new_columns)


def extractFeatures(subjects: list, sensors: list) -> None:
    
    '''
    Extract features for each subject and save it into .csv
    
    Important:
    This is a draft, column names will eventually be computed from the dataframe
    '''
    
    ## Lists for across-sensor metrics
    mags = []
    sums = []
    diffs = []
    
    for subject in subjects:
        features = []
        raw_data = []
        for sensor in sensors:
            raw_data = loadData(subject, sensor)
            print(sensor[1])
            features.append(countStatistics(selectSensorColumns(raw_data), str(sensor[1])))
            
        cols1 = ["Acc_tMid_sum", "Acc_lR_sum", "Acc_lL_sum", "Acc_X_sum", "Acc_Y_sum", "Acc_Z_sum"]
        cols2 = ["Acc_tMid_mag", "Acc_lR_mag", "Acc_lL_mag", "Acc_X_mag", "Acc_Y_mag", "Acc_Z_mag"]
        cols3 = ["Acc_tMid_lR_diff_X","Acc_tMid_lR_diff_Y","Acc_tMid_lR_diff_Z",
                "Acc_tMid_lL_diff_X","Acc_tMid_lL_diff_Y","Acc_tMid_lL_diff_Z",
                "Acc_lR_lL_diff_X","Acc_lR_lL_diff_Y","Acc_lR_lL_diff_Z",
                "Acc_XY_diff_tMid","Acc_XY_diff_lR","Acc_XY_diff_lL",
                "Acc_XZ_diff_tMid","Acc_XZ_diff_lR","Acc_XZ_diff_lL",
                "Acc_YZ_diff_tMid","Acc_YZ_diff_lR","Acc_YZ_diff_lL"]
        cols_final = cols1+cols2+cols3

        accs = []
        gyrs = []

        for sens_feat in features:
            cols_sum = [x for x in sens_feat.columns if x.split("_")[-1] == "sum"]
            acc_lab = cols_sum[:3]
            gyr_lab = cols_sum[3:]
            accs.append(sens_feat[acc_lab].to_numpy())
            gyrs.append(sens_feat[gyr_lab].to_numpy())

        # Sums for axes and sensors
        axis_acc_sums = np.sum(accs, axis=0).T # sum for given axis across sensors
        sensor_acc_sums = np.sum(accs, axis=2) # sum for given sensor across axes

        # Magnitudes for axes and sensors
        axis_acc_mags = np.sqrt(np.sum(np.square(accs), axis=0)).T
        sensor_acc_mags = np.sqrt(np.sum(np.square(accs), axis=2))

        # Correlation for each axis between sensors
        # we should do np correlate for these windows, to see the corelation between values, same with difference scores
        # musze tutaj jakos liczyc jednak 
        
        # Differences for each axes between sensor (each for all 3 axes)
        axis_acc_diffs = np.subtract(accs[0],accs[1]).T # tMid-lR
        axis_acc_diffs2 = np.subtract(accs[0],accs[2]).T # tMid-lL
        axis_acc_diffs3 = np.subtract(accs[1],accs[2]).T # lR-lL

        # Differences for each sensor between axes
        sensor_acc_diffs = np.subtract(np.abs(accs[0].T[0]),np.abs(accs[0].T[1])) # X-Y - tMid
        sensor_acc_diffs2 = np.subtract(accs[0].T[0],accs[0].T[2]) # X-Z - tMid
        sensor_acc_diffs3 = np.subtract(accs[0].T[1],accs[0].T[2]) # Y-Z - tMid
        sensor_acc_diffs4 = np.subtract(accs[1].T[0],accs[0].T[1]) # X-Y - lR
        sensor_acc_diffs5 = np.subtract(accs[1].T[0],accs[0].T[2]) # X-Z - lR
        sensor_acc_diffs6 = np.subtract(accs[1].T[1],accs[0].T[2]) # Y-Z - lR
        sensor_acc_diffs7 = np.subtract(accs[2].T[0],accs[0].T[1]) # X-Y - lL
        sensor_acc_diffs8 = np.subtract(accs[2].T[0],accs[0].T[2]) # X-Z - lL
        sensor_acc_diffs9 = np.subtract(accs[2].T[1],accs[0].T[2]) # Y-Z - lL
        
        diffs_axis = np.vstack((axis_acc_diffs,axis_acc_diffs2,axis_acc_diffs3))
        diffs_sens = np.vstack((sensor_acc_diffs,sensor_acc_diffs2,sensor_acc_diffs3,sensor_acc_diffs4,sensor_acc_diffs5,sensor_acc_diffs6,sensor_acc_diffs7,sensor_acc_diffs8,sensor_acc_diffs9))
        print(diffs_axis.shape)
        print(diffs_sens.shape)
        C = pd.DataFrame(np.vstack((axis_acc_sums,sensor_acc_sums,axis_acc_mags,sensor_acc_mags,diffs_axis,diffs_sens)).T, columns=cols_final)

        final = pd.concat([C,features[0], features[1], features[2]], axis=1)
        final.to_csv(f"{subject.split('/')[-3]}_features.csv")

        break
    
    return (raw_data, features)


def filterSensorData(data, path):
    
    '''
    This function is going to load the data and return the interpolated data.

    Input: data -> input loaded data
            path -> path for a one sensor file example 

    Output: frequency -> the frequency of the sensor
            dataFiltered -> the filtered data
    '''
    
    ## Start the process
    print("Filtering and Interpolating data \n")
    
    try:
        # Get the first file of the list and load it
        frequencyFile = pd.read_csv(path, sep="\t", skiprows=4) # that highly depends on files structure
        for iRow in range(frequencyFile.shape[0]):
            for iColumn in range(frequencyFile.shape[1]):
                if ("Hz" in frequencyFile.iloc[iRow,iColumn]):
                    try:
                        frequency = float(frequencyFile.iloc[iRow,iColumn].split("Hz")[0].strip())
                    except Exception as e:
                        frequency = input("There waas a problem - please write the sensor frequency: ")
                        
                    print("Frequency value of {} Hz found in the sensor file.".format(frequency))
    except Exception as e:
        print("A problem ocurred: {} \n Inputting 60 Hz as default".format(e.args))
        frequency = 60.0
        
    ## We need to interpolate missing packages so the time series are comparable.
    ## Prepare the data for interpolation
    # I think Im missing some functions + not sure on what files exactly this one is working with
    # TO DO


def interpolateSensorData(inputData, frequency):
    
    '''
    This function is going to interpolate the data from the sensor data to
    remove missing values. Initially a spline inteporlation is done, but in
    the future further expansions can be developed.

    Input: 
        inputData: The original data 
        frequency: frequency of the data acquisition
    Ouput: 
        dataInterpolated: The interpolated and filtereddata.
    '''

    ## TO DO
    # Fix the error in interp1d

    # We remove the columns 1-2 as we will remove them as well later on
    columns = inputData.columns[2:]
    dataInterpolated = []
    
    ## Interpolate the data
    # We remove the 1-2 column because it contains
    # irrelevant data that does not need to be interpolated.
    if ~(inputData.empty):
        for iColumn in range(2,inputData.shape[1]):
            if (inputData.iloc[:,iColumn].isna()-1).sum() == 0:
                dataInterpolated.append([inputData.iloc[:,iColumn]])
            else:
                # Calculate error between the mediann filter model and original
                x = inputData.iloc[:,iColumn]
                # Interpolate them using spline interpolation
                xTime = np.arange(1/frequency,1/frequency*(inputData.iloc[:,0].shape[0]+1),1/frequency)
                # Before interpolating we need to make sure that the last values are
                # not NaN because that can return problems in the interpolation
                if np.isnan(x.iloc[-1]):
                    for iNaN in range(len(x)-1,-1,-1):
                        if np.isnan(x[iNaN]):
                            pass
                        else:
                            position = iNaN
                            x[position+1:-1] = x[position]
                            break
                x[np.isnan(x)] = interp1d(xTime[~np.isnan(x)],x[~np.isnan(x)],"cubic")
                ## ERROR to fix
                # AttributeError: type object 'numpy.float64' has no attribute 'kind'
                X_Interpolated = x
                # Smooth the data a bit under a median filter
                dataInterpolated.append(medfilt(X_Interpolated, 3))
                
    return pd.DataFrame(dataInterpolated, columns=columns)


# D = loadData(PATHS[0],SENSORS[0])
# D.to_csv("data.csv")
extractFeatures(PATHS,SENSORS)

## TO DO
# countStstistic musi byc samo liczeniem statystyk, bo po iwndowsach chodzimy liczac wszystkie staty
# correlation jest w np.correlate, wiec tam okienak porownujemy i jest git
# difference score - moze david podpowie
# generalnie extract features rbobi wsio