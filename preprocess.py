import os
import typing
import itertools
from matplotlib.pyplot import axes
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


def equalSplit(l: list, n: int) -> list:
    return itertools.zip_longest(*[iter(l)]*n)


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


def countCrossStatistics(raw_data: list, sensor_names: str, window:int=240, step:int=60) -> pd.DataFrame:

    '''
    This function counts the cross-sensor and cross-axis statistics for a group of sensors for a given subject
    '''

    ## TO DO
    #-- Check if features actually contain dataframes
    #-- Do something not to create combs2
    #-- Rewrite the nasty part at the end with counting diffs and corrs

    ## Create useful variables
    df_len = len(raw_data[0].values)
    raw_values = np.array([np.array(x.values[:,0:3],dtype="float") for x in raw_data]) #-- Take just the X,Y,Z columns
    combs = [x for x in itertools.combinations([0,1,2],2)] #-- Create combinations for counting correlations and differences
    axes_names = ["X", "Y", "Z"]
    sensor_names = [x[1] for x in sensor_names]
    sumnames = [*[f"{x}_sum" for x in sensor_names],*[f"{x}_sum" for x in axes_names]]
    magnames = [*[f"{x}_mag" for x in sensor_names],*[f"{x}_mag" for x in axes_names]]
    diffnames = [*[f"{sensor_names[int(x[0])]}{sensor_names[int(x[1])]}_diff" for x in combs],*[f"{axes_names[int(x[0])]}{axes_names[int(x[1])]}_diff" for x in combs]]
    corrnames = [*[f"{sensor_names[int(x[0])]}{sensor_names[int(x[1])]}_corr" for x in combs],*[f"{axes_names[int(x[0])]}{axes_names[int(x[1])]}_corr" for x in combs]]
    new_columns = [*sumnames, *magnames, *diffnames, *corrnames]
    
    ## Lists for counted statistics
    na_sum_axis: list = []
    na_mag_axis: list = []
    na_sum_sensors: list = []
    na_mag_sensors: list = []
    na_diff_axis: list = []
    na_corr_axis: list = []
    na_diff_sensors: list = []
    na_corr_sensors: list = []

    ## Iterate over windows on raw sensor data
    for iWindow in range(0,df_len,step):
        temp_s: list = []
        df_window = raw_values[:,iWindow:iWindow+window,:] #-- Grab the window from raw data for calculations
        na_sum_sensors.append(np.sum(np.nansum(df_window,axis=1),axis=1)) #-- Sums for each sensor
        na_sum_axis.append(np.sum(np.nansum(df_window,axis=0),axis=0)) #-- Sums for X,Y,Z
        na_mag_sensors.append(np.sqrt(np.sum(np.nansum(np.square(df_window), axis=1),axis=1))) #-- Magnitude for each sensor
        na_mag_axis.append(np.sqrt(np.sum(np.nansum(np.square(df_window), axis=0),axis=0))) #-- Magnitude for X,Y,Z
        
        ## Iterate over combinations to make pairwise corrs and diffs
        for c in combs:
            c1, c2 = int(c[0]), int(c[1])

            df_window_1 = raw_values[:,iWindow:iWindow+window,c1]
            df_window_2 = raw_values[:,iWindow:iWindow+window,c2]
            na_diff_sensors.append(np.mean(np.nanmean(np.subtract(df_window_1, df_window_2), axis=1),axis=0)) #-- Difference for each sensor per axis pair
            temp_na_corr_sensors: list = []
            for i in range(len(df_window_1)):
                temp_na_corr_sensors.append(np.corrcoef(df_window_1[i], df_window_2[i])[0,-1]) #-- Coeffs for each sensor per axis pair
            na_corr_sensors.append(np.nanmean(temp_na_corr_sensors,axis=0)) #-- Complete coeffs for each sensor per axis pair


            df_window_1 = raw_values[c1,iWindow:iWindow+window,:]
            df_window_2 = raw_values[c2,iWindow:iWindow+window,:]
            na_diff_axis.append(np.mean(np.nanmean(np.subtract(df_window_1, df_window_2), axis=0),axis=0)) #-- Difference for X,Y,Z per sensor pair
            temp_na_corr_axis: list = []
            for i in range(len(df_window_1[0])):
                temp_na_corr_axis.append(np.corrcoef(df_window_1[:,i], df_window_2[:,i])[0,-1]) #-- Coeffs for X,Y,Z per sensor pair
            na_corr_axis.append(np.nanmean(temp_na_corr_axis,axis=0)) #-- Complete coeffs for X,Y,Z per sensor pair
            
    ## Separate the long vectors into separate columns for each axis and sensor
    na_diff_axis = [x for x in equalSplit(na_diff_axis, 571)]
    na_corr_axis = [x for x in equalSplit(na_corr_axis, 571)]
    na_diff_sensors = [x for x in equalSplit(na_diff_sensors, 571)]
    na_corr_sensors = [x for x in equalSplit(na_corr_sensors, 571)]
    
    ## Turn np.arrays into list for later merging
    na_sum_sensors = [list(x) for x in na_sum_sensors]
    na_sum_axis = [list(x) for x in na_sum_axis]
    na_mag_sensors = [list(x) for x in na_mag_sensors]
    na_mag_axis = [list(x) for x in na_mag_axis]

    ## Merge all the statistics into one list
    all_stats = [[*na_sum_sensors[i], *na_sum_axis[i], *na_mag_sensors[i], *na_mag_axis[i], 
    na_diff_sensors[0][i],na_diff_sensors[1][i],na_diff_sensors[2][i],
    na_diff_axis[0][i],na_diff_axis[1][i],na_diff_axis[2][i],
    na_corr_sensors[0][i],na_corr_sensors[1][i],na_corr_sensors[2][i],
    na_corr_axis[0][i],na_corr_axis[1][i],na_corr_axis[2][i]] for i in range(len(na_sum_sensors))]
    
    return pd.DataFrame(all_stats,columns=new_columns)


def extractFeatures(subjects: list, sensors: list, stop_iter: int = 1, save_each: bool = True, window: int = 240, step: int = 60) -> dict:
    
    '''
    Extract features for each subject and save it into .csv
    
    Important:
    This is a draft, column names will eventually be computed from the dataframe
    '''
    
    ## Create variables
    all_features: dict = {} #-- Final feature dict
    
    ## Iterate over each subject
    iii: int = 0
    while iii != stop_iter:
        subject = subjects[iii]
        sub_name = subject.split("/")[1]

        ## Start with creating variables for storing calculated features and raw sensor data
        features: list = [] #-- Here we deposit calculated stats for given subject
        raw_data: list = [] #-- Here we store raw sensor data
        for sensor in sensors:
            current_df = selectSensorColumns(loadData(subject, sensor))
            raw_data.append(current_df)
            features.append(countStatistics(current_df, str(sensor[1]))) #-- Count statistics for sensor
        features.append(countCrossStatistics(raw_data, sensors)) # Count cross statistics

        ## Save each subject here if save_each is True
        if save_each:
            pd.concat(features,axis=1).to_csv(f"{str(sub_name)}_features.csv")

        ## Create an entry in dictionary for given subject and check if we want to sto iterating
        all_features[str(subject)] = pd.concat(features,axis=1)
        iii += 1
    
    return all_features


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


extractFeatures(PATHS,SENSORS)