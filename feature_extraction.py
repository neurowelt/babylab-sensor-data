import os
import typing
import itertools
import traceback
import warnings
import numpy as np
import pandas as pd
import scipy as sp
from scipy.signal import medfilt
from scipy.stats import kurtosis, skew
from scipy.interpolate import interp1d
from sklearn.decomposition import FastICA


InfantTrunkMid = ("00B45AF6", "tMid") 
InfantTrunkLeft = ("00B45AFD", "tL")
InfantRightLeg = ("00B45AFC", "lR")
InfantLeftLeg = ("00B45ADA", "lL")
InfantHead = ("00B45ADB", "h")

NAMES_DICT = {
    "ankle":"lL",
    "hip":"h",
    "thigh":"tMid",
}

NAMES_TO_FRANCHAK = {
    "tMid":"3",
    "lL": "1",
    "lR": "2"
}

PATHS = ["Freeplay Sensors/"+d+"/sensors/" for d in os.listdir("Freeplay Sensors") if ".D" not in d]
# SENSORS = [InfantTrunkMid, InfantRightLeg, InfantLeftLeg]
SENSORS = [InfantLeftLeg, InfantTrunkMid, InfantHead]
RESPIRATION = [InfantTrunkMid,InfantTrunkLeft]

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


def standardizeData(data, sensor_name):

    '''
    Standardize data using given min-maxing.
    Here we use values from Franchak et al. (2021).
    '''

    ranges = pd.read_csv('ranges_franchak.csv')
    scaled_vals = {}

    for sensor in ranges["sensor"]:
        our_sensor = NAMES_DICT[sensor.split("_")[0]]
        metric = f"{sensor.split('_')[-2][0].upper()}{sensor.split('_')[-2][1:]}_{sensor.split('_')[-1][0].upper()+sensor.split('_')[-1][1:]}"
        if sensor.split("_")[0] in metric.lower():
            metric = metric.split("_")[-1]
        if our_sensor == sensor_name:
            idx = ranges.index[ranges["sensor"] == sensor].to_list()[0]
            fr_max = ranges.at[idx, "max_value"]
            fr_min = ranges.at[idx, "min_value"]
            raws_std = (data[metric] - np.min(data[metric],axis=0)) / (np.max(data[metric],axis=0) - np.min(data[metric],axis=0))
            raws_scaled = raws_std * (fr_max - fr_min) + fr_min
            scaled_vals[metric] = raws_scaled

    return pd.DataFrame.from_dict(scaled_vals)


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


def createDict(keys, vars) -> dict:
    D = {}
    for i in range(len(keys)):
        D[keys[i]] = vars[i]
    return D


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
    measures = ["mean","median","std","skewness","kurtosis","per25","per75","min","max",] #-- sum removed

    ## Extract windows and calculate statistics
    for iWindow in range(0,len(df),step):
        temp_s = []
        
        #na_sum = np.nansum(df.iloc[iWindow:iWindow+window], axis=0)
        na_mean = np.nanmean(df.iloc[iWindow:iWindow+window], axis=0)
        na_sd = np.nanstd(df.iloc[iWindow:iWindow+window], axis=0)
        na_med = np.nanmedian(df.iloc[iWindow:iWindow+window], axis=0)
        na_skew = skew(df.iloc[iWindow:iWindow+window], axis=0, nan_policy="omit")
        na_kurtosis = kurtosis(df.iloc[iWindow:iWindow+window], axis=0, nan_policy="omit")
        na_max = np.nanmax(df.iloc[iWindow:iWindow+window], axis=0)
        na_min = np.nanmin(df.iloc[iWindow:iWindow+window], axis=0)
        na_25 = np.nanquantile(df.iloc[iWindow:iWindow+window], .25, axis=0)
        na_75 = np.nanquantile(df.iloc[iWindow:iWindow+window], .75, axis=0)
            
        *temp_l, = [l for l in [na_mean, na_med, na_sd, na_skew, na_kurtosis, na_25, na_75, na_min, na_max] if len(l) > 1] #- sum removed
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

    ## Create useful variables
    df_len = len(raw_data[0].values)
    raw_a_values = np.array([np.array(x.values[:,0:3],dtype="float32") for x in raw_data]) #-- Take just the X,Y,Z columns
    raw_g_values = np.array([np.array(x.values[:,3:6],dtype="float32") for x in raw_data]) #-- Take just the Roll, Pitch, Yaw columns
    combs = [x for x in itertools.combinations([0,1,2],2)] #-- Create combinations for counting correlations and differences
    axes_names = ["X", "Y", "Z"]
    gyr_names = ["Roll", "Pitch", "Yaw"]
    sensor_names = [x[1] for x in sensor_names]
    window_problem = False

    # Acceleration variables
    ssum_x: list = []
    ssum_y: list = []
    ssum_z: list = []
    diff_x_y: list = []
    diff_x_z: list = []
    diff_y_z: list = []
    corr_x_y: list = []
    corr_x_z: list = []
    corr_y_z: list = []
    acc_dir_vars = [*[f"{x}_sum" for x in axes_names],*[f"diff_{axes_names[x[0]]}_{axes_names[x[1]]}" for x in combs],*[f"corr_{axes_names[x[0]]}_{axes_names[x[1]]}" for x in combs]]

    ssum_tmid: list = []
    ssum_lr: list = []
    ssum_ll: list = []
    mmag_tmid: list = []
    mmag_lr: list = []
    mmag_ll: list = []
    diff_tmid_lr: list = []
    diff_tmid_ll: list = []
    diff_lr_ll: list = []
    corr_tmid_lr: list = []
    corr_tmid_ll: list = []
    corr_lr_ll: list = []
    acc_sens_vars = [*[f"{x}_sum" for x in sensor_names],*[f"{x}_mag" for x in sensor_names],*[f"diff_{sensor_names[x[0]]}_{sensor_names[x[1]]}" for x in combs],*[f"corr_{sensor_names[x[0]]}_{sensor_names[x[1]]}" for x in combs]]

    diffm_tmid_lr: list = []
    diffm_tmid_ll: list = []
    diffm_lr_ll: list = []
    corrm_tmid_lr: list = []
    corrm_tmid_ll: list = []
    corrm_lr_ll: list = []
    acc_mag_vars = [*[f"diffm_{sensor_names[x[0]]}_{sensor_names[x[1]]}" for x in combs],*[f"corrm_{sensor_names[x[0]]}_{sensor_names[x[1]]}" for x in combs]]

    # Gyroscope variables
    #-- The commented are not used in Franchak's published datasets
    # ssumg_x: list = []
    # ssumg_y: list = []
    # ssumg_z: list = []
    # diffg_x_y: list = []
    # diffg_x_z: list = []
    # diffg_y_z: list = []
    # corrg_x_y: list = []
    # corrg_x_z: list = []
    # corrg_y_z: list = []

    ssumg_tmid: list = []
    ssumg_lr: list = []
    ssumg_ll: list = []
    mmagg_tmid: list = []
    mmagg_lr: list = []
    mmagg_ll: list = []
    diffg_tmid_lr: list = []
    diffg_tmid_ll: list = []
    diffg_lr_ll: list = []
    corrg_tmid_lr: list = []
    corrg_tmid_ll: list = []
    corrg_lr_ll: list = []
    gyr_sens_vars = [*[f"{x}_sumg" for x in sensor_names],*[f"{x}_magg" for x in sensor_names],*[f"{sensor_names[x[0]]}_{sensor_names[x[1]]}_diffg" for x in combs],*[f"{sensor_names[x[0]]}_{sensor_names[x[1]]}_corrg" for x in combs]]

    diffmg_tmid_lr: list = []
    diffmg_tmid_ll: list = []
    diffmg_lr_ll: list = []
    corrmg_tmid_lr: list = []
    corrmg_tmid_ll: list = []
    corrmg_lr_ll: list = []
    gyr_mag_vars = [*[f"{sensor_names[x[0]]}_{sensor_names[x[1]]}_diffmg" for x in combs],*[f"{sensor_names[x[0]]}_{sensor_names[x[1]]}_corrmg" for x in combs]]

    new_columns = [*acc_dir_vars, *acc_sens_vars, *acc_mag_vars, *gyr_sens_vars, *gyr_mag_vars]
    
    ## Iterate over windows on raw sensor data
    for iWindow in range(0,df_len,step):

        try:
            df_window = raw_a_values[:,iWindow:iWindow+window,:] #-- Grab the window from acceleration raw data for calculations
            df_window_g = raw_g_values[:,iWindow:iWindow+window,:] #-- Grab the window from gyroscope raw data for calculations
        except IndexError as e:
            window_problem_desc = traceback.format_exc()
            print(window_problem_desc)
            break
        
        # Count stats for acceleration
        sum_tmid = np.nansum(df_window[:,:,0],axis=1) #-- these sums are basically the sum that we then have as the overall statistics - NOT added into Franchaks data at the end, it seems
        sum_lr = np.nansum(df_window[:,:,1],axis=1)
        sum_ll = np.nansum(df_window[:,:,2],axis=1)
        sum_x = np.nansum(df_window[0,:,:],axis=0)
        sum_y = np.nansum(df_window[1,:,:],axis=0)
        sum_z = np.nansum(df_window[2,:,:],axis=0)        
        mag_tmid = np.sqrt(np.nansum(np.square(df_window[:,:,0]),axis=1))
        mag_lr = np.sqrt(np.nansum(np.square(df_window[:,:,1]),axis=1))
        mag_ll = np.sqrt(np.nansum(np.square(df_window[:,:,2]),axis=1))

        ssum_x.append(np.sum(sum_x))
        ssum_y.append(np.sum(sum_y))
        ssum_z.append(np.sum(sum_z))
        diff_x_y.append(np.mean(np.subtract(sum_x, sum_y)))
        diff_x_z.append(np.mean(np.subtract(sum_x, sum_z)))
        diff_y_z.append(np.mean(np.subtract(sum_y, sum_z)))
        corr_x_y.append(np.corrcoef(sum_x, sum_y)[0,-1])
        corr_x_z.append(np.corrcoef(sum_x, sum_z)[0,-1])
        corr_y_z.append(np.corrcoef(sum_y, sum_z)[0,-1])

        ssum_tmid.append(np.sum(sum_tmid))
        ssum_lr.append(np.sum(sum_lr))
        ssum_ll.append(np.sum(sum_ll))
        mmag_tmid.append(np.sum(mag_tmid)) 
        mmag_lr.append(np.sum(mag_lr))
        mmag_ll.append(np.sum(mag_ll))
        diff_tmid_lr.append(np.mean(np.subtract(sum_tmid, sum_lr)))
        diff_tmid_ll.append(np.mean(np.subtract(sum_tmid, sum_ll)))
        diff_lr_ll.append(np.mean(np.subtract(sum_lr, sum_ll)))
        corr_tmid_lr.append(np.corrcoef(sum_tmid, sum_lr)[0,-1])
        corr_tmid_ll.append(np.corrcoef(sum_tmid, sum_ll)[0,-1])
        corr_lr_ll.append(np.corrcoef(sum_lr, sum_ll)[0,-1])

        diffm_tmid_lr.append(np.mean(np.subtract(mag_tmid, mag_lr)))
        diffm_tmid_ll.append(np.mean(np.subtract(mag_tmid, mag_ll)))
        diffm_lr_ll.append(np.mean(np.subtract(mag_lr, mag_ll)))
        corrm_tmid_lr.append(np.corrcoef(mag_tmid, mag_lr)[0,-1])
        corrm_tmid_ll.append(np.corrcoef(mag_tmid, mag_ll)[0,-1])
        corrm_lr_ll.append(np.corrcoef(mag_lr, mag_ll)[0,-1])

        # Repeat for gyroscope data
        sumg_tmid = np.nansum(df_window_g[:,:,0],axis=1)
        sumg_lr = np.nansum(df_window_g[:,:,1],axis=1)
        sumg_ll = np.nansum(df_window_g[:,:,2],axis=1)
        # sumg_x = np.nansum(df_window_g[0,:,:],axis=0)
        # sumg_y = np.nansum(df_window_g[1,:,:],axis=0)
        # sumg_z = np.nansum(df_window_g[2,:,:],axis=0)        
        magg_tmid = np.sqrt(np.nansum(np.square(df_window_g[:,:,0]),axis=1))
        magg_lr = np.sqrt(np.nansum(np.square(df_window_g[:,:,1]),axis=1))
        magg_ll = np.sqrt(np.nansum(np.square(df_window_g[:,:,2]),axis=1))

        # Omitted by Franchak in the published datasets
        # ssumg_x.append(np.sum(sumg_x))
        # ssumg_y.append(np.sum(sumg_y))
        # ssumg_z.append(np.sum(sumg_z))
        # diffg_x_y.append(np.mean(np.subtract(sumg_x, sumg_y)))
        # diffg_x_z.append(np.mean(np.subtract(sumg_x, sumg_z)))
        # diffg_y_z.append(np.mean(np.subtract(sumg_y, sumg_z)))
        # corrg_x_y.append(np.corrcoef(sumg_x, sumg_y)[0,-1])
        # corrg_x_z.append(np.corrcoef(sumg_x, sumg_z)[0,-1])
        # corrg_y_z.append(np.corrcoef(sumg_y, sumg_z)[0,-1])

        ssumg_tmid.append(np.sum(sumg_tmid))
        ssumg_lr.append(np.sum(sumg_lr))
        ssumg_ll.append(np.sum(sumg_ll))
        mmagg_tmid.append(np.sum(magg_tmid))
        mmagg_lr.append(np.sum(magg_lr))
        mmagg_ll.append(np.sum(magg_ll))
        diffg_tmid_lr.append(np.mean(np.subtract(sumg_tmid, sumg_lr)))
        diffg_tmid_ll.append(np.mean(np.subtract(sumg_tmid, sumg_ll)))
        diffg_lr_ll.append(np.mean(np.subtract(sumg_lr, sumg_ll)))
        corrg_tmid_lr.append(np.corrcoef(sumg_tmid, sumg_lr)[0,-1])
        corrg_tmid_ll.append(np.corrcoef(sumg_tmid, sumg_ll)[0,-1])
        corrg_lr_ll.append(np.corrcoef(sumg_lr, sumg_ll)[0,-1])

        diffmg_tmid_lr.append(np.mean(np.subtract(magg_tmid, magg_lr)))
        diffmg_tmid_ll.append(np.mean(np.subtract(magg_tmid, magg_ll)))
        diffmg_lr_ll.append(np.mean(np.subtract(magg_lr, magg_ll)))
        corrmg_tmid_lr.append(np.corrcoef(magg_tmid, magg_lr)[0,-1])
        corrmg_tmid_ll.append(np.corrcoef(magg_tmid, magg_ll)[0,-1])
        corrmg_lr_ll.append(np.corrcoef(magg_lr, magg_ll)[0,-1])

    ## IMPORTANT - this order fits franchaks data, but labels above do not match
    var_list = [
        ssum_x,ssum_y,ssum_z,
        corr_x_y,corr_x_z,corr_y_z,
        diff_x_y,diff_x_z,diff_y_z,
        ssum_tmid,ssum_lr,ssum_ll,
        corr_tmid_lr,corr_tmid_ll,corr_lr_ll,
        diff_tmid_lr,diff_tmid_ll,diff_lr_ll,
        mmag_tmid,mmag_lr,mmag_ll,
        corrm_tmid_lr,corrm_tmid_ll,corrm_lr_ll,
        diffm_tmid_lr,diffm_tmid_ll,diffm_lr_ll,
        ssumg_tmid,ssumg_lr,ssumg_ll,
        corrg_tmid_lr,corrg_tmid_ll,corrg_lr_ll,
        diffg_tmid_lr,diffg_tmid_ll,diffg_lr_ll,
        mmagg_tmid,mmagg_lr,mmagg_ll,
        corrmg_tmid_lr,corrmg_tmid_ll,corrmg_lr_ll,
        diffmg_tmid_lr,diffmg_tmid_ll,diffmg_lr_ll,
    ]

    return pd.DataFrame.from_dict(createDict(new_columns,var_list),orient="columns")


def extractFeatures(subjects: list, sensors: list, stop_iter: int = 1, save_raw: bool = True, save_each: bool = True, window: int = 240, step: int = 60, standardize: bool = True) -> dict:
    
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
        print(f"Current subject: {sub_name}\n")

        ## Start with creating variables for storing calculated features and raw sensor data
        features: list = [] #-- Here we deposit calculated stats for given subject
        raw_data: list = [] #-- Here we store raw sensor data
        for sensor in sensors:
            rawDataFrame = loadData(subject, sensor)
            if save_raw:
                print("Saving raw DataFrame.\n")
                rawDataFrame.to_csv(f"{str(sub_name)}_{str(sensor[1])}_rawData.csv")
            if standardize:
                current_df = standardizeData(interpolateSensorData(selectSensorColumns(rawDataFrame),60,sensor),sensor[1])
            else:
                current_df = interpolateSensorData(selectSensorColumns(rawDataFrame),60,sensor)
            ## standardize data to Franchaks
            raw_data.append(current_df)
            features.append(countStatistics(current_df, str(sensor[1]))) #-- Count statistics for sensor
        features.insert(0,countCrossStatistics(raw_data, sensors)) # Count cross statistics

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
    ### UNFINISHED FUNCTION (but it was not used, as the sensor data was already prepared)

def interpolateSensorData(inputData, frequency, sensor):
    
    '''
    This function is going to interpolate the data from the sensor data to
    remove missing values. Initially a spline inteporlation is done, but in
    the future further expansions can be developed.

    Input: 
        inputData: The original data 
        frequency: frequency of the data acquisition
    Ouput: 
        dataInterpolated: The interpolated and filtereddata.

    Warnings:
        VisibleDeprecationWarning - I don't really know where is this one coming from, but features seem to be all alright
    '''

    # We remove the columns 1-2 as we will remove them as well later on
    columns = inputData.columns
    dataInterpolated = []
    
    ## Interpolate the data
    print("Interpolating data\n")
    if ~(inputData.empty):
        for iColumn in range(inputData.shape[1]):
            if (inputData.iloc[:,iColumn].isna()-1).sum() == 0:
                dataInterpolated.append([inputData.iloc[:,iColumn]])
            else:
                # Calculate error between the mediann filter model and original
                x = inputData.iloc[:,iColumn]

                # Interpolate them using spline interpolation
                xTime = np.arange(1/frequency,1/frequency*(inputData.iloc[:,0].shape[0]+1),1/frequency, dtype="float32")
                x = np.asarray(x,dtype="float32")
                f = interp1d(xTime[~np.isnan(x)],x[~np.isnan(x)],"cubic",fill_value="extrapolate")

                # Interpolate if there are NaNs in x
                if sum(np.isnan(x)) > 0:
                    X_Interpolated = [*x[~np.isnan(x)],f(xTime[np.isnan(x)])]
                else:
                    X_Interpolated = x

                # Smooth the data a bit under a median filter
                dataInterpolated.append(medfilt(X_Interpolated, 3))
    print("Data interpolated succesfully.\n")
    dataInterpolated = np.array(dataInterpolated,dtype="float32").T
    d = pd.DataFrame(dataInterpolated, columns=columns)
    d.to_csv(f'{sensor[1]}_interpolated.csv')
    return pd.DataFrame(dataInterpolated, columns=columns)


def isolate_respiration(subjects: list, sensors: list, stop_iter: int = 1, save_raw: bool = True):
    ## Iterate over each subject
    iii: int = 0
    while iii != stop_iter:
        subject = subjects[iii]
        sub_name = subject.split("/")[1]
        raws = []
        cols = []
        print(f"Current subject: {sub_name}\n")

        for sensor in sensors:
            rawDataFrame = interpolateSensorData(selectSensorColumns(loadData(subject, sensor),cols=['Acc_X', 'Acc_Y', 'Acc_Z', "Gyr_X", "Gyr_Y","Gyr_Z", "Mag_X", "Mag_Y", "Mag_Z"]),60,sensor)
            rawDataFrame.columns = [f"{c}_{sensor[1]}" for c in rawDataFrame.columns]
            raws.append(rawDataFrame)

        raws_conc = pd.concat([*raws],names=[*[*cols]],axis=1)
        if save_raw:
            raws_conc.to_csv(f"tMid_all_{sub_name}.csv")
        raws_conc.astype('float16')
        raws_conc = raws_conc.fillna(0)
        print(raws_conc.head())
        print(np.any(np.isnan(raws_conc)))
        print(np.all(np.isfinite(raws_conc)))

        ICA = FastICA(n_components=3)
        Component = ICA.fit_transform(raws_conc.values)
        comp_d = pd.DataFrame(data=Component, columns=["IC1","IC2","IC3"])
        comp_d.to_csv(f"ics_tMid_{sub_name}.csv")

        iii += 1
    

if __name__ == "__main__":
    # extractFeatures(PATHS,SENSORS,save_raw=False)
    isolate_respiration(PATHS,RESPIRATION)