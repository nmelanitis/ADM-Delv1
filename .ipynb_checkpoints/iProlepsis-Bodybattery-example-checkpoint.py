# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 14:40:29 2024

@author: n.melanitis
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tsfresh

from tsfresh.utilities.dataframe_functions import impute
#These are testing utilities, we will integrate the parts later on

#Lets assume we have queried the db, using iProlepsis-swatch-data approach
#And lets start the processing

DATA_PATH = r'C:/Users/n.melanitis/Documents/GitHub/AllDayMetrics/iprTest.csv'

#I know its a FULL body-battery record, it has many patient-ids, timestamps and 
#   Bodybattery values

#readcsv to dataframe
rawd = pd.read_csv(DATA_PATH)

#all the field in the data
field_names = rawd.columns.tolist()

#I will do an example on these>patientid and values
pid = field_names[1]
bbv = field_names[4]
#how many patients
patnames = rawd[pid].unique()
npats = len(patnames)

def mybox(value, xslabel = 'xvalues', tlabel = 'The plot' ):
        q1 = np.percentile(value, 1)
        q3 = np.percentile(value, 99)
        
        # Define bin edges between quartiles
        bin_edges = np.linspace(q1, q3, 100) 
        plt.figure(figsize=(8, 6))
        plt.hist(value, bins=bin_edges, edgecolor='black')
        plt.xlabel(xslabel)
        plt.title(tlabel)
        plt.grid(True)
        plt.show() 
        #Bodybattery: Its almost always every 60seconds, unless something wrong is going on
        #do a boxplot
        plt.figure()
        plt.boxplot(value)
        
def patDataQuality(df, field, pname, valuename, viz=False, fsavefig=False):
    #get values from df
    #from dataframe df get all columns for which df[field] = patientname
    df = df[df[field] == pname] #logical indexing in dataframes
    values = df[valuename]
    #missing values as NaNs
    mvalToken = 'XX'
    df[df[valuename] == mvalToken] = np.nan
    #data quality metrics
    nan_counts = values.isna().sum()
    nan_percentage = nan_counts/ len(df) * 100
    print(f"Subject {pname} has fields {', '.join(map(str, df.columns.tolist()))}")
    print(f"We have {str(nan_counts)} missing values for {valuename} which are {nan_percentage:.2f} percentages of the records")
    
    #remove nan lines from df
    df = df.dropna()
    #fix index
    df = df.reset_index()
    #convert to datetime, check regularity in sampling data
    datetime_obj = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S')
    tintervals = datetime_obj.diff()
    secintervals = datetime_obj.map(pd.datetime.timestamp).diff() #intervals in seconds
    
    # Calculate the mean and standard deviation of the time differences
    mean_time_diff = pd.Timedelta(seconds=sum(secintervals[1:]) / len(secintervals[1:]))
    std_dev_time_diff = pd.Timedelta(seconds=pd.Series(secintervals[1:]).std())
    
    #get the outliers:
    # Calculate quartiles
    q1 = secintervals.quantile(0.25)
    q3 = secintervals.quantile(0.75)
    
    # Calculate the interquartile range
    iqr = q3 - q1
    
    # Define the outlier range
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # Find outliers

    outliers = [(datetime_obj[i-1], datetime_obj[i]) for i, diff in enumerate(secintervals) if diff < lower_bound or diff > upper_bound]

    #outliers indexes
    outIndx = [i for i, diff in enumerate(secintervals) if diff < lower_bound or diff > upper_bound]
    inlInd = values.index.difference(outIndx)

    # Get values at the indices to keep
    inlierv = values[inlInd].astype(int);
    # Print the periods before and after each outlier
    outlier_periods = []
    for i, (before, after) in enumerate(outliers):
            outlier_periods.append((before, after))
            print("Outlier:", ((after-before)))
            print("Time before outlier:", before)
            print("Time after outlier:", after)
            print()
    
    # Store outlier periods as a tuple list
    print("Outlier periods:", outlier_periods)

    
    print("Mean time difference:", mean_time_diff)
    print("Standard deviation of time difference:", std_dev_time_diff)
    
    #get some stats for the values
    values = values.dropna()
    values = values.reset_index()
    
    #figures-Flag viz
    if viz:
        #timeintervals
        mybox(secintervals[1:],xslabel = 'Time Difference (seconds)', tlabel ='Histogram of Time Differences between Consecutive Timestamps')        
        # do vizualizations for values 
        #0. do box plot and histogram
        mybox(values.values.astype(int),xslabel = 'Bodybattery', tlabel ='Histogram of Bodybattery')  
        # repeat but w/o outliers
        mybox(inlierv.values.astype(int),xslabel = 'Bodybattery', tlabel ='Histogram of Bodybattery-inliers') #no diff
        #1. plot a timecourse
        ri = np.random.randint(0, len(values)-2880)
        
        plt.figure()
        plt.plot(values.values.astype(int)[ri:ri+2880])
        plt.title('Bodybattery values')
        if fsavefig:
            plt.savefig('allpointsTimecourse.png')        
        ri = np.random.randint(0, len(values)-2880)
        plt.figure()
        plt.plot(inlierv.values.astype(int)[ri:ri+2880])
        plt.title('Bodybattery inlier values')
        if fsavefig:
            plt.savefig('inliersTimecourse.png')
        
def patfeatures(df, field, pname, valuename, viz=False, fsavefig=False):  
    df = df[df[field] == pname] #logical indexing in dataframes
    df[valuename] = df[valuename].astype(int)
    values = df[valuename]
    extracted_features = tsfresh.extract_features(df, column_id = "patientid", column_sort="timestamp", column_value=valuename) #783 features
    impute(extracted_features)

    