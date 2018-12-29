# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import time
from multiprocessing import Pool





def parallelize_dataframe(df, func):
    df_split = np.array_split(df, num_partitions)
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

def spiliting_time(df):
    df['year'] = df.click_time.apply(lambda t: t.year)
    df['weekday'] = df.click_time.apply(lambda t: t.weekday())
    df['hour'] = df.click_time.apply(lambda t: t.hour)
    return df



if __name__ == '__main__':
    
    root = r'D:/data/2_FraudDetection/raw_data'
    train_fname ='train.csv'
    test_fname = 'test.csv'
    
    print("reading data")
    
    df_train = pd.read_csv(os.path.join(root,train_fname) ,parse_dates=["click_time"])
    num_partitions = 100 #number of partitions to split dataframe
    num_cores = 6 #number of cores on your machine

    print("Starting Multi processing ")
                                     
    start = time.time()
    df_train = parallelize_dataframe(df_train,spiliting_time)      
    end = time.time()
    print('Time elapse :',end - start) 
    
    outpath = os.path.join(root.replace('raw_data','preprocessed'))
    df_train.to_csv(os.path.join(outpath,'all_train_parsed_date.csv'), index = False)
