#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ALGO LECTURE: Import / save/ play around
Created on Sun Nov 12 18:56:29 2017

@author: vilkov
"""

#%% DEFINE NAMES/ IMPORT MODULES 
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import time 
import pandas_datareader.data as web
import openpyxl


import quandl

quandl.ApiConfig.api_key = 'xHyhpe-jZNm_ADHMp17C'

# define paths and names 
path_input  = '/Users/vilkov/Dropbox/zzAlgo/zShared2017/zzData/Source/'
path_output = '/Users/vilkov/Dropbox/zzAlgo/zShared2017/zzData/Processed/'

path_input = 'C:/Dropbox/Dropbox/Studium/Master/Algorithmic Trading/Data/Source/'
path_output = 'C:/Dropbox/Dropbox/Studium/Master/Algorithmic Trading/Data/Processed/'


fn_retdata_csv = path_input + 'retdata'
fn_retdata     = path_input + 'retdata.h5'



#%% IMPORT DATA FROM WEB/ SAVE CSV/ EXCEL/ H5/ IMPORT FROM CSV
tickers_collection = {
'index':  ['SPY'],
'bonds':  ['TLT','TMV','TMF'],
'voletf': ['SVXY','VXX','XIV'],
'sectorSPDR': ['XLB','XLV','XLP','XLY','XLE','XLF','XLI','XLK','XLU'],
'dj30comp': ['CVX', 'XOM', 'IBM', 'GE', 'MSFT', 'MMM', 'WMT', 'BA', 'VZ', 'UNH',
        'AAPL', 'TRV', 'MCD', 'KO', 'JPM', 'HD', 'NKE', 'DIS', 'JNJ',
        'PFE', 'UTX', 'CSCO', 'DD', 'MRK', 'PG', 'AXP', 'CAT', 'INTC', 'GS']        
        }


#%% read data from HDF file 
store = pd.HDFStore(fn_retdata)
store.keys()
#store['stoxx/ret'] = ret
#store['stoxx/prc'] = prc
#store['stoxx/vol'] = vol
#
#store['indicators/cboe'] = cboe
#
#store['vixfut/vx1'] = vx1
#store['vixfut/vx2'] = vx2
#store['vixfut/vx3'] = vx3


# example how to read
#a = pd.read_hdf(fn_retdata,'stoxx/ret')

# alternative method

ret  = store['/stoxx/ret']
vx1  = store['/vixfut/vx1']
vx2  = store['/vixfut/vx2']
vx3  = store['/vixfut/vx3']
cboe = store['/indicators/cboe']

store.close()


# example how to read
#a = pd.read_hdf(fn_retdata,'stoxx/ret')

#%% SLICE/ SELECT/ PROCESS/ PLOT

set1 = ['SPY','SVXY']

ret1 = ret.loc['2016':'2017',set1]
ret2 = ret.loc['2017',set1]

ret2.columns = [z + '_prc' for z in ret2.columns]

ret3 = pd.concat([ret1,ret2],axis=1,join='outer')

ret4 = ret3.loc[ret3.SPY>0,:]



# group by / rolling

ret.replace(np.nan, 1.0, inplace=True)

x = ret.rolling(window=251).std()

x.plot()
print('done')
#%% GO TO A PRIMITIVE ALGO -- NEXT TIME? 














