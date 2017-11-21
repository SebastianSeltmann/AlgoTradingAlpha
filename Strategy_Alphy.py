
## ---------------------- IMPORT PACKAGES ----------------------------
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import datetime as dt
import time 
import pandas_datareader.data as web
import openpyxl
import quandl
import wrds

#import matplotlib.pyplot as plt

## ----------------------- SETTINGS ----------------------------------
# # Constants
paths = {}
paths['quandl_key']    = "C:\\AlgoTradingData\\quandl_key.txt"
paths['stockprices']   = "C:\\AlgoTradingData\\stockprices.h5"
paths['pseudo_store']  = "C:\\AlgoTradingData\\retdata.h5"
paths['sp500list_old'] = "C:\\AlgoTradingData\\Constituents.xlsx"
paths['sp500list']     = "C:\\AlgoTradingData\\SP500_Index_Constitutes.csv"
paths['permno_list'] = "C:\\AlgoTradingData\\PERMNO.h5"
paths['options'] = []
for y in range(1996, 2017):
    paths['options'].append("C:\\AlgoTradingData\\rawopt_" + str(y) + "AllIndices.csv")

number_of_timesplits = 10


## -------------------------------------------------------------------
#                           DATA SOURCING
## -------------------------------------------------------------------
def store_sp500list():

    ## ---------------------- CONNECTION AND TEST ------------------------

    db = wrds.Connection()

    test = db.raw_sql('SELECT date, dji FROM djones.djdaily')

    ## ----------------- SOURCING S&P 500 CONSTITUENTS --------------------

    # source historical S&P 500 constituents
    indices = db.get_table('compm', 'IDX_INDEX')
    const = db.get_table('compm', 'idxcst_his')

    # index the indices that contain S&P 500 in the name
    ind_list = [s for s in indices['conm'] if "S&P 500" in s]
    idx = pd.DataFrame(np.isin(indices['conm'], ind_list))
    indices2 = indices[idx.values]

    # extract the data according to the index
    sp500_const = const[const['gvkeyx'] == indices2.loc[2, 'gvkeyx']]
    sp500_const = sp500_const[['gvkey', 'from', 'thru']]
    idx1 = [x is None for x in sp500_const['thru']]
    idx2 = sp500_const['thru'] > dt.date(1990, 1, 1)
    idx = idx1 + idx2
    sp500_const = sp500_const[idx].reset_index(drop=True)

    # source identifiers and id data for the constituents
    gvkey_const = sp500_const['gvkey']
    id_const = db.get_table('compm', 'NAMES_ADSPRATE')
    sp500_const = pd.merge(sp500_const, id_const, how='inner', on='gvkey')
    idx = np.isin(id_const['gvkey'], sp500_const['gvkey'])

    # source CRSP identifiers
    crsp_id = pd.read_csv(paths['sp500list'])
    permnos = crsp_id['PERMNO'].values
    temp = db.raw_sql(
        "Select comnam, permno from crspa.dse where permno in (" + ", ".join(str(x) for x in permnos) + ")")
    temp = temp.dropna(axis=0, how='any').reset_index(drop=True).drop_duplicates()
    temp = pd.merge(crsp_id, temp, how='left', left_on=['PERMNO'], right_on=['permno'])
    temp = temp.drop(['permno'], 1)

    ## ------------------ SOURCING ACCOUNTING DATA -------------------------


    ## ----------------------------- EXPORT --------------------------------

    writer = pd.ExcelWriter(paths['sp500list_old'])
    sp500_const.to_excel(writer, 'identifiers')
    writer.save()

    store = pd.HDFStore(paths['permno_list'])
    store['PERMNO_IDs'] = temp
    store.close()

SP500_symbols = [ 'MMM', 'ABT',]

# # Data Storing

def store_data():
    # This function is called manually
    # it needs to be called only once
    
    # Retrieve Data from original sources
    # Clean them appropriately
    # Store them in a format ready to be loaded by main process:
    # -  stockprices,
    # -  characteristics,
    # -  factors,
    # -  options,

    F = open(paths['quandl_key'], "r")
    quandl_key = F.read()
    F.close()
    print(quandl_key)

    store_sp500()
    pass


def store_sp500():
    problematic_symbols = []
    stockprices = pd.DataFrame()
    remaining = SP500_symbols

    number_of_retries = 10

    retry = 1
    while len(remaining)>0 and retry <= number_of_retries:
        problematic_symbols = []
        for i in range(len(remaining)):
            #'\b'*30 +
            print(str(retry) + ':' + str(i)+ '/' + str(len(remaining))+ '|'+remaining[i])
            try:
                stock = web.DataReader(remaining[i], 'yahoo')
                if(len(stockprices.columns) == 0):
                    renamed_cols = {}
                    for col in stock.columns:
                        renamed_cols[col] = col + '_' + remaining[i]
                    stockprices = stock.rename(columns=renamed_cols)
                else:
                    stockprices = stockprices.join(stock, how='outer', rsuffix='_'+remaining[i])
            except:
                problematic_symbols.append(remaining[i])
        print(str(retry) + ':' + str(len(remaining)) + '/' + str(len(remaining)) + '| '
              + str(len(problematic_symbols)) + ' failed')
        remaining = problematic_symbols
        retry = retry + 1
    store = pd.HDFStore(paths['stockprices'])
    store['sp500'] = stockprices
    store.close()



# # Optimization Process
def objective(
        coverage,
        quantile,
        strike_0,
        strike_1,
        dates,
        stockprices,
        characteristics,
        factors,
        options,):
    results = evaluate_strategy(
        coverage,
        quantile,
        strike_0,
        strike_1,
        dates,
        stockprices,
        characteristics,
        factors,
        options,)
    # We are interested in maximizing the first of the evaluation metrics
    # We achieve this by minimizing the negative of the first metric
    return - (results[0])

def optimize(
        dates,
        stockprices,
        characteristics,
        factors,
        options, ):
    # create the parameters of our strategy: coverage, quantile & strike regression
    # call scipy library and let it optimize the values for the parameters, for certain return metrics
    # return these optimized parameters

    bounds = [
        (0.1,1),        # coverage
        (0.01,1),       # quantile
        (None,None),    # strike_0
        (None,None),    # strike_1
    ]

    initial_guesses = [0.9, 0.9, 0, 1]
    (
        optimized_coverage,
        optimized_quantile,
        optimized_strike_0,
        optimized_strike_1,
    ) = minimize(
        objective,
        initial_guesses,
        args=(dates,stockprices,characteristics,factors,options),
        bounds=bounds
    )
    return (optimized_coverage, optimized_quantile, optimized_strike_0, optimized_strike_1)


# # Strategy Evaluation

def evaluate_strategy(
        coverage,
        quantile,
        strike_0,
        strike_1,
        dates,
        stockprices,
        characteristics,
        factors,
        options,
        cash = 10**6, ):
    
    # define initial portfolio with cash only
    # loop through all assigned dates
    # call timestep at each period
    # determine overall success after all time
    # return report of success
    
    metric_of_success1 = 1
    metric_of_success2 = 1
    return (metric_of_success1, metric_of_success2)


# # TimeStep

def timestep(
        time,
        portfolio,
        coverage,
        quantile,
        strike_0,
        strike_1,
        dates,
        stockprices,
        characteristics,
        factors,
        options, ):
    # determine P&L from maturing batch of Puts
    # pick stocks
    # pick strikes
    # pick weights (depending on remaining capital)
    # return updated portfolio

    portfolio_new = portfolio
    return portfolio_new

def load_data():
    stockprices = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))
    characteristics = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))
    factors = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))

    sp500_members = "todo"
    options = pd.DataFrame()
    for file in paths['options'][0:1]:
        data = options.append(pd.read_csv(file))[['id', 'date', 'strike_price', 'best_bid']]
        options = data

    return stockprices, characteristics,factors, options

# # Main: Data Loading & Approach Evaluation

# load the datasets:
# -  stockprices,
# -  characteristics,
# -  factors,
# -  options,

# Rolling window backtest approach
# split the timerange into chunks
# call optimize for each training-range
# call evaluate_strategy for each testing range with optimized params
# collect metrics of success
# report overall success of the strategy

store = pd.HDFStore(paths['pseudo_store'])
ret = store['/stoxx/ret']
dates = store['/stoxx/ret'].index

stockprices, characteristics, factors, options = load_data()

periods = TimeSeriesSplit(n_splits=number_of_timesplits).split(dates)
metrics = []
for train, test in periods:
    (
        optimized_coverage,
        optimized_quantile,
        optimized_strike_0,
        optimized_strike_1,
    ) = optimize(
        dates,
        stockprices,
        characteristics,
        factors,
        options )

    (metric_of_success1, metric_of_success2) = evaluate_strategy(
        optimized_coverage,
        optimized_quantile,
        optimized_strike_0,
        optimized_strike_1,
        test,
        stockprices,
        characteristics,
        factors,
        options
    )

    metrics.append((metric_of_success1, metric_of_success2))

print(metrics)
