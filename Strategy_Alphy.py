
## ---------------------- IMPORT PACKAGES ----------------------------
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import datetime as dt
#import matplotlib.pyplot as plt
import time 
import pandas_datareader.data as web
#import openpyxl
import quandl
import wrds


## ----------------------- SETTINGS ----------------------------------
# # Constants
paths = {}
paths['quandl_key'] = "C:\\AlgoTradingData\\quandl_key.txt"
paths['stockprices'] = "C:\\AlgoTradingData\\stockprices.h5"
paths['pseudo_store'] = "C:\\AlgoTradingData\\retdata.h5"
paths['sp500list'] = "C:\\AlgoTradingData\\Constituents.xlsx"
paths['sp500_permnos'] = "C:\\AlgoTradingData\\SP500_permnos.csv"
paths['h5 constituents & prices'] = "C:\\AlgoTradingData\\[IDs, constituents, prices].h5"
paths['xlsx constituents & prices'] = "C:\\AlgoTradingData\\[IDs, constituents, prices].xlsx"
paths['raw prices'] = "C:\\AlgoTrading\\Data[raw prices].csv"
paths['options'] = []
for y in range(1996, 2017):
    paths['options'].append("C:\\AlgoTradingData\\rawopt_" + str(y) + "AllIndices.csv")

number_of_timesplits = 10


## -------------------------------------------------------------------
#                           DATA SOURCING
## -------------------------------------------------------------------
def store_sp500list():
    # - stockprices


    ## ---------------------- WRDS CONNECTION  ------------------------

    db = wrds.Connection()


    ## ----------------- SOURCING S&P 500 CONSTITUENTS --------------------

    # source historical S&P 500 constituents
    const = db.get_table('compm', 'idxcst_his')

    # source CRSP identifiers
    crsp_id = pd.read_csv(paths['sp500_permnos'])
    crsp_id = crsp_id[crsp_id['ending'] > "1990-12-31"]
    permnos = crsp_id['PERMNO'].values

    ## ------------------ SOURCING ACCOUNTING DATA -------------------------

    # Test source of accounting data
    # gvkeys_list = gvkeys.values
    # SP500_price = db.raw_sql("Select PRCCD,  from comp.g_secd where GVKEY in (" + ", ".join(str(x) for x in gvkeys_list) + ")")

    # No permission to access through python. Check WRDS online querry


    ## ------------------- SOURCING PRICE DATA -----------------------------

    prices = db.raw_sql("Select date, permno, cusip, PRC, shrout from crspa.dsf where permno in (" + ", ".join(
        str(x) for x in permnos) + ")" + " and date between '1990-01-01' and '2017-11-22'")
    prices_sp50 = prices

    permnos_m = prices_sp50['permno'].unique()

    # Process the price data

    for i in permnos_m:
        if i == 10057:
            x = prices_sp50[prices_sp50['permno'] == i][['date', 'prc']].set_index('date', drop=True)
            x.columns = [i]
            prc_merge = x
        else:
            y = prices_sp50[prices_sp50['permno'] == i][['date', 'prc']].set_index('date', drop=True)
            y.columns = [i]
            prc_merge = pd.merge(prc_merge, y, how='outer', left_index=True, right_index=True)

    ## ----------------------------- EXPORT --------------------------------

    writer1 = pd.ExcelWriter(paths['xlsx constituents & prices'])
    const.to_excel(writer1, 'Compustat_const')
    crsp_id.to_excel(writer1, 'CRSP_const')
    prc_merge.to_excel(writer1, 'Prices')
    writer1.save()

    prices.to_csv(paths['raw prices'], sep='\t', encoding='utf-8')

    store = pd.HDFStore(paths['h5 constituents & prices'])
    store['Compustat_const'] = const
    store['CRSP_const'] = crsp_id
    store['Prices_raw'] = prices
    store['Prices'] = prc_merge
    store.close()


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

def load_data():

    # # Data Storing and calling
    store = pd.HDFStore(paths['constituents & prices'])
    prices = store['Prices']
    prices_raw = store['Prices_raw']
    comp_const = store['Compustat_const']
    CRSP_const = store['CRSP_const']
    store.close()

    return prices, prices_raw, comp_const, CRSP_const

prices, prices_raw, comp_const, CRSP_const = load_data()


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
