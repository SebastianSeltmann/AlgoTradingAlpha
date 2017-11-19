
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import time 
import pandas_datareader.data as web
import openpyxl
import quandl

# # Constants
paths = {}
paths['quandl_key'] = "C:\\AlgoTradingData\\quandl_key.txt"
paths['pseudo_store'] = "C:\\AlgoTradingData\\retdata.h5"

number_of_timesplits = 10

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
    pass

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


# # Backtesting Process

# Redundant, we just call evaluate_strategy directly

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

stockprices     = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))
characteristics = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))
factors         = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))
options         = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))

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