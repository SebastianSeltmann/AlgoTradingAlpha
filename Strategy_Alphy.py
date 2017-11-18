
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import time 
import pandas_datareader.data as web
import openpyxl
import quandl
from scipy.optimize import minimize



# # Data Storing

def store_data():
    # This function is called manually
    # it needs to be called only once
    F = open("C:\AlgoTradingData\quandl_keyx.txt", "r")
    quandl_key = F.read()
    F.close()
    
    # Retrieve Data from original sources
    # Clean them appropriately
    # Store them in a format ready to be loaded by main process:
    # -  stockprices,
    # -  characteristics,
    # -  factors,
    # -  options,
    pass


# # Data Loading & Approach Evaluation

# load the datasets: 
# -  stockprices,
# -  characteristics,
# -  factors,
# -  options,

# split the timerange into chunks
# call optimize for each training-range
# call evaluate_strategy for each testing range with optimized params
# collect metrics of success
# report overall success of the strategy


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

    bounds=[
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
        options, ):
    
    # define initial, blank portfolio
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
