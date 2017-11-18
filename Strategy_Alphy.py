
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import time 
import pandas_datareader.data as web
import openpyxl
import quandl



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

def optimize(
        dates,
        stockprices,
        characteristics,
        factors,
        options, ):
    # create the parameters of our strategy: coverage, quantile & strike regression
    # call scipy library and let it optimize the values for the parameters, for certain return metrics
    # return these optimized parameters
    return (optimized_coverage, optimized_quantile, optimized_strike_0,
            optimized_strike_1)


# # Backtesting Process

# Redundant, we just call evaluate_strategy directly

# # Strategy Evaluation

def evaluate_strategy(
        dates,
        stockprices,
        characteristics,
        factors,
        options,
        coverage,
        quantile,
        strike_0,
        strike_1, ):
    
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
        dates,
        stockprices,
        characteristics,
        factors,
        options,
        coverage,
        quantile,
        strike_0,
        strike_1,
        time,
        portfolio, ):
    # determine P&L from maturing batch of Puts
    # pick stocks
    # pick strikes
    # pick weights (depending on remaining capital)
    # return updated portfolio

    portfolio_new = portfolio
    return portfolio_new
