
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
import wrds


# # Constants
paths = {}
paths['quandl_key'] = "C:\\AlgoTradingData\\quandl_key.txt"
paths['stockprices'] = "C:\\AlgoTradingData\\stockprices.h5"
paths['pseudo_store'] = "C:\\AlgoTradingData\\retdata.h5"

number_of_timesplits = 10


def store_sp500list():
    ## -------------------------------------------------------------------
    #                           DATA SOURCING
    ## -------------------------------------------------------------------


    ## ---------------------- IMPORT PACKAGES ----------------------------


    ## ----------------------- SETTINGS ----------------------------------


    EstablishConnection = 1  # 1 = Establishing connection. Input WRDS username and pass
    # 0 = No action

    ## ---------------------- CONNECTION AND TEST ------------------------

    if EstablishConnection == 1:
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
    cusips = sp500_const['cusip'].values
    crsp_id = db.raw_sql("selectp permno, permco, cusip from crspa.dsf where cusip in (" + cusips + ")")
    crsp_id = db.get_table('crspa', 'dsf', columns=['permno', 'cusip'], obs=1000)

    ## ------------------ SOURCING ACCOUNTING DATA -------------------------


    ## ----------------------------- EXPORT --------------------------------

    writer = pd.ExcelWriter('C:/Users/Ion Tapordei/Dropbox/FS/Modules/AT & DA in Python/Algo/Constituents.xlsx')
    sp500_const.to_excel(writer, 'identifiers')
    writer.save()

SP500_symbols = [
    'MMM', 'ABT', 'ABBV', 'ACN', 'ATVI', 'AYI', 'ADBE', 'AMD', 'AAP', 'AES',
    'AET', 'AMG', 'AFL', 'A', 'APD', 'AKAM', 'ALK', 'ALB', 'ARE', 'ALXN',
    'ALGN', 'ALLE', 'AGN', 'ADS', 'LNT', 'ALL', 'GOOGL', 'GOOG', 'MO', 'AMZN',
    'AEE', 'AAL', 'AEP', 'AXP', 'AIG', 'AMT', 'AWK', 'AMP', 'ABC', 'AME',
    'AMGN', 'APH', 'APC', 'ADI', 'ANDV', 'ANSS', 'ANTM', 'AON', 'AOS', 'APA',
    'AIV', 'AAPL', 'AMAT', 'ADM', 'ARNC', 'AJG', 'AIZ', 'T', 'ADSK', 'ADP',
    'AZO', 'AVB', 'AVY', 'BHGE', 'BLL', 'BAC', 'BK', 'BCR', 'BAX', 'BBT',
    'BDX', 'BRK-B', 'BBY', 'BIIB', 'BLK', 'HRB', 'BA', 'BWA', 'BXP', 'BSX',
    'BHF', 'BMY', 'AVGO', 'BF-B', 'CHRW', 'CA', 'COG', 'CDNS', 'CPB', 'COF',
    'CAH', 'CBOE', 'KMX', 'CCL', 'CAT', 'CBG', 'CBS', 'CELG', 'CNC', 'CNP',
    'CTL', 'CERN', 'CF', 'SCHW', 'CHTR', 'CHK', 'CVX', 'CMG', 'CB', 'CHD',
    'CI', 'XEC', 'CINF', 'CTAS', 'CSCO', 'C', 'CFG', 'CTXS', 'CLX', 'CME',
    'CMS', 'KO', 'CTSH', 'CL', 'CMCSA', 'CMA', 'CAG', 'CXO', 'COP', 'ED',
    'STZ', 'COO', 'GLW', 'COST', 'COTY', 'CCI', 'CSRA', 'CSX', 'CMI', 'CVS',
    'DHI', 'DHR', 'DRI', 'DVA', 'DE', 'DLPH', 'DAL', 'XRAY', 'DVN', 'DLR',
    'DFS', 'DISCA', 'DISCK', 'DISH', 'DG', 'DLTR', 'D', 'DOV', 'DWDP', 'DPS',
    'DTE', 'DRE', 'DUK', 'DXC', 'ETFC', 'EMN', 'ETN', 'EBAY', 'ECL', 'EIX',
    'EW', 'EA', 'EMR', 'ETR', 'EVHC', 'EOG', 'EQT', 'EFX', 'EQIX', 'EQR',
    'ESS', 'EL', 'ES', 'RE', 'EXC', 'EXPE', 'EXPD', 'ESRX', 'EXR', 'XOM',
    'FFIV', 'FB', 'FAST', 'FRT', 'FDX', 'FIS', 'FITB', 'FE', 'FISV', 'FLIR',
    'FLS', 'FLR', 'FMC', 'FL', 'F', 'FTV', 'FBHS', 'BEN', 'FCX', 'GPS', 'GRMN',
    'IT', 'GD', 'GE', 'GGP', 'GIS', 'GM', 'GPC', 'GILD', 'GPN', 'GS', 'GT',
    'GWW', 'HAL', 'HBI', 'HOG', 'HRS', 'HIG', 'HAS', 'HCA', 'HCP', 'HP',
    'HSIC', 'HSY', 'HES', 'HPE', 'HLT', 'HOLX', 'HD', 'HON', 'HRL', 'HST',
    'HPQ', 'HUM', 'HBAN', 'IDXX', 'INFO', 'ITW', 'ILMN', 'IR', 'INTC', 'ICE',
    'IBM', 'INCY', 'IP', 'IPG', 'IFF', 'INTU', 'ISRG', 'IVZ', 'IRM', 'JEC',
    'JBHT', 'SJM', 'JNJ', 'JCI', 'JPM', 'JNPR', 'KSU', 'K', 'KEY', 'KMB',
    'KIM', 'KMI', 'KLAC', 'KSS', 'KHC', 'KR', 'LB', 'LLL', 'LH', 'LRCX', 'LEG',
    'LEN', 'LUK', 'LLY', 'LNC', 'LKQ', 'LMT', 'L', 'LOW', 'LYB', 'MTB', 'MAC',
    'M', 'MRO', 'MPC', 'MAR', 'MMC', 'MLM', 'MAS', 'MA', 'MAT', 'MKC', 'MCD',
    'MCK', 'MDT', 'MRK', 'MET', 'MTD', 'MGM', 'KORS', 'MCHP', 'MU', 'MSFT',
    'MAA', 'MHK', 'TAP', 'MDLZ', 'MON', 'MNST', 'MCO', 'MS', 'MOS', 'MSI',
    'MYL', 'NDAQ', 'NOV', 'NAVI', 'NTAP', 'NFLX', 'NWL', 'NFX', 'NEM', 'NWSA',
    'NWS', 'NEE', 'NLSN', 'NKE', 'NI', 'NBL', 'JWN', 'NSC', 'NTRS', 'NOC',
    'NCLH', 'NRG', 'NUE', 'NVDA', 'ORLY', 'OXY', 'OMC', 'OKE', 'ORCL', 'PCAR',
    'PKG', 'PH', 'PDCO', 'PAYX', 'PYPL', 'PNR', 'PBCT', 'PEP', 'PKI', 'PRGO',
    'PFE', 'PCG', 'PM', 'PSX', 'PNW', 'PXD', 'PNC', 'RL', 'PPG', 'PPL', 'PX',
    'PCLN', 'PFG', 'PG', 'PGR', 'PLD', 'PRU', 'PEG', 'PSA', 'PHM', 'PVH',
    'QRVO', 'PWR', 'QCOM', 'DGX', 'Q', 'RRC', 'RJF', 'RTN', 'O', 'RHT', 'REG',
    'REGN', 'RF', 'RSG', 'RMD', 'RHI', 'ROK', 'COL', 'ROP', 'ROST', 'RCL',
    'CRM', 'SBAC', 'SCG', 'SLB', 'SNI', 'STX', 'SEE', 'SRE', 'SHW', 'SIG',
    'SPG', 'SWKS', 'SLG', 'SNA', 'SO', 'LUV', 'SPGI', 'SWK', 'SBUX', 'STT',
    'SRCL', 'SYK', 'STI', 'SYMC', 'SYF', 'SNPS', 'SYY', 'TROW', 'TPR', 'TGT',
    'TEL', 'FTI', 'TXN', 'TXT', 'TMO', 'TIF', 'TWX', 'TJX', 'TMK', 'TSS',
    'TSCO', 'TDG', 'TRV', 'TRIP', 'FOXA', 'FOX', 'TSN', 'UDR', 'ULTA', 'USB',
    'UA', 'UAA', 'UNP', 'UAL', 'UNH', 'UPS', 'URI', 'UTX', 'UHS', 'UNM', 'VFC',
    'VLO', 'VAR', 'VTR', 'VRSN', 'VRSK', 'VZ', 'VRTX', 'VIAB', 'V', 'VNO',
    'VMC', 'WMT', 'WBA', 'DIS', 'WM', 'WAT', 'WEC', 'WFC', 'HCN', 'WDC', 'WU',
    'WRK', 'WY', 'WHR', 'WMB', 'WLTW', 'WYN', 'WYNN', 'XEL', 'XRX', 'XLNX',
    'XL', 'XYL', 'YUM', 'ZBH', 'ZION', 'ZTS'
]

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
