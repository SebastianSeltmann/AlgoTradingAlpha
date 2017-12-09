## ---------------------- IMPORT PACKAGES ----------------------------
import sys
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import datetime as dt
# import matplotlib.pyplot as plt
import time
import pandas_datareader.data as web
# import openpyxl
import quandl
import wrds
import cProfile
import pstats
import gc
import pickle

## ----------------------- SETTINGS ----------------------------------
# gc.disable()
# gc.enable()

# # Constants
F = open("rootpath.txt", "r")
rootpath = F.read()
F.close()
# rootpath = "C:\\AlgoTradingData\\"
paths = {}
paths['quandl_key'] = rootpath + "quandl_key.txt"
paths['stockprices'] = rootpath + "stockprices.h5"
paths['vix'] = rootpath + "vix.h5"
paths['FCFF'] = rootpath + "FCFF.h5"
paths['pseudo_store'] = rootpath + "retdata.h5"
paths['sp500list'] = rootpath + "Constituents.xlsx"
paths['sp500_permnos'] = rootpath + "SP500_permnos.csv"
paths['h5 constituents & prices'] = rootpath + "Data[IDs, constituents, prices].h5"
paths['xlsx constituents & prices'] = rootpath + "Data[IDs, constituents, prices].xlsx"
paths['raw prices'] = rootpath + "Data[raw prices].csv"
paths['fn_AccountingData_xlsx'] = rootpath + "Accounting_data_raw.xlsx"
paths['Fundamentals.xlsx'] = rootpath + "Fundamentals.xlsx"
paths['Fundamentals.h5'] = rootpath + "Fundamentals.h5"
paths['permno_gvkeys_linkage.xlsx'] = rootpath + "permno_gvkeys_linkage.xlsx"
paths['FCFF.xlsx'] = rootpath + "FCFF.xlsx"
paths['FCFF.h5'] = rootpath + "FCFF.h5"
paths['fn_Prices_xlsx'] = rootpath + "Data[IDs, constituents, prices].xlsx"
paths['Prices.xlsx'] = rootpath + "Prices.xlsx"
paths['Prices.h5'] = rootpath + "Prices.h5"
paths['Linkage.xlsx'] = rootpath + "Linkage.xlsx"
paths['Linkage.h5'] = rootpath + "Linkage.h5"
paths['all_options_csv'] = rootpath + "all_options.csv"
paths['all_options_h5'] = rootpath + "all_options.h5"
paths['all_options2_h5'] = rootpath + "all_options2.h5"
paths['all_options3_h5'] = rootpath + "all_options3.h5"
paths['preprocessed_options'] = rootpath + "preprocessed_options.h5"
paths['options_nested_df'] = rootpath + "options_nested_df.h5"
paths['profiler'] = rootpath + "profile_data"
paths['results'] = rootpath + "results.pkl"

paths['options_pickl_path'] = {}
paths['options'] = []
for y in range(1996, 2017):
    paths['options'].append(rootpath + "OptionsData\\rawopt_" + str(y) + "AllIndices.csv")
    paths['options_pickl_path'][y] = rootpath + "OptionsData\\options_" + str(y) + ".pkl"


## -------------------------------------------------------------------
#                           DATA SOURCING
## -------------------------------------------------------------------
def store_sp500():
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
    print('Loading Price Data')
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

    print('Price Data Loaded')
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
    return prc_merge, crsp_id


def store_fundamentals():
    print('reading accounting data excel')
    df_AccountingData = pd.read_excel(paths['fn_AccountingData_xlsx'])

    # %%
    # fundamentalNames = pd.DataFrame(df_AccountingData.columns.values)

    # %% Create new Date called as Final Date

    df_AccountingData["New Date"] = 0
    df_AccountingData["Final Date"] = 0
    print('labelling fiscal quarters uniformally Part 1/3')

    row = 0
    for i in df_AccountingData["Fiscal Quarter"]:
        if i == 1:
            df_AccountingData.iloc[row, 21] = "0331"
        elif i == 2:
            df_AccountingData.iloc[row, 21] = "0630"
        elif i == 3:
            df_AccountingData.iloc[row, 21] = "0930"
        elif i == 4:
            df_AccountingData.iloc[row, 21] = "1231"
        row = row + 1

    print('labelling fiscal quarters uniformally Part 2/3')
    row = 0
    for d in df_AccountingData["Fiscal Year"]:
        df_AccountingData.iloc[row, 22] = str(df_AccountingData.iloc[row, 2].round()) + str(
            df_AccountingData.iloc[row, 21])
        row = row + 1

    print('labelling fiscal quarters uniformally Part 3/3')
    df_AccountingData["Final Date"] = pd.to_datetime(df_AccountingData["Final Date"], format="%Y/%m/%d")

    df_AccountingData["Final Date"] = [dt.datetime.strftime(d, "%Y/%m/%d") for d in df_AccountingData["Final Date"]]

    # %% Processing the Fundamentals Data

    # Creating a dictionary. All characterstics saved as DataFrames in the Dictionary

    fundamentals = {}
    i = 0
    print('Pivoting accounting data')
    for col in df_AccountingData.columns[6:]:
        #   x= pd.pivot_table(df_AccountingData,index=["Data Date"],columns=["Global Company Key"],values=[col],fill_value=0)  # Take output file=fundamentals1.xlsx to compare

        try:
            x = pd.pivot_table(df_AccountingData, index=["Final Date"], columns=["Global Company Key"], values=[col],
                               fill_value=0)
            x.columns = x.columns.droplevel()
            fundamentals[col] = x
        except:
            pass

        i = i + 1

    date_index = pd.date_range(start='1989.09.30', end='2018.03.31', freq='D')
    date_index = [dt.datetime.strftime(d, "%Y/%m/%d") for d in date_index]

    s1 = pd.DataFrame(fundamentals["Capital Expenditures"])
    s1 = pd.DataFrame(s1.reindex(date_index, method='ffill'))
    # Define the date range here
    s1 = s1.loc['1990/01/02':'2018/03/31']

    s2 = pd.DataFrame(fundamentals["Long-Term Debt - Total"])
    s2 = pd.DataFrame(s2.reindex(date_index, method='ffill'))
    # Define the date range here
    s2 = s2.loc['1990/01/02':'2018/03/31']

    s3 = pd.DataFrame(fundamentals["Gain/Loss Pretax"])
    s3 = pd.DataFrame(s3.reindex(date_index, method='ffill'))
    # Define the date range here
    s3 = s3.loc['1990/01/02':'2018/03/31']

    s4 = pd.DataFrame(fundamentals["Net Interest Margin"])
    s4 = pd.DataFrame(s4.reindex(date_index, method='ffill'))
    # Define the date range here
    s4 = s4.loc['1990/01/02':'2018/03/31']

    s5 = pd.DataFrame(fundamentals["Net Income (Loss)"])
    s5 = pd.DataFrame(s5.reindex(date_index, method='ffill'))
    # Define the date range here
    s5 = s5.loc['1990/01/02':'2018/03/31']

    s6 = pd.DataFrame(fundamentals["Pretax Income"])
    s6 = pd.DataFrame(s6.reindex(date_index, method='ffill'))
    # Define the date range here
    s6 = s6.loc['1990/01/02':'2018/03/31']

    s7 = pd.DataFrame(fundamentals["Income Taxes - Total"])
    s7 = pd.DataFrame(s7.reindex(date_index, method='ffill'))
    # Define the date range here
    s7 = s7.loc['1990/01/02':'2018/03/31']

    s8 = pd.DataFrame(fundamentals["Interest and Related Expense- Total"])
    s8 = pd.DataFrame(s8.reindex(date_index, method='ffill'))
    # Define the date range here
    s8 = s8.loc['1990/01/02':'2018/03/31']

    s9 = pd.DataFrame(fundamentals["Financing Activities - Net Cash Flow"])
    s9 = pd.DataFrame(s9.reindex(date_index, method='ffill'))
    # Define the date range here
    s9 = s9.loc['1990/01/02':'2018/03/31']

    s10 = pd.DataFrame(fundamentals["Investing Activities - Net Cash Flow"])
    s10 = pd.DataFrame(s10.reindex(date_index, method='ffill'))
    # Define the date range here
    s10 = s10.loc['1990/01/02':'2018/03/31']

    s11 = pd.DataFrame(fundamentals["Operating Activities - Net Cash Flow"])
    s11 = pd.DataFrame(s11.reindex(date_index, method='ffill'))
    # Define the date range here
    s11 = s11.loc['1990/01/02':'2018/03/31']

    s12 = pd.DataFrame(fundamentals["Market Value - Total"])
    s12 = pd.DataFrame(s12.reindex(date_index, method='ffill'))
    # Define the date range here
    s12 = s12.loc['1990/01/02':'2018/03/31']

    # %% Calculation of FCFF
    # FCFF = [CFO + Interest Expense (1- tax rate) - CAPEX]/(Enterprise Value)
    # EV = Market Value + Debt - Cash
    print('Calculating FCFF')

    sum1 = s5 + s7
    tax_rate = s7 / sum1
    EV = s2 + s12
    FCFF = pd.DataFrame((s11 + s8 * (1 - tax_rate) - s1) / EV).fillna(0)
    # Replacing infinite values by nan
    FCFF = FCFF.replace([np.inf, -np.inf], np.nan)

    # %% Linking gvKey and Premno
    df_Linkage = pd.read_excel(paths['permno_gvkeys_linkage.xlsx'])

    df_Linkage1 = df_Linkage[["Global Company Key", "Historical CRSP PERMNO Link to COMPUSTAT Record",
                              "Historical CRSP PERMCO Link to COMPUSTAT Record", "Company Name",
                              "First Effective Date of Link", "Last Effective Date of Link"]]
    df_Linkage2 = df_Linkage1.drop_duplicates(keep='last')

    # %% Making FCFF1 DataFrame with premno as columns

    FCFF1 = FCFF.reindex(columns=df_Linkage2["Global Company Key"])
    for c in FCFF1.columns:
        FCFF1.columns = df_Linkage2["Historical CRSP PERMNO Link to COMPUSTAT Record"]

    FCFF1 = FCFF1.T.groupby(level=0).first().T

    # %% To have common dates values for Price and FCFF1 dataframes

    df_Prices = pd.read_excel(paths['fn_Prices_xlsx'], sheet_name="Prices")
    # file = pd.ExcelFile(paths['fn_Prices_xlsx'])
    # df_Prices = file.parse('Prices')

    df_Prices.set_index('date', inplace=True)
    date_index = df_Prices.index
    date_index = [dt.datetime.strftime(d, "%Y/%m/%d") for d in date_index]
    FCFF1 = FCFF1.reindex(index=date_index)

    # %% To have common columns for Price and FCFF1 dataframes
    x = pd.DataFrame(FCFF1.columns)
    y = pd.DataFrame(df_Prices.columns)

    y.columns = ['Historical CRSP PERMNO Link to COMPUSTAT Record']
    z = x.append(y)
    z1 = pd.DataFrame(z.duplicated(keep='last'))
    z2 = []
    i = 0
    j = 0

    for a in z1.values:
        if a == True:
            z2.append(z.iloc[j, 0])
            i = i + 1
        j = j + 1

    # z2 has Premnos common between Prices and FCFF1 DataFrame
    z2 = pd.DataFrame(z2)
    z2.columns = ['Historical CRSP PERMNO Link to COMPUSTAT Record']

    i = 0
    FCFF2 = pd.DataFrame([], index=FCFF1.index, columns=[])
    df_Prices1 = pd.DataFrame([], index=FCFF1.index, columns=[])

    # Final FCFF is FCFF2 and Final Price df is df_Prices1
    for c in z2.values:
        FCFF2.loc[:, i] = FCFF1.loc[:, c]
        FCFF2.columns.values[i] = c
        df_Prices1.loc[:, i] = df_Prices.loc[:, c]
        df_Prices1.columns.values[i] = c
        i = i + 1

    FCFF2 = FCFF2.applymap(lambda x: np.nan if x == 0 else x)

    FCFF2['pdDates'] = list(x.date() for x in list(pd.to_datetime(FCFF2.index)))
    FCFF2['date'] = list(x.date() for x in list(pd.to_datetime(FCFF2.index)))
    FCFF2 = FCFF2.set_index('date')

    # %% Saving the FCFF data
    # writing to excel

    writer = pd.ExcelWriter(paths['FCFF.xlsx'])
    FCFF2.to_excel(writer, 'FCFF')
    writer.save()

    # writing to hd5 file
    store = pd.HDFStore(paths['FCFF.h5'])
    store['FCFF'] = FCFF2
    store.close()

    # %% Saving the Price data

    # writing to excel
    writer = pd.ExcelWriter(paths['Prices.xlsx'])
    df_Prices1.to_excel(writer, 'Prices')
    writer.save()

    # writing to hd5 file
    store = pd.HDFStore(paths['Prices.h5'])
    store['Prices'] = df_Prices1
    store.close()

    # %% Saving Linkage file to excel and HD5
    writer = pd.ExcelWriter(paths['Linkage.xlsx'])
    df_Linkage2.to_excel(writer, 'Linkage')
    writer.save()

    # writing to HD5 file
    store = pd.HDFStore(paths['Linkage.h5'])
    store['Linkage'] = df_Linkage2
    store.close()

    # Saving the Fundamentals

    # writing to xlsx file
    writer = pd.ExcelWriter(paths['Fundamentals.xlsx'])
    for z in fundamentals.keys():
        fundamentals[z].to_excel(writer, z[:3])
    writer.save()

    # writing to hd5 file
    fNames = list(fundamentals.keys())
    store = pd.HDFStore(paths['Fundamentals.h5'])
    row = 0
    for x in fundamentals:
        store[fNames[row]] = fundamentals[x]
        row = row + 1
    store.close()


'''
def store_cleaned_FCFF(prices, linkage):

    store = pd.HDFStore(paths['FCFF'])
    FCFF = store['FCFF']
    store.close()
    FCFF = FCFF.applymap(lambda x: np.nan if x == 0.0 else x)
    FCFF.columns


    #FCFF.loc[:,'pdDates'] = list(x.date() for x in list(pd.to_datetime(vix.index)))
    #fitted_FCFF = prices.merge(FCFF, how='left', right_on='pdDates', left_index=True)
    fitted_FCFF = prices.merge(FCFF, how='left', right_index=True, left_index=True)
    #fitted_FCFF.drop('pdDates',axis=1, inplace= True)
    fitted_FCFF.drop(prices.columns, axis=1, inplace=True)
    fitted_FCFF.columns
    prices.columns

    store = pd.HDFStore(paths['vix'])
    store['vix'] = fitted_vix
    store.close()
'''


def store_vix(prices):
    '''
    # Earliest VIX available from quandl is 2005
    F = open(paths['quandl_key'], "r")
    quandl_key = F.read()
    F.close()
    quandl.ApiConfig.api_key = quandl_key
    VIX = quandl.get('CHRIS/CBOE_VX1', start_date="1990-12-31", end_date="2017-12-31")
    print(VIX.index.min())
    VIX = quandl.get('CHRIS/CBOE_VX1', start_date="2003-12-31", end_date="2006-12-31")
    print(VIX.index.min())
    '''

    years = range(1990, 2018)
    attempts = 30
    for i in range(len(years) - 1):
        print(str(years[i]))
        while (attempts > 0):
            try:
                if i == 0:
                    vix = web.DataReader('^VIX', 'yahoo', str(years[i]) + '-01-01', str(years[i + 1]) + '-01-01')
                else:
                    vix = vix.append(
                        web.DataReader('^VIX', 'yahoo', str(years[i]) + '-01-01', str(years[i + 1]) + '-01-01'))
                break
            except:
                print('trying again ' + str(attempts - 1) + '...')
                attempts = attempts - 1
    vix['pdDates'] = list(x.date() for x in list(pd.to_datetime(vix.index)))
    vix.drop_duplicates(inplace=True)
    fitted_vix = prices.merge(vix, how='left', right_on='pdDates', left_index=True)
    fitted_vix['date'] = list(x.date() for x in list(pd.to_datetime(fitted_vix.index)))
    fitted_vix = fitted_vix.set_index('date')
    fitted_vix.drop(prices.columns, axis=1, inplace=True)
    fitted_vix.drop('pdDates', axis=1, inplace=True)
    store = pd.HDFStore(paths['vix'])
    store['vix'] = fitted_vix
    store.close()


def store_options(CRSP_const, prices):
    ## Create constituents data frame
    open(paths['all_options_h5'], 'w').close()  # delete HDF
    CRSP_const = CRSP_const[CRSP_const['ending'] > '1996-01-01']

    ids = CRSP_const

    ## cut the prices data to start in 1996
    dates_p = np.asarray(prices.index)
    prices = prices[dates_p >= pd.datetime.date(pd.datetime(1996, 1, 4))]

    ##
    st_y = pd.to_datetime(CRSP_const['start'])
    en_y = pd.to_datetime(CRSP_const['ending'])

    for file in paths['options']:
        with open(file, 'r') as o:
            print(file)
            data = pd.read_csv(o)
            year_index = file.find('rawopt_')
            cur_y = file[year_index + 7:year_index + 7 + 4]
            idx1 = st_y <= cur_y
            idx2 = en_y >= cur_y
            idx3 = data.best_bid > 0
            idx = idx1 & idx2 & idx3
            const = CRSP_const.loc[idx, :].reset_index(drop=True)
            listO = pd.merge(data[['id', 'date', 'days', 'best_bid', 'impl_volatility', 'delta', 'strike_price']],
                             const[['PERMNO']], how='inner', left_on=['id'], right_on=['PERMNO'])
            listO['date'] = pd.to_datetime(listO['date'], format='%d%b%Y')
            idx3 = listO['delta'] < 0
            listO = listO.loc[idx3, :]
            listO['strike_price'] = listO['strike_price'] / 1000
            print(listO.shape)
            store = pd.HDFStore(paths['all_options_h5'])
            store.append('options' + cur_y, listO, index=False, data_columns=True)
            store.close()


            ##


def preprocess_options_data():
    start_year = 1996
    end_year = 2016
    df = pd.DataFrame()
    for year in range(start_year, end_year):  # range(1996, 2016)
        options_data_year = get_optionsdata_for_year(year)
        df = df.append(options_data_year)

    df_fridays = df[df.date.apply(lambda x: x.weekday()) == 4]
    df_nice_maturities = df_fridays[df_fridays['days'].isin([29, 22, 43, 36])]
    # if len(df_nice_maturities) == 0:
    print("fixing")
    '''
    on fridays the remaining days will always be one of these values: 8, 15, 22, 29, 36, 43, 50, 57
    this was true for all 1996 and 1997, so I assume it is true for all years
    on any given friday, two of these maturities will be available for selling, and these two are 28 days apart
    that makes filtering for just the ones we want to invest in straightforward
    29,22,43,36
    '''
    to_be_replaced = [6, 7, 8, 13, 14, 15, 20, 21, 27, 28, 34, 35, 41, 42, 48, 49, 56]
    replacements = [8, 8, 8, 15, 15, 15, 22, 22, 29, 29, 36, 36, 43, 43, 50, 50, 57]
    df_fridays['days'].replace(to_be_replaced, replacements, inplace=True)
    df_nice_maturities = df_fridays[df_fridays['days'].isin([29, 22, 43, 36])]
    print("fixed")

    store = pd.HDFStore(paths['preprocessed_options'])
    store['preprocessed_options'] = df_nice_maturities
    store.close()

command = 'load_data()'
command = 'store_options_as_nested_df(CRSP_const, prices)'
command = 'laufen()'
def run_profiler(command):
    print('Profiler Running')
    cProfile.run(command, filename=paths['profiler'])
    p = pstats.Stats(paths['profiler'])
    p.sort_stats('cumulative').print_stats(10)
    p.sort_stats('tottime').print_stats(10)


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
    '''
    F = open(paths['quandl_key'], "r")
    quandl_key = F.read()
    F.close()
    '''
    prices, CRSP_const = store_sp500()
    store_fundamentals()
    store_options(CRSP_const)
    store_vix(prices)


def load_data():
    # # Data Storing and calling
    store = pd.HDFStore(paths['h5 constituents & prices'])
    prices = store['Prices']
    prices_raw = store['Prices_raw']
    comp_const = store['Compustat_const']
    CRSP_const = store['CRSP_const']
    store.close()

    store = pd.HDFStore(paths['vix'])
    vix = store['vix']
    store.close()

    store = pd.HDFStore(paths['FCFF.h5'])
    FCFF = store['FCFF']
    store.close()

    store = pd.HDFStore(paths['Linkage.h5'])
    linkage = store['Linkage']
    store.close()

    store = pd.HDFStore(paths['preprocessed_options'])
    options = store['preprocessed_options']
    store.close()
    '''
    # Options data sourcing (test)
    store = pd.HDFStore(paths['all_options_h5'])
    x = store['options1996']
    store.close()
    '''

    return (prices, prices_raw, comp_const, CRSP_const, vix, FCFF, options)


def load_chunked_data(chunksize=251):
    reader_stockprices = pd.read_table(paths['options'][0], sep=',', chunksize=chunksize)
    # reader_options          = pd.read_table(paths['options'][1], sep=',', chunksize=chunksize)
    reader_options = pd.read_hdf(paths['all_options_h5'], 'options', chunksize=chunksize)
    reader_FCFF = pd.read_table(paths['options'][1], sep=',', chunksize=chunksize)
    reader_membership = pd.read_table(paths['options'][1], sep=',', chunksize=chunksize)
    reader_vix = pd.read_hdf(paths['vix'], sep=',', chunksize=chunksize)
    return zip(reader_stockprices, reader_options, reader_FCFF, reader_membership)


def call_options_data():
    store = pd.HDFStore(paths['all_options2_h5'])
    x = store.select(key='options2', where='date == ["02JAN2000"]')

    store.close()

    store = pd.HDFStore(rootpath + "draft_options2.h5")
    x = store.append('options2', x, data_columns=True)
    store['options2'] = x
    store.close()


# # Optimization Process
def objective(
        params,
        stockprices,
        options,
        FCFF,
        VIX,
        membership, ):
    portfolio_sharperatio, portfolio_returns, portfolio_maxdrawdown = evaluate_strategy(
        params[0],  # coverage,
        params[1],  # quantile,
        params[2],  # strike_0,
        params[3],  # strike_1,
        stockprices,
        options,
        FCFF,
        VIX,
        membership, )
    # We are interested in maximizing the first of the evaluation metrics
    # We achieve this by minimizing the negative of the first metric
    return - (portfolio_sharperatio)


def optimize(
        stockprices,
        options,
        FCFF,
        VIX,
        membership, ):
    # create the parameters of our strategy: coverage, quantile & strike regression
    # call scipy library and let it optimize the values for the parameters, for certain return metrics
    # return these optimized parameters

    bounds = [
        (0.1, 1),  # coverage
        (0.01, 1),  # quantile
        (None, None),  # strike_0
        (None, None)  # strike_1
    ]

    initial_guesses = [0.9, 0.9, 1, 0]
    minimization = minimize(
        fun=objective,
        x0=initial_guesses,
        args=(stockprices, options, FCFF, VIX, membership),
        bounds=bounds
    )
    print(minimization)
    (
        optimized_coverage,
        optimized_quantile,
        optimized_strike_0,
        optimized_strike_1,
    ) = minimization.x
    return (optimized_coverage, optimized_quantile, optimized_strike_0, optimized_strike_1)


# # Strategy Evaluation

def max_dd(ser):
    ser = pd.Series(ser)
    zzmax = ser.expanding().max()
    zzmin = ser[-1::-1].expanding().min()[-1::-1]
    mdd = (zzmin / zzmax - 1).min()
    return mdd


def get_optionsdata_for_year(year):
    store = pd.HDFStore(paths['all_options_h5'])
    optionsdata_for_year = store['options' + str(year)]
    store.close()
    return optionsdata_for_year


def get_nested_optionsdata_for_year(year):
    return pd.read_pickle(paths['options_pickl_path'][year])

def run_and_store_results():
    '''
    (portfolio_sharperatio, portfolio_returns, portfolio_volatility, portfolio_metrics, sales) = single_run(year=2009)
    portfolio_metrics.cash.plot()
    metrics.cash.plot()
    '''
    results = pd.DataFrame(columns=['strike_0', 'strike_1', 'SR', 'ret', 'vol', 'metrics', 'sales'])
    strike_1 = 0
    for strike_0 in [0.01, 0.9, 1, 1.2, 1.5, 2.0, 9001.0]:
        print(str(strike_0))
        (SR, ret, vol, metrics, sales) = evaluate_strategy(strike_0=strike_0)
        row = {
            'strike_0': strike_0,
            'strike_1':strike_1,
            'SR':SR,
            'ret':ret,
            'vol':vol,
            'metrics':metrics,
            'sales':sales
        }
        results = results.append(row, ignore_index=True)
        # for year in range(1996, 2016):
        #    # results[strike_0, year] = single_run(year= year, strike_0 = strike_0, strike_1 = strike_1)

    with open(paths['results'], 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(paths['results'], 'rb') as handle:
        loaded_results = pickle.load(handle)

        loaded_results.loc[0,'metrics'].cash


command = "single_run()"
command = "evaluate_strategy( stockprices = current_stockprices, FCFF = current_FCFF, VIX = current_VIX )"


# run_profiler(command)

def evaluate_strategy(
        coverage=1,
        quantile=0.1,  # lower mean we include fewer stocks
        strike_0=0.9,
        strike_1=0.0,
        initial_cash=10 ** 6,
        multiplier=100,
        weekly_risk_free_rate=1.0, ):
    # define initial portfolio with cash only
    # loop through all assigned dates
    # determine overall success after all time
    # return report of success

    '''
    # Run these statements to prepare everything needed from outside this function
    print("loading general data")
    stockprices, prices_raw, comp_const, CRSP_const, VIX, FCFF = load_data()
    year = 2014
    relevant_days_index = stockprices[dt.date(year, 1, 1):dt.date(year + 1, 1, 1)].index
    current_stockprices = stockprices.loc[relevant_days_index]
    current_FCFF        = FCFF.loc[relevant_days_index]
    current_VIX         = VIX.loc[relevant_days_index]
    coverage=1
    initial_cash = 10**6
    quantile=0.1
    strike_0=0.9
    strike_1=0.0
    multiplier = 100
    weekly_risk_free_rate=1.0
    day = dt.date(2005, 2, 4)
    stock = 11850.0
    stockprices = current_stockprices
    FCFF = current_FCFF
    VIX = current_VIX
    '''

    print('Loading data')
    (all_stockprices, all_prices_raw, all_comp_const, all_CRSP_const, all_VIX, all_FCFF, all_options) = load_data()


    start_year = 1996
    end_year = 2016
    print('Preprocessing Data')
    relevant_days_index = all_stockprices[dt.date(start_year, 1, 1):dt.date(end_year, 1, 1)].index
    stockprices = all_stockprices.loc[relevant_days_index]
    FCFF = all_FCFF.loc[relevant_days_index]
    VIX = all_VIX.loc[relevant_days_index]
    df_options = all_options.loc[all_options.date.dt.year == start_year]
    # df = optionsdata_for_year
    # len(df) == 622719


    # keeping only the top q quantile
    FCFF_filtered = FCFF[FCFF > list(FCFF.quantile(1 - quantile, axis=1))]
    stockprices_filtered = stockprices + FCFF_filtered * 0
    target_strike = stockprices_filtered.multiply((strike_0 + strike_1 * VIX)["Adj Close"], axis="index")

    # len(df_nice_maturities) == 65736

    selected_target_strikes = df_options[['date', 'id']].apply(lambda x: target_strike.loc[x[0].date(), x[1]], axis=1)
    df_selected = pd.concat([df_options, selected_target_strikes], axis=1).dropna(axis=0).rename(columns={0: 'target_strike'})
    # len(df_selected) == 31461

    diffs = np.abs(df_selected.strike_price - df_selected.target_strike)
    df_with_diff = pd.concat([df_selected, diffs], axis=1).rename(columns={0: 'abs_difference'})

    min_dict = {}

    def get_min_diff(date, id):
        # optimized this by running it only once for each date+id combination
        if not (date, id) in min_dict:
            min_diff = df_with_diff[(df_with_diff.date == date) & (df_with_diff.id == id)].abs_difference.min()
            min_dict[(date, id)] = min_diff
        else:
            min_diff = min_dict[(date, id)]
        return min_diff

    is_best_fit = df_with_diff[['date', 'id', 'abs_difference']].apply(lambda x: get_min_diff(x[0], x[1]) == x[2],
                                                                       axis=1)

    df_best = df_with_diff[is_best_fit]
    # len(df_best) == 9907

    df_risky = pd.concat([df_best, df_best.delta * df_best.impl_volatility], axis=1).rename(columns={0: 'risk'})

    risk_dict = {}
    def get_weight(risk, date):
        if not (risk, date) in risk_dict:
            risks = df_risky[df_best.date == date].risk
            total_risk = risks.sum()
            total_weight = (total_risk / risks).sum()
            risk_dict[risk, date] = total_risk, total_weight
        else:
            (total_risk, total_weight) = risk_dict[risk, date]
        weight = (total_risk / risk) / total_weight
        return weight

    allocation = df_risky[['risk', 'date', 'id']].apply( lambda x: get_weight(x[0], x[1]) / stockprices.loc[x[1].date(), x[2]] / coverage, axis=1)
    df_allocated = pd.concat([df_risky, allocation], axis=1).rename(columns={0: 'allocation'})
    df_sorted = df_allocated.sort_values(by=['date'])

    # Transaction fees based on Interactive Brokers
    # https://www.interactivebrokers.com/en/index.php?f=commission&p=options1
    def choose_commission(premium):
        if premium >= 0.1:
            return 0.7
        elif premium >= 0.05:
            return 0.5
        else:
            return 0.25

    commissions = df_sorted['best_bid'].apply(lambda x: choose_commission(x)).rename('commission')
    df_final = pd.concat([df_sorted, commissions], axis=1)
    df_final = df_final.assign(amount=np.nan) # adding column for sale amount - they will be inserted later


    portfolio_metrics = pd.DataFrame(index=df_final.date.unique(),
                                     columns=['portfolio_value', 'cash', 'earnings', 'payments', 'fees', 'margin_required'])

    payments_column_index = portfolio_metrics.columns.get_loc('payments')
    cash_column_index = portfolio_metrics.columns.get_loc('cash')
    amount_column_index = df_final.columns.get_loc('amount')
    portfolio_metrics.fillna({'payments': 0, 'earnings': 0, 'fees': 0, 'margin_required': 0}, inplace=True)
    # sale = df_final.iloc[0]
    portfolio_metrics.iloc[0, cash_column_index] = initial_cash
    previous_day = df_final.iloc[0, 1]  # first day
    print('Iterating through evaluation')
    # sale = df_final.iloc[200]
    tmp = np.nan
    # sale = df_final.loc[429770]
    amount_counter = {}
    max_amount = 0
    max_amount_sale = np.nan

    def inc_amount_counter(amount, sale, max_amount, max_amount_sale):
        if amount in amount_counter:
            amount_counter[amount] += 1
        else:
            if amount > max_amount:
                max_amount = amount
                max_amount_sale = sale
            amount_counter[amount] = 1
        return max_amount, max_amount_sale

    try:
        for index, sale in df_final.iterrows():
            tmp = sale
            if previous_day != sale.date:
                portfolio_value, cash, payments, earnings, fees = portfolio_metrics.loc[previous_day][
                    ['portfolio_value', 'cash', 'payments', 'earnings', 'fees']]
                portfolio_metrics.loc[sale.date, 'cash'] = cash * weekly_risk_free_rate - payments + earnings - fees
                previous_day = sale.date

            else:
                portfolio_value, cash, payments, earnings = portfolio_metrics.loc[sale.date][
                    ['portfolio_value', 'cash', 'payments', 'earnings']]

            amount = np.floor(cash * sale.allocation / 4 / multiplier)  # 25% at each week

            df_final.loc[index, 'amount'] = amount
            max_amount, max_amount_sale = inc_amount_counter(amount, sale, max_amount, max_amount_sale)
            portfolio_metrics.loc[sale.date, 'fees'] += min(1, multiplier * sale.commission)
            '''
            https://www.interactivebrokers.com/en/index.php?f=26660&hm=us&ex=us&rgt=1&rsk=0&pm=1&ot=0&rst=1|0|1|0|1|0|1|0|0|1|0|0|1|0|1|0|1|0|0|0|1|0|0|0|1|0|0|0|0|0|0|1|0|0|0|1
            Interactive Brokers Stock Options Margin:
                Put Price
                +
                Maximum (
                    (20% * Underlying Price - Out of the Money Amount),
                    (10% * Strike Price)
                )
            '''
            margin_required = amount * multiplier *  (sale.best_bid + max(0.2*stockprices.loc[sale.date.date(), sale.id] - (sale.best_bid - sale.strike_price), 0.1*sale.strike_price))


            try:  # trying and catching if it fails is faster than checking beforehand, because fails are rate
                expiry = sale.date + dt.timedelta(days=int(sale.days)) + dt.timedelta(
                    days=6)  # adding 6 days so that it ends up on a rebalancing friday
                portfolio_metrics.loc[(portfolio_metrics.index >= sale.date) & (portfolio_metrics.index <= expiry), 'margin_required'] += margin_required

                if not expiry > df_final.date.max():
                    portfolio_metrics.loc[expiry, 'payments'] += multiplier * amount * max(0, sale.strike_price -
                                                                                           stockprices.loc[
                                                                                               expiry.date(), sale.id])
                portfolio_metrics.loc[sale.date, 'earnings'] += multiplier * amount * sale.best_bid
            except:
                # 1997-03-28 was was a good friday holiday:
                # http://www.cboe.com/aboutcboe/cboe-cbsx-amp-cfe-press-releases?DIR=ACNews&FILE=na321.doc&CreateDate=21.03.1997&Title=CBOE%20announces%20Good%20Friday%20trading%20schedule
                # we do not have a stockprice for that day, so we pick the closest previous stockprice
                # the losses will be booked onto the following rebalancing day
                closest_expiry = expiry
                # next_rebalancing_day = expiry + dt.timedelta(days=7)
                next_rebalancing_index = portfolio_metrics.index.get_loc(sale.date) + 1
                for i in range(10):
                    closest_expiry -= dt.timedelta(days=1)
                    if (stockprices.index == closest_expiry.date()).any():
                        if not expiry > df_final.date.max():
                            portfolio_metrics.iloc[
                                next_rebalancing_index, payments_column_index] += multiplier * amount * max(0,
                                                                                                            sale.strike_price -
                                                                                                            stockprices.loc[
                                                                                                                closest_expiry.date(), sale.id])
                        portfolio_metrics.loc[sale.date, 'earnings'] += multiplier * amount * sale.best_bid
                        break

        # not sure which annualization method to use
        returns = portfolio_metrics.cash.pct_change()
        # annual_return = (1+returns.mean())**len(portfolio_metrics)
        annual_return = returns.mean() * len(portfolio_metrics)
        annual_vola = returns.std() * np.sqrt(len(portfolio_metrics))
        sharperatio = annual_return / annual_vola
        # portfolio_maxdrawdown   = portfolio['metrics'].portfolio_value.rolling(window=ddwin).apply(max_dd).min()

    except:
        print("Iterating through evaluation crashed on this sale:")
        print(tmp)
        sharperatio, annual_return, annual_vola = (0, 0, 0)

    returns = portfolio_metrics.cash.pct_change()
    # annual_return = (1+returns.mean())**len(portfolio_metrics)
    annual_return = returns.mean() * len(portfolio_metrics)
    annual_vola = returns.std() * np.sqrt(len(portfolio_metrics))
    sharperatio = annual_return / annual_vola

    return (sharperatio, annual_return, annual_vola, portfolio_metrics, df_final)


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


def main():
    metrics = []
    counter = 0
    for stockprices, options, FCFF, membership in load_chunked_data():
        print(stockprices.shape)

        if counter >= 0:
            (metric_of_success1, metric_of_success2) = evaluate_strategy(
                optimized_coverage,
                optimized_quantile,
                optimized_strike_0,
                optimized_strike_1,
                stockprices,
                options,
                FCFF,
                membership,
            )

            metrics.append((metric_of_success1, metric_of_success2))

        (
            optimized_coverage,
            optimized_quantile,
            optimized_strike_0,
            optimized_strike_1,
        ) = optimize(
            stockprices,
            options,
            FCFF,
            membership, )
        counter = counter + 1

    print(metrics)
