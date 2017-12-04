
## ---------------------- IMPORT PACKAGES ----------------------------
import sys
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
import cProfile
import pstats


## ----------------------- SETTINGS ----------------------------------
# # Constants
F = open("rootpath.txt", "r")
rootpath = F.read()
F.close()
paths = {}
paths['quandl_key']                 = rootpath + "quandl_key.txt"
paths['stockprices']                = rootpath + "stockprices.h5"
paths['vix']                        = rootpath + "vix.h5"
paths['FCFF']                       = rootpath + "FCFF.h5"
paths['pseudo_store']               = rootpath + "retdata.h5"
paths['sp500list']                  = rootpath + "Constituents.xlsx"
paths['sp500_permnos']              = rootpath + "SP500_permnos.csv"
paths['h5 constituents & prices']   = rootpath + "Data[IDs, constituents, prices].h5"
paths['xlsx constituents & prices'] = rootpath + "Data[IDs, constituents, prices].xlsx"
paths['raw prices']                 = rootpath + "Data[raw prices].csv"
paths['fn_AccountingData_xlsx']     = rootpath + "Accounting_data_raw.xlsx"
paths['Fundamentals.xlsx']          = rootpath + "Fundamentals.xlsx"
paths['Fundamentals.h5']            = rootpath + "Fundamentals.h5"
paths['permno_gvkeys_linkage.xlsx'] = rootpath + "permno_gvkeys_linkage.xlsx"
paths['FCFF.xlsx']                  = rootpath + "FCFF.xlsx"
paths['FCFF.h5']                    = rootpath + "FCFF.h5"
paths['fn_Prices_xlsx']             = rootpath + "Data[IDs, constituents, prices].xlsx"
paths['Prices.xlsx']                = rootpath + "Prices.xlsx"
paths['Prices.h5']                  = rootpath + "Prices.h5"
paths['Linkage.xlsx']               = rootpath + "Linkage.xlsx"
paths['Linkage.h5']                 = rootpath + "Linkage.h5"
paths['all_options_csv']            = rootpath + "all_options.csv"
paths['all_options_h5']             = rootpath + "all_options.h5"
paths['all_options2_h5']            = rootpath + "all_options2.h5"
paths['all_options3_h5']            = rootpath + "all_options3.h5"
paths['options_nested_df']          = rootpath + "options_nested_df.h5"
paths['profiler']                   = rootpath + "profiler.txt"

paths['options'] = []

for y in range(1996, 2016):
    paths['options'].append(rootpath + "OptionsData\\rawopt_" + str(y) + "AllIndices.csv")


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
    #fundamentalNames = pd.DataFrame(df_AccountingData.columns.values)

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
            x = pd.pivot_table(df_AccountingData, index=["Final Date"], columns=["Global Company Key"], values=[col], fill_value=0)
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
    #file = pd.ExcelFile(paths['fn_Prices_xlsx'])
    #df_Prices = file.parse('Prices')

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

    FCFF2 = FCFF2.applymap (lambda x: np.nan if x == 0 else x)

    # %% Saving the FCFF data
    # writing to excel

    writer = pd.ExcelWriter(paths['FCFF.xlsx'] )
    FCFF2.to_excel(writer, 'FCFF')
    writer.save()

    # writing to hd5 file
    store = pd.HDFStore(paths['FCFF.h5'] )
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

    years = range(1990,2018)
    attempts = 30
    for i in range(len(years)-1):
        print(str(years[i]))
        while(attempts > 0):
            try:
                if i == 0:
                    vix = web.DataReader('^VIX', 'yahoo', str(years[i]) + '-01-01', str(years[i+1]) + '-01-01')
                else:
                    vix = vix.append(web.DataReader('^VIX', 'yahoo', str(years[i]) + '-01-01', str(years[i+1]) + '-01-01'))
                break
            except:
                print('trying again ' + str(attempts-1) + '...')
                attempts = attempts - 1
    vix.loc[:,'pdDates'] = list(x.date() for x in list(pd.to_datetime(vix.index)))
    fitted_vix = prices.merge(vix, how='left', right_on='pdDates', left_index=True)
    fitted_vix.drop('pdDates',axis=1, inplace= True)
    fitted_vix.drop(prices.columns, axis=1, inplace=True)

    store = pd.HDFStore(paths['vix'])
    store['vix'] = fitted_vix
    store.close()

def store_options(CRSP_const, prices):

    ## Create constituents data frame
    CRSP_const = CRSP_const[CRSP_const['ending'] > '1996-01-01']

    ids = CRSP_const

    ## cut the prices data to start in 1996
    dates_p = np.asarray(prices.index)
    prices = prices[dates_p >= pd.datetime.date(pd.datetime(1996, 1, 4))]

    ##
    store = pd.HDFStore(paths['all_options3_h5'])
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
            idx = idx1 & idx2
            const = CRSP_const.loc[idx, :].reset_index(drop=True)
            listO = pd.merge(data[['id', 'date', 'days', 'best_bid', 'impl_volatility', 'strike_price']],
                             const[['PERMNO']], how='inner', left_on=['id'], right_on=['PERMNO'])
            listO['date'] = pd.to_datetime(listO['date'], format='%d%b%Y')
            print(listO.shape)
            store = pd.HDFStore(paths['all_options3_h5'])
            store.append('options' + cur_y, listO, index=False, data_columns=True)
            store.close()
            '''
            if counter == 0:
                with open(paths['all_options_csv'], 'w') as f:
                    listO.to_csv(f, header=True, index=False)
            else:
                with open(paths['all_options_csv'], 'a') as f:
                    listO.to_csv(f, header=False, index=False)
            counter = counter + 1
            '''


    df = pd.DataFrame(columns = ids, index = prices.index)
    for file in paths['options']:
        with open(file, 'r') as o:
            print(file)
            data = pd.read_csv(o)
            year_index = file.find('rawopt_')
            cur_y = file[year_index + 7:year_index + 7 + 4]
            idx1 = st_y <= cur_y
            idx2 = en_y >= cur_y
            idx = idx1 & idx2
            const = CRSP_const.loc[idx, :].reset_index(drop=True)
            listO = pd.merge(data[['id', 'date', 'days', 'best_bid', 'impl_volatility', 'strike_price']],
                             const[['PERMNO']], how='inner', left_on=['id'], right_on=['PERMNO'])
            print(listO.shape)

            temp = listO['date']
            listO['date'] = pd.to_datetime(temp, format = '%d%b%Y')

            dates_index = listO['date'].drop_duplicates()
            listO = listO.set_index('date')
            listO = listO.set_index('id', append=True)
            for date in dates_index:
                for permno in const:
                    index = listO.groupby(listO.index)
                    df.loc[date, permno] = index.get_group((date, permno))

            store.append('options/' + cur_y, df, index=False)

            '''
            if counter == 0:
                with open(paths['all_options_csv'], 'w') as f:
                    listO.to_csv(f, header=True, index=False)
            else:
                with open(paths['all_options_csv'], 'a') as f:
                    listO.to_csv(f, header=False, index=False)
            counter = counter + 1
            '''
    store.close()

    ##

def store_options_as_nested_df(CRSP_const, prices):
    open(paths['options_nested_df'], 'w').close() #deletes previous hdf store
    store = pd.HDFStore(paths['options_nested_df'])
    for file in paths['options']:
        #with open(file, 'r') as file:
        #file = paths['options'][0]
        print('reading: ' + file)
        year_index = file.find('rawopt_')
        year = int(file[year_index + 7:year_index + 7 + 4])

        unprocessed_data = pd.read_csv(file)
        dt.datetime.strptime(unprocessed_data.date[0], '%d%b%Y')


        print('transforming dates into usable format')
        unprocessed_data.loc[:,'formatted_day'] = list(dt.datetime.strptime(x, '%d%b%Y').date() for x in unprocessed_data.date)
        counter = 0
        relevant_days_index = prices[dt.date(year,1,1):dt.date(year+1,1,1)].index

        target = len(relevant_days_index)*len(prices.columns)
        optionsdata = pd.DataFrame(index=relevant_days_index, columns=prices.columns)

        def to_date(string):
            return dt.datetime.strptime(string, '%Y-%m-%d').date()

        for day in relevant_days_index:
            CRSP_const
            for stock in prices.columns:
                str(int(stock))

                if ((CRSP_const.PERMNO == int(stock)) & (CRSP_const.start.apply(to_date) >= day) & (CRSP_const.ending.apply(to_date) <= day)).any():
                    optionsday = unprocessed_data[(unprocessed_data.formatted_day == day) | (unprocessed_data.id == stock)]
                    if len(optionsday.index) == 0:
                        optionsdata.at[day, stock] = np.nan
                    else:
                        formatted_optionsday = pd.DataFrame(
                            columns=['strike', 'daysToExpiry', 'price', 'delta', 'implied_volatiliy']
                        )
                        formatted_optionsday.strike             = optionsday.strike_price
                        formatted_optionsday.daysToExpiry       = optionsday.days
                        formatted_optionsday.price              = optionsday.best_bid
                        formatted_optionsday.delta              = - optionsday.delta # removing the negativity, we don't need it
                        formatted_optionsday.implied_volatiliy  = optionsday.impl_volatility
                        optionsdata.at[day,stock] = formatted_optionsday
                else:
                    optionsdata.at[day, stock] = np.nan
                counter = counter + 1
                print (str(year) + ': ' + str(counter) + '/' + str(target))
        store.append('options', optionsdata, index=False)
    store.close()


command = 'load_data()'
command = 'store_options_as_nested_df(CRSP_const, prices)'
def run_profiler(command):
    #prices, prices_raw, comp_const, CRSP_const, vix, FCFF = load_data()

    cProfile.run(command, filename=paths['profiler'])
    p = pstats.Stats(paths['profiler'])
    p.sort_stats('cumulative').print_stats(10)


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

    # Options data sourcing (test)
    store = pd.HDFStore(paths['all_options3_h5'])
    x = store['options2015']
    store.close()

    return prices, prices_raw, comp_const, CRSP_const, vix, FCFF

def load_chunked_data(chunksize=251):
    reader_stockprices      = pd.read_table(paths['options'][0], sep=',', chunksize=chunksize)
    #reader_options          = pd.read_table(paths['options'][1], sep=',', chunksize=chunksize)
    reader_options          = pd.read_hdf(paths['all_options_h5'], 'options', chunksize=chunksize)
    reader_FCFF             = pd.read_table(paths['options'][1], sep=',', chunksize=chunksize)
    reader_membership       = pd.read_table(paths['options'][1], sep=',', chunksize=chunksize)
    reader_vix              = pd.read_hdf(paths['vix'], sep=',', chunksize=chunksize)
    return zip(reader_stockprices, reader_options, reader_FCFF, reader_membership)

def call_options_data():

    store = pd.HDFStore(paths['all_options2_h5'])
    x     = store.select(key = 'options2', where = 'date == ["02JAN2000"]')

    store.close()

    store = pd.HDFStore(rootpath + "draft_options2.h5")
    x = store.append('options2', x, data_columns = True)
    store['options2'] = x
    store.close()

# # Optimization Process
def objective(
        params,
        stockprices,
        options,
        FCFF,
        VIX,
        membership,):
    portfolio_sharperatio, portfolio_returns, portfolio_maxdrawdown = evaluate_strategy(
        params[0], #coverage,
        params[1], #quantile,
        params[2], #strike_0,
        params[3], #strike_1,
        stockprices,
        options,
        FCFF,
        VIX,
        membership,)
    # We are interested in maximizing the first of the evaluation metrics
    # We achieve this by minimizing the negative of the first metric
    return - (portfolio_sharperatio)

def optimize(
            stockprices,
            options,
            FCFF,
            VIX,
            membership,):
    # create the parameters of our strategy: coverage, quantile & strike regression
    # call scipy library and let it optimize the values for the parameters, for certain return metrics
    # return these optimized parameters

    bounds = [
        (0.1,1),        # coverage
        (0.01,1),       # quantile
        (None,None),    # strike_0
        (None,None)     # strike_1
    ]

    initial_guesses = [0.9, 0.9, 1, 0]
    minimization = minimize(
        fun = objective,
        x0 = initial_guesses,
        args = (stockprices, options, FCFF, VIX, membership),
        bounds = bounds
    )
    print(minimization)
    (
        optimized_coverage,
        optimized_quantile,
        optimized_strike_0,
        optimized_strike_1,
    ) = minimization.x
    return (optimized_coverage, optimized_quantile, optimized_strike_0, optimized_strike_1)

'''
# Execute these lines to test the functions:

stockprices, prices_raw, comp_const, CRSP_const, VIX = load_data()
pricy = stockprices.iloc[:5,:5]
stockprices = pricy
FCFF = pd.DataFrame(np.random.randint(0,100,size=(len(pricy), len(pricy.columns))), index=pricy.index, columns=pricy.columns)
VIX = pd.DataFrame(np.random.randint(0,100,size=(len(pricy), 1)), index=pricy.index, columns=['Adj Close'])

optimize(stockprices,options,FCFF,VIX,np.nan)
objective(0.9,0.9,0,1,stockprices,options,FCFF,VIX,np.nan)
'''

# # Strategy Evaluation

def max_dd(ser):
    ser    = pd.Series(ser)
    zzmax  = ser.expanding().max()
    zzmin  = ser[-1::-1].expanding().min()[-1::-1]
    mdd    = (zzmin/zzmax-1).min()
    return mdd

def get_optionsdata_for_year(year):
    store = 'something'


# ToDo: implement filtering based on membership
def evaluate_strategy(
        coverage=1,
        quantile=0.5,
        strike_0=1,
        strike_1=0.01,
        stockprices=None,
        options=None,
        FCFF=None,
        VIX=None,
        membership=None,
        initial_cash = 10**6,
        rebalancing_frequency = 4,
        ddwin = 30):

    # define initial portfolio with cash only
    # loop through all assigned dates
    # call timestep at each period
    # determine overall success after all time
    # return report of success
    '''
    pricy = stockprices.iloc[:5,:5]
    stockprices = pricy
    FCFF = pd.DataFrame(np.random.randint(0,100,size=(len(pricy), len(pricy.columns))), index=pricy.index, columns=pricy.columns)
    VIX = pd.DataFrame(np.random.randint(0,100,size=(len(pricy), 1)), index=pricy.index, columns=['Adj Close'])
    '''
    '''
    day = dt.datetime.strptime('1990-01-03', '%Y-%m-%d')
    day2 = dt.datetime.strptime('1990-01-04', '%Y-%m-%d')

    tuples = list(zip(*[[day,day,day,day,day,day2,day2,day2,day2,day2], [10,20,30,40,50,10,20,30,40,50]]))
    index = pd.MultiIndex.from_tuples(tuples, names=['date', 'strike'])
    options = pd.DataFrame(np.random.randint(0, 100, size=(len(pricy)*2, len(pricy.columns))), index=index,
                           columns=pricy.columns)
    options.xs(20, level='strike')
    options.xs(day, level='date')
    options.xs(day, level='date')[10057.0]
    options.xs(day, level='date')[options.xs(day, level='date')[10057.0] > 7]
    options.xs(day, level='date')[options.xs(day, level='date').index > 30]
    options.xs(day, level='date').iloc[1,1]
    options.xs(day, level='date').loc[20]
    '''

    options = pd.DataFrame(index=pricy.index, columns=pricy.columns)
    def single_optionset():
        return pd.DataFrame([
            [20,2,np.random.randint(1,10),np.random.rand(), np.random.rand()*5],
            [30,2,np.random.randint(1,10),np.random.rand(), np.random.rand()*5],
            [40,2,np.random.randint(1,10),np.random.rand(), np.random.rand()*5],
            [50,2,np.random.randint(1,10),np.random.rand(), np.random.rand()*5],
            [60,2,np.random.randint(1,10),np.random.rand(), np.random.rand()*5],
            ],
            columns=['strike','daysToExpiry','price','delta','implied_volatility']
        )
    #options = options.applymap(lambda x: single_optionset())


    # keeping only the top q quantile
    FCFF_filtered = FCFF[FCFF > list(FCFF.quantile(quantile,axis=1))]
    stockprices_filtered = stockprices + FCFF_filtered*0
    target_strike = stockprices_filtered.multiply((strike_0 + strike_1 * VIX)["Adj Close"], axis="index")

    portfolio = {}
    portfolio['metrics']            = pd.DataFrame(index=stockprices.index, columns=['portfolio_value', 'payments', 'earnings', 'returns'])
    portfolio['cash']               = pd.DataFrame(index=stockprices.index, columns=['cash'])
    portfolio['amounts']            = pd.DataFrame(index=stockprices.index, columns=stockprices.columns)
    portfolio["strikes"]            = pd.DataFrame(index=stockprices.index, columns=stockprices.columns)
    portfolio["daysToExpiry"]       = pd.DataFrame(index=stockprices.index, columns=stockprices.columns)
    portfolio["expiry"]             = pd.DataFrame(index=stockprices.index, columns=stockprices.columns)
    portfolio["price"]              = pd.DataFrame(index=stockprices.index, columns=stockprices.columns)
    portfolio["delta"]              = pd.DataFrame(index=stockprices.index, columns=stockprices.columns)
    portfolio["implied_volatility"] = pd.DataFrame(index=stockprices.index, columns=stockprices.columns)

    day.date().year

    reb_freq = 7
    k = 0
    for year in range(1996,2016):
        pass

    previous_year = 0
    for day in options.index:
        if previous_year != day.year:
            options_data_year = get_optionsdata_for_year(day.year)

        opday = options_data_year.loc[options_data_year.date == day]
        for stock in options.loc[day].index:
            opstock = opday.loc[opday.id == int(stock)]

            if not len(opstock)!=0:
                best_fit_index = opstock.iloc[np.absolute((opstock['strike'] - target_strike.loc[day,stock]).values).argsort()].index[0]
                portfolio["strikes"].loc[day, stock]            = opstock.loc[best_fit_index]['strike']
                portfolio["daysToExpiry"].loc[day, stock]       = opstock.loc[best_fit_index]['daysToExpiry']
                portfolio["expiry"].loc[day, stock]             = day + dt.timedelta(days=opstock.loc[best_fit_index]['daysToExpiry'])
                portfolio["price"].loc[day, stock]              = opstock.loc[best_fit_index]['price']
                portfolio["delta"].loc[day, stock]              = opstock.loc[best_fit_index]['delta']
                portfolio["implied_volatility"].loc[day, stock] = opstock.loc[best_fit_index]['implied_volatility']


    risk = portfolio["implied_volatility"] * portfolio["delta"]

    weights = (risk.T / risk.sum(axis=1)).T
    allocation = (weights / stockprices) / coverage
    portfolio['metrics'].loc[stockprices.index[0], 'portfolio_value'] = initial_cash
    portfolio['cash'].iloc[0] = initial_cash

    for i in range(len(stockprices.index)):
        investment_volume = portfolio['metrics'].portfolio_value.iloc[i] / rebalancing_frequency
        portfolio['amounts'].iloc[i] = ((investment_volume * allocation.iloc[i])).apply(lambda x: np.floor(x))
        payments = 0
        for stock in stockprices.columns:
            for j in range(-5,0):
                if i+j > 0:
                    expiry = portfolio["expiry"].iloc[i+j].loc[stock]
                    if (not pd.isnull(expiry)) and expiry >= stockprices.index[i]:
                        payments = payments + max(0,portfolio["strikes"].iloc[i+j].loc[stock] - stockprices.iloc[i].loc[stock])
                        portfolio['expiry'].loc[stockprices.index[i+j], stock] = np.nan
                        #payments = payments + sum(portfolio["strikes"].iloc[i+j] - stockprices.iloc[i]).apply(lambda x: max(0,x))

        np.nan_to_num(portfolio['amounts'].iloc[i] * portfolio['price'].iloc[i])
        earnings = sum((portfolio['amounts'].iloc[i] * portfolio['price'].iloc[i]).apply(lambda x: np.nan_to_num(x)))
        portfolio['metrics'].loc[stockprices.index[i],'payments'] = payments
        portfolio['metrics'].loc[stockprices.index[i],'earnings'] = earnings

        if not i+1 >= len(stockprices.index):
            portfolio['cash'].iloc[i+1] = portfolio['cash'].iloc[i] - payments + earnings
            # very rough approximation
            portfolio['metrics'].loc[stockprices.index[i + 1], 'portfolio_value'] = portfolio['cash'].iloc[i+1].cash


    ann = 251 / rebalancing_frequency
    portfolio['metrics'].returns = portfolio['metrics'].portfolio_value.pct_change()
    portfolio_returns       = portfolio['metrics'].returns.mean()*ann
    portfolio_vola          = portfolio['metrics'].returns.std()*np.sqrt(ann)
    portfolio_sharperatio   = portfolio_returns / portfolio_vola
    portfolio_maxdrawdown   = portfolio['metrics'].portfolio_value.rolling(window=ddwin).apply(max_dd).min()

    return (portfolio_sharperatio, portfolio_returns, portfolio_maxdrawdown)

'''
stockprices, prices_raw, comp_const, CRSP_const, VIX = load_data()
evaluate_strategy(stockprices = stockprices)
'''

'''
# # TimeStep (obsolete)

def timestep(
        time,
        portfolio,
        coverage,
        quantile,
        strike_0,
        strike_1,
        stockprices,
        options,
        FCFF,
        VIX,
        membership,):
    # determine P&L from maturing batch of Puts
    # pick stocks
    # pick strikes
    # pick weights (depending on remaining capital)
    # return updated portfolio

    portfolio_new = portfolio
    return portfolio_new
'''

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
