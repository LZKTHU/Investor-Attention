from jqdata import *
from jqdata import finance 
from jqfactor import get_factor_values
from lib_analysis import *

import empyrical as em 
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import time
import datetime
from datetime import datetime, timedelta
import math
import itertools
import scipy.stats as st
import statsmodels.api as sm
from functools import partial
from pandas.tseries.offsets import QuarterEnd, YearEnd
from pandas import Series, DataFrame
import seaborn as sns
import copy
import pickle
from collections import OrderedDict
import warnings
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings('ignore')
mpl.rcParams['font.sans-serif']=['SimHei'] 
mpl.rcParams['axes.unicode_minus']=False 


def get_index_univ(index_id, date, n=90):
    """
    获取指数某天的成分股
    params:
        index_id：指数代码
        date：日期
        n：次新股天数
    return：
        univ：指数某天的成分股
    """
    try:
        stockList = get_index_stocks(index_id, date)
    except:
        stockList = get_industry_stocks(index_id, date)

    # 剔除科创板股票
    stockList = [stk for stk in stockList if stk[:3] !='688']

    # 剔除st股
    st_data = get_extras('is_st', stockList, count = 1, end_date=date)
    stockList = [stock for stock in stockList if not st_data[stock][0]]

    # 剔除n天内的次新股
    univ = []
    dt = datetime.strptime(date, "%Y-%m-%d")
    for stock in stockList:
        start_date = get_security_info(stock).start_date
        if start_date < (dt-timedelta(days = n)).date():
            univ.append(stock)
    return univ
 
 
 
def get_rank_ic(factor, forward_return):
    """
    计算因子的信息系数
    params：
        factor:DataFrame，index为日期，columns为股票代码，value为因子值
        forward_return:DataFrame，index为日期，columns为股票代码，value为下一期的股票收益率
    return：
        DataFrame:index为日期，columns为IC，IC t检验的pvalue
    """
    common_index = factor.index.intersection(forward_return.index)
    ic_data = pd.DataFrame(index=common_index, columns=['IC','pValue'])

    for dt in ic_data.index:
        tmp_factor = factor.ix[dt]
        tmp_ret = forward_return.ix[dt]
        cor = pd.DataFrame(tmp_factor)
        ret = pd.DataFrame(tmp_ret)
        cor.columns = ['corr']
        ret.columns = ['ret']
        cor['ret'] = ret['ret']
        cor.dropna(inplace=True)
        if len(cor) < 5:
            continue

        ic, p_value = st.spearmanr(cor['corr'], cor['ret'])   
        ic_data['IC'][dt] = ic
        ic_data['pValue'][dt] = p_value
    return ic_data
 
def group_ret(fac, forward_return, n_quantile=10, hedging=True):
    """
    计算分组超额收益
    params：
        factor:DataFrame，index为日期，columns为股票代码，value为因子值
        forward_return:DataFrame，index为日期，columns为股票代码，value为收益率
        n_quantile:int，分组数量
        hedging: 是否对冲所选域的平均收益
    return：
        DataFrame：index为日期，columns为分组序号，值为每个调仓周期的组合收益率
    """
    # 统计分位数
    cols_mean = [i+1 for i in range(n_quantile)]
    cols = cols_mean

    common_index = fac.index.intersection(forward_return.index)
    excess_returns_means = pd.DataFrame(index=common_index, columns=cols)
    for dt in excess_returns_means.index:
        qt_mean_results = []
        tmp_factor = fac.loc[dt].dropna()
        tmp_return = forward_return.loc[dt].dropna()
        tmp_return = tmp_return.loc[tmp_factor.index]
        tmp_return_mean = tmp_return.mean()

        pct_quantiles = 1.0 / n_quantile
        for i in range(n_quantile):
            down = tmp_factor.quantile(pct_quantiles*i)
            up = tmp_factor.quantile(pct_quantiles*(i+1))
            
            i_quantile_index = tmp_factor[(tmp_factor<=up) & (tmp_factor>=down)].index
            if hedging == True:
                # 计算超额收益
                mean_tmp = tmp_return[i_quantile_index].mean() - tmp_return_mean
            elif hedging == False:
                mean_tmp = tmp_return[i_quantile_index].mean()
            qt_mean_results.append(mean_tmp)
            
        excess_returns_means.loc[dt] = qt_mean_results
    return excess_returns_means
 
def group_ret_plot(group_return, title):
    """
    分组收益绘图
    group_return：分组收益，index为日期，columns为分组序号，值为每个调仓周期的组合收益率
    title：str
    """
    fig = plt.figure(figsize=(7, 5))
    ax1 = fig.add_subplot(111)
    
    # 分组平均收益
    excess_returns_means_dist = group_return.mean()
    lns1 = ax1.bar(excess_returns_means_dist.index, excess_returns_means_dist.values, 
                   align='center', color='#0099ff', width=0.35)

    ax1.set_xlim(left=0.5, right=len(excess_returns_means_dist)+0.5)
    ax1.set_xticks(excess_returns_means_dist.index)
    ax1.set_title(title, fontsize=16)
    plt.show()

def factor_report(factor_dict, forward_return, direction, n_quantile, freq):
    """
    params：
        factor_dict：dict，key为因子名，值为DataFrame(index为日期，columns为股票代码，value为因子值)
        forward_return：DataFrame，index为日期，columns为股票代码，value为股票收益率
        n_quantile:int，分组数量
        direction：因子方向，-1为第十组多头，1为第一组多头
        freq：数据频率，'monthly', 'weekly' , 'daily'
    return：
        DataFrame：因子的IC，IC_IR，多空分析
    """
    all_report = pd.DataFrame()
    for factor_name, factor_a_neu in factor_dict.items():

        rank_ic_neu_a = get_rank_ic(factor_a_neu, forward_return)
        
        rank_ic_neu_a_po = rank_ic_neu_a[rank_ic_neu_a['IC'] > 0]
        positive_ratio = float(len(rank_ic_neu_a_po[rank_ic_neu_a_po['pValue'] <= 0.05])
                      ) / len(rank_ic_neu_a)

        rank_ic_neu_a_ne = rank_ic_neu_a[rank_ic_neu_a['IC'] < 0]
        negative_ratio = float(len(rank_ic_neu_a_ne[rank_ic_neu_a_ne['pValue'] <= 0.05])
                      ) / len(rank_ic_neu_a)

        rank_ic_neu_a_mean = rank_ic_neu_a['IC'].mean()
        rank_ic_ir_neu_a = rank_ic_neu_a['IC'].mean() / rank_ic_neu_a['IC'].std()

        a_excess_returns = group_ret(factor_a_neu, forward_return, n_quantile, hedging=True)
        a_long_short_ret = (a_excess_returns.iloc[:, np.sign(direction-1)] - 
                            a_excess_returns.iloc[:, -np.sign(direction+1)]).fillna(0)
        
        a_long_short_annual_ret = em.annual_return(a_long_short_ret, period=freq)
        a_long_short_annual_vol = em.annual_volatility(a_long_short_ret, period=freq)

        a_long_short_sharp_ratio = em.sharpe_ratio(a_long_short_ret, risk_free=0, period=freq)

        a_long_short_max_drawdown = em.max_drawdown(a_long_short_ret)

        report = pd.DataFrame(index=[factor_name], 
                          columns=['IC', 'IC_IR', '正显著率', '负显著率', '多空组合年化收益率', 
                                   '多空组合年化波动率', '多空组合夏普比率', '多空组合最大回撤'])

        report.loc[factor_name, 'IC'] = str(np.round(rank_ic_neu_a_mean * 100, 2)) + '%'
        report.loc[factor_name, 'IC_IR'] = str(np.round(rank_ic_ir_neu_a, 2))
        report.loc[factor_name, '正显著率'] = str(np.round(positive_ratio * 100, 2)) + '%'
        report.loc[factor_name, '负显著率'] = str(np.round(negative_ratio * 100, 2)) + '%'
        report.loc[factor_name, '多空组合年化收益率'] = str(np.round(a_long_short_annual_ret * 100,2)) + '%'
        report.loc[factor_name, '多空组合年化波动率'] = str(np.round(a_long_short_annual_vol * 100,2)) + '%'
        report.loc[factor_name, '多空组合夏普比率'] = str(np.round(a_long_short_sharp_ratio, 2))
        report.loc[factor_name, '多空组合最大回撤'] = str(np.round(a_long_short_max_drawdown * 100,2)) + '%'
        all_report = all_report.append(report)
    return all_report
 
def ICIR_weight(factor_dict, stock_return, window):
    """
    IC_IR_加权
    输入：
        factor_dict：dict，key为因子名，值为DataFrame(index为日期，columns为股票代码，value为因子值)
        stock_return：DataFrame，index为日期，columns为股票代码，value为股票收益率
        window：IC_IR的滚动期
    返回：
        DataFrame：最终合成后的因子
    """
    all_rolling_ic_ir_list = []
    for factor_name, factor in factor_dict.items():
        # 滞后1期
        factor_pre1m = factor.shift(1)
        ic = get_rank_ic(factor_pre1m, stock_return)['IC']
        ic_ir = (ic.rolling(window=window, min_periods=window).mean() / 
                ic.rolling(window=window, min_periods=window).std()
                )
        ic_ir.name = factor_name
        all_rolling_ic_ir_list.append(ic_ir)

    all_rolling_ic_ir_df = pd.concat(all_rolling_ic_ir_list, axis=1)
    all_rolling_ic_ir_df[all_rolling_ic_ir_df < 0] = 0.001
    all_rolling_ic_ir_df = all_rolling_ic_ir_df.divide(all_rolling_ic_ir_df.abs().sum(axis=1), axis=0)

    weighted_factor = 0
    for factor_name, factor in factor_dict.items():
        weighted_factor += factor.multiply(all_rolling_ic_ir_df[factor_name], axis=0)
    
    weighted_factor = weighted_factor.dropna(how='all')
    all_rolling_ic_ir_df = all_rolling_ic_ir_df.dropna(how='all')
    return weighted_factor, all_rolling_ic_ir_df
 
def equal_weight(factor_dict):
    """
    因子等权
    输入：
        factor_dict：dict，key为因子名，值为DataFrame(index为日期，columns为股票代码，value为因子值)
    返回：
        DataFrame：最终合成后的因子
    """
    equal_factor = 0
    for factor_name, factor in factor_dict.items():
        equal_factor += factor
    return equal_factor
 
def get_period_date(peroid, start_date, end_date):
    """
    获取指定周期的日期列表
    param:
        peroid: 周是'W',月'M',季度线'Q',五分钟'5min',12天'12D'
        start_date: 开始时间
        end_date: 结束时间
    return:
        list，开始至结束时间的指定周期的日期列表
    """
    stock_data = get_price('000001.XSHE',start_date,end_date,'daily',fields=['close'])
    stock_data['date']=stock_data.index
    period_stock_data=stock_data.resample(peroid).last()
    
    date=period_stock_data.index
    pydate_array = date.to_pydatetime()
    date_only_array = np.vectorize(lambda s: s.strftime('%Y-%m-%d'))(pydate_array )
    date_only_series = pd.Series(date_only_array)

    date_list = date_only_series.values.tolist()
    TradeDate = []
    for i in date_list:
        temp = list(get_trade_days(end_date=i, count=1))[0]
        TradeDate.append(str(temp))
    return TradeDate
 
def get_JQfactor(date, univ):
    """
    获取聚宽因子库因子
    param:
        date: 日期
        univ: 股票池
    return:
        df_jq_factor，某一天的聚宽因子库因子
    """
    factors_list = ['sale_expense_to_operating_revenue', # 营业费用与营业总收入之比
                    'gross_income_ratio', # 销售毛利率
                    'cfo_to_ev', # 经营活动产生的现金流量净额与企业价值之比TTM
                    'roe_ttm', # 权益回报率TTM
                    'Turnover60', # 60日平均换手率
                    'Skewness20', # 个股收益的20日偏度
                    'Skewness60', # 个股收益的60日偏度
                    'residual_volatility', # 残差波动率
                   ]
    
    factor_data = get_factor_values(securities=univ, 
                                    factors=factors_list, 
                                    count=1, 
                                    end_date=date)
    
    factor_df = pd.DataFrame(index=univ)
    for i in factor_data.keys():
        factor_df[i]=factor_data[i].iloc[0,:]
    factor_df.index.name = 'code'
    return factor_df.reset_index()
 
def get_JQfactor_quarter(date, univ):
    """
    获取聚宽单季度因子
    param:
        date: 日期
        univ: 股票池
    return:
        df_jq_factor，某一天的聚宽单季度因子
    """
    q = query(indicator.code, 
              valuation.pe_ratio, #市盈率（TTM）
              valuation.pb_ratio, #市净率（TTM）
              indicator.inc_revenue_year_on_year,  #营业收入增长率（同比）
              indicator.inc_net_profit_year_on_year, #净利润增长率（同比）
              indicator.inc_operation_profit_year_on_year, #营业利润增长率（同比）
              valuation.market_cap, #总市值
             ).filter(
              indicator.code.in_(univ)
             )
    factor_q_df = get_fundamentals(q, date = date)
    return factor_q_df

def get_cum_JQfactor(date_list, univ):
    """
    获取一段时间内的聚宽因子
    param:
        date_list: 日期列表
        univ: 股票池
    return:
        cum_df，一段时间内的聚宽因子
    """
    cum_list = []
    for date in date_list:
        factor_df = get_JQfactor(date, univ)
        factor_q_df = get_JQfactor_quarter(date, univ)

        temp = pd.merge(factor_df, factor_q_df, on=['code'])
        temp['date'] = date
        cum_list.append(temp)
    cum_df = pd.concat(cum_list)   
    return cum_df

def get_additional_factors(date_list, index_id):
    '''
    返回指定日期的各个因子值
    param:
        date: 查询因子的日期
    return:
        DataFrame，所查询日期的各个因子值
    '''
    def get_factors(date, index_id):
        code=get_index_univ(index_id=index_id, date=date)
        # w=weight.pivot(index='date',columns='code',values='weight')
        
        q = query(valuation.turnover_ratio,
                valuation.market_cap,
                ).filter(valuation.code.in_(code))
        df1=get_fundamentals_continuously(q, end_date=date, count=20,panel=True)

        #计算CGO所需的权重
        df1['turnover_ratio']=df1['turnover_ratio']/100
        df1['turnover_ratio'].fillna(inplace=True,method='ffill')
        df1['turnover_ratio'].fillna(inplace=True,value=0)
        turn=df1['turnover_ratio'].sort_index(ascending=False)
        turn.iloc[0]=0
        turn=1-turn
        cgo_weight=turn.cumprod().sort_index(ascending=True)*df1['turnover_ratio']/(1-df1['turnover_ratio'])
  
        #获取行情数据
        price=get_price(code, 
              end_date=date, 
              frequency='daily', 
              fields=['open', 'close', 'low', 'high', 'volume', 'money','avg', 'pre_close'], 
              fq='pre',
              count=20,
              panel=True, fill_paused=True)
  
  #计算各个因子
  #1.计算市值因子
        factor=pd.DataFrame(df1['market_cap'].T[df1['market_cap'].index[-1]])
        factor['date']=factor.columns[0]
        factor['code']=factor.index
        factor=factor.rename(columns={factor.columns[0]:'market_cap'})
        factor['logcap']=factor[['market_cap']].applymap(lambda x:math.log(x))
        #2.计算R20
        factor['close_L20']=price['pre_close'].T[price['pre_close'].index[0]]
        factor['close']=price['close'].T[price['close'].index[-1]]
        factor['R20']=(factor['close']-factor['close_L20'])/factor['close_L20']/20
        #3.计算最大换手率
        factor['max_turnover']=df1['turnover_ratio'].rolling(20).max().iloc[-1]
        #4.计算CGO
        factor['avg_price']=price['avg'].iloc[-1]
        factor['C']=(((price['avg']*cgo_weight).cumsum())/cgo_weight.cumsum()).iloc[-2]
        factor['CGO']=(factor['avg_price']-factor['C'])/factor['avg_price']


        #做回归得到残差，从而计算RCGO
        results = smf.ols('CGO ~ R20 + logcap', data=factor).fit()
        factor['RCGO']=results.resid

        return factor[['date','code','max_turnover','RCGO','C','CGO','R20','logcap']]

    for date in date_list:
        if date==date_list[0]:
            factor=get_factors(date, index_id)
        else:
            factor=factor.append(get_factors(date, index_id),ignore_index=True)
   
    return factor

def get_cum_factor(date_list, index):

    univ = []
    for date in date_list:
        temp = get_index_univ(index, date)
        univ = list(set(univ + temp))
    
    raw_data = get_cum_JQfactor(date_list, univ)
    
    additional_data = pd.read_csv('investor_attention.csv')
    raw_data['PNN'] = additional_data['PNN']
    raw_data['NNN'] = additional_data['NNN']
    raw_data['SAAvg'] = additional_data['SAAVG']
    raw_data['SN'] = additional_data['SN']
    raw_data['PN'] = additional_data['PN']
 
    raw_data['market_cap'] = np.log(raw_data['market_cap'])
    raw_data = raw_data.replace([np.inf, -np.inf], np.nan)

    col_list = [x for x in raw_data.columns.tolist() if x != 'date' and x != 'code']
    raw_data = raw_data[['date','code']+col_list]

    # 因子处理，动态股票池
    factor_dict = {}
    for date in date_list:
        factor_dict[date] = get_index_univ(index, date)

    raw_data_df = pd.DataFrame()   
    for date in sorted(factor_dict.keys()):
        temp = raw_data[raw_data['date'] == date].copy()
        temp = temp[temp['code'].isin(factor_dict[date])]
        raw_data_df = raw_data_df.append(temp)  
    raw_data = raw_data_df
 
    addition_factor = get_additional_factors(date_list, index)
 
    addition_factor = addition_factor[['date', 'code', 'RCGO', 'max_turnover']]
    raw_data = raw_data.merge(addition_factor, on=['date', 'code'])
 
    return raw_data

def get_processed_cum_factor(date_list, index):
    raw_data = get_cum_factor(date_list, index)
 
    inv_factors = {"reciprocal":['pe_ratio', 'pb_ratio'], 
               
               "reverse":['NNN', 'Turnover60', 'Skewness20', 'Skewness60', 
                         'residual_volatility']
              }

    raw_data = factor_ascending(raw_data, inv_factors)

    col_list = [x for x in raw_data.columns.tolist() if x != 'date' and x != 'code']
    raw_data = mad_winsorize(raw_data, col_list)
    raw_data = standardized(raw_data, col_list)

    raw_data_list = []
    Y_factor_list = [x for x in col_list if x != 'market_cap']

    for Y_factor in Y_factor_list:
        temp_data = factor_regression(raw_data, Y_factor, 'market_cap')
        raw_data_list.append(temp_data.set_index(['date','code']))
    raw_data = pd.concat(raw_data_list,axis=1).reset_index()

    raw_data = mad_winsorize(raw_data, Y_factor_list)
    raw_data = standardized(raw_data, Y_factor_list)

    raw_data = raw_data.fillna(0)

    return raw_data

def mad_winsorize(dframe, col_list, sigma_n=4.449):
    """
    param:
        dframe: panel/横截面/时间序列数据, 列至少包括['date', 'code', col_list]
        col_list: 需要进行winsorize的因子列表
        sigma_n: 3*1.483=4.449
    return:
        经过去极值后的dframe
    """
    def mad_winsor_by_day(dframe_tdate, col_list, sigma_n):
        """
        按照[dm+sigma_n*dm1, dm-sigma_n*dm1]进行winsorize
        dm: median
        dm1: median(abs(origin_data - median)), 即 MAD值
        param:
            dframe_tdate: 某一期的多个因子值的dataframe
        return
            去极值后的dframe_tdate
        """
        dm = dframe_tdate[col_list].median()
        dm1 = (dframe_tdate[col_list] - dm).abs().median()
        upper = dm + sigma_n * dm1
        lower = dm - sigma_n * dm1
        for col in col_list:
            tmp_col = dframe_tdate[col]
            tmp_col[tmp_col > upper[col]] = upper[col]
            tmp_col[tmp_col < lower[col]] = lower[col]
            dframe_tdate[col] = tmp_col
        return dframe_tdate
    
    dframe = dframe.groupby(['date']).apply(mad_winsor_by_day, col_list, sigma_n)
    return dframe

def standardized(dframe, col_list):
    """
    param:
        dframe: panel/横截面/时间序列数据, 列至少包括['date', 'code', col_list]
        col_list: 需要进行standardized的因子列表
    return:
        经过标准化后的dframe
    """
    dframe[col_list] = dframe.groupby('date')[col_list].apply(lambda df: (df-df.mean())/df.std())
    return dframe

def factor_regression(total_df, Y_factor, X_factors):
    """
    param:
        total_df: panel/横截面/时间序列数据, 列至少包括['date', 'code', col_list]
        Y_factor: 因子y
        X_factors: 因子x
    return:
        经过回归后的dframe，包括['date', 'code', Y_factor]
    """
    all_dates = sorted(list(set(total_df['date'])))
    total_out = pd.DataFrame()
    for date in all_dates:
        part = total_df[total_df['date']==date].copy()
        part = part[['date', 'code']+[Y_factor, X_factors]]
        part = part.dropna()
        X_values = part[X_factors].values
        Y_values = part[Y_factor].values
        try:
            result = sm.OLS(Y_values, sm.add_constant(X_values)).fit()     
            Y_resi = result.resid
        except:
            Y_resi = np.nan
            
        part[Y_factor] = Y_resi
        total_out = pd.concat([total_out, part])
    total_out = total_out[['date', 'code']+[Y_factor]]
    return total_out
 
def factor_ascending(factor_frame, inv_factors):
    """
    调整因子方向
    param:
        factor_frame：dataframe，列为：['date', 'code'] + factor list
        inv_factors：dict，{"reciprocal":['PE', 'PB'], "reverse":['Price1M']}
    return:
        调整因子方向后的 factor_frame
    """
    factor_frame = factor_frame.copy()
    for inv_type in inv_factors:
        # 取倒数
        if inv_type == 'reciprocal':
            for factor_name in inv_factors[inv_type]:
                if factor_name not in factor_frame.columns:
                    continue
                factor_frame[factor_name] = 1.0/factor_frame[factor_name]       
        # 取负数
        elif inv_type == 'reverse':
            for factor_name in inv_factors[inv_type]:
                if factor_name not in factor_frame.columns:
                    continue
                factor_frame[factor_name] = -1.0*factor_frame[factor_name]
    return factor_frame
 
def get_forward_return(univ, date_list, n=1):
    """
    股票未来n期的收益率
    params：
        date_list：交易日列表
        univ：代码列表
        n：期数
    return：
        price_df：股票未来n期的收益率
    """
    price_list = []
    for dt in date_list:
        univ_price = get_price(univ, 
                              start_date=dt, 
                              end_date=dt,
                              frequency='daily',
                              fields=['close'],
                              fq='pre',
                              panel=False)
        price_list.append(univ_price)

    price_df = pd.concat(price_list)
    price_df = price_df.pivot(index='time', columns='code', values='close')
    price_df = price_df.pct_change(n).shift(-n)
    
    price_df.index = [dt.strftime('%Y-%m-%d') for dt in price_df.index]
    price_df.index.name = 'date'
    price_df = price_df.stack().to_frame('forward_return').reset_index()
    return price_df
 
def get_monthly_return(univ, date_list):
    """
    股票月收益率
    params：
        date_list：交易日列表
        univ：代码列表
    return：
        price_df：股票月收益率
    """
    price_list = []
    for dt in date_list:
        univ_price = get_price(univ, 
                              start_date=dt, 
                              end_date=dt,
                              frequency='daily',
                              fields=['close'],
                              fq='pre',
                              panel=False)
        price_list.append(univ_price)

    price_df = pd.concat(price_list)
    price_df = price_df.pivot(index='time', columns='code', values='close')
    price_df = price_df.pct_change(1).dropna(how='all')
    
    price_df.index = [dt.strftime('%Y-%m-%d') for dt in price_df.index]
    price_df.index.name = 'date'
    return price_df
 
def get_daily_return(univ, start_date, end_date):
    """
    股票日收益率
    params：
        date_list：交易日列表
        univ：代码列表
    return：
        price_df：股票日收益率
    """
    univ_price = get_price(univ, 
                          start_date=start_date, 
                          end_date=end_date,
                          frequency='daily',
                          fields=['close'],
                          fq='pre',
                          panel=False)

    price_df = univ_price.pivot(index='time', columns='code', values='close')
    price_df = price_df.pct_change(1).dropna(how='all')
    return price_df
 
def bucketize_by_quantile(factor, quantiles=10):
    """
    因子分组
    params：
        factor：series，index为date，code，values为因子值
        quantiles：分组数目
    """
    quantile_labels = list(range(1, quantiles + 1))
    quantile_groups = (factor
                       .groupby(level='date', group_keys=False)
                       .apply(pd.qcut,
                              q=quantiles,
                              labels=quantile_labels))
    return quantile_groups


def get_portfolios_from_quantile_groups(quantile_groups, quantile):
    """
    构建因子分组组合
    params：
        quantile_groups：函数bucketize_by_quantile输出的因子分组
        quantile：第几组
    """
    selected = quantile_groups[quantile_groups == quantile]
    
    def build_portfolio(x):
        # 等权组合, 权重为 1/len(x), index为股票代码---x.index的第二层
        return pd.Series(1 / len(x), index=x.index.get_level_values('code'))
    
    portfolios = OrderedDict()
    for date, x in selected.groupby(level='date'):
        portfolios[date] = build_portfolio(x)
    return portfolios


def net_value(rebalance_dates, portfolio_weights, daily_returns, cost_adj=True, cost=0.005):
    """
    计算组合净值
    params：
        rebalance_dates: list. 回测期内所有换仓日
        portfolio_weights: dict. 每个换仓日的目标组合. 
                                 key为换仓日, value为代表目标组合的Series, index为股票代码, value为权重
        daily_return: DataFrame. 回测期内所有股票的日度复权收益率. index为日期, columns为股票代码
        cost_adj: 是否加入交易费用
        cost: 交易费用比例（双边），默认千分之五
    """
    net_value_list = []
    pre_assets_held = []
    
    for start, end in zip(rebalance_dates[:-1], rebalance_dates[1:]):
        weights = portfolio_weights[start]
        
        # 每个持仓周期为左闭右开区间,以确保一个交易日属于且只属于一个持仓周期
        holding_period = (daily_returns.index >= start) & (daily_returns.index < end)
        assets_held = weights.index
        
        rtn = daily_returns.loc[holding_period, assets_held].fillna(0)
        temp_nav = period_net_value(rtn, weights)
        
        if cost_adj:
            cur_assets_held = list(assets_held)
            portfolio_change = ((len(cur_assets_held) - len(set(pre_assets_held) & set(cur_assets_held))) 
                                 / len(cur_assets_held)
                               )
            pre_assets_held = cur_assets_held

            temp_nav[-1] = temp_nav[-1] * (1 - portfolio_change * cost)
        net_value_list.append(temp_nav)
        
    res = merge_period_net_value(net_value_list)
    return res


def period_net_value(daily_returns, weights):
    """辅助函数：计算组合净值net_value
    """
    asset_net_value = (1 + daily_returns).cumprod()
    normalized_weights = weights / weights.sum()
    portf_net_value = asset_net_value.dot(normalized_weights)
    return portf_net_value


def merge_period_net_value(period_net_values):
    """辅助函数：计算组合净值net_value
    """
    net_value_list = []
    init_capital = 1
    for nv in period_net_values:
        nv *= init_capital
        net_value_list.append(nv)   
        # 下一段净值的初始资金是上一段最后一天的值
        init_capital = nv.iat[-1]
    res = pd.concat(net_value_list)
    # 整个回测期第一天的净值一定是1, 第一天的return其实用不到
    res.iloc[0] = 1
    return res
 
def get_dynamic_univ(date_list, index_id):
    """
    获取指数的动态成分股
    params：
        date_list：交易日列表
        index_id：指数代码
    """
    univ_list = []
    for date in date_list:
        temp_univ = get_index_univ(index_id, date)
        temp_df = pd.DataFrame(temp_univ, index=[date]*len(temp_univ))
        univ_list.append(temp_df)
    univ_df = pd.concat(univ_list)
    univ_df.index.name = 'date'
    univ_df.columns = ['code']
    return univ_df
 
def simple_group_backtest(factor, start_date, end_date, daily_returns, group=5, freq='M'):
    """
    简单分组回测
    params：
        factor：dataframe，index为date，columns为code，values为因子值
        start_date：开始时间
        end_date：结束时间
        group：分几组
        freq：调仓频率，默认月频，根据因子频率来设定
    """
    rebalance_dates = get_period_date(freq, start_date, end_date)
    rebalance_dates = [datetime.strptime(dt, '%Y-%m-%d') for dt in rebalance_dates]
    
    factor = factor.copy()
    factor.index = pd.to_datetime(factor.index)
    factor = factor.stack().dropna()
    factor.name = 'factor'

    quantile_groups = bucketize_by_quantile(factor, group)
    QUANTILE_GROUPS = list(range(1,group+1))

    quantile_portf_net_values = OrderedDict()
    for group in QUANTILE_GROUPS:
        portfolio_weights = get_portfolios_from_quantile_groups(quantile_groups, group)
        nv = net_value(rebalance_dates, portfolio_weights, daily_returns)
        quantile_portf_net_values[group] = nv

    pd.concat(quantile_portf_net_values, axis=1).plot(figsize=(18, 6), grid=True, 
                                                      title='因子分 '+str(group)+' 组回测')
               
def easy_backtest(factor1, factor2, start_date, end_date, daily_returns, group=5, freq='M'):
    """
    简单回测(策略和基准的对比)
    params：
        factor1,2：dataframe，index为date，columns为code，values为因子值
        start_date：开始时间
        end_date：结束时间
        group：第几组作为多头组合
        freq：调仓频率，默认月频，根据因子频率来设定
    return：
        nav_df：策略和基准的净值数据
    """
    rebalance_dates_M = get_period_date(freq, start_date, end_date)
    rebalance_dates_M = [datetime.strptime(dt, '%Y-%m-%d') for dt in rebalance_dates_M]
    
    rebalance_dates_Q = get_period_date('Q', start_date, end_date)
    rebalance_dates_Q = [datetime.strptime(dt, '%Y-%m-%d') for dt in rebalance_dates_Q]
    
    factor = factor1.copy()
    factor.index = pd.to_datetime(factor.index)
    factor = factor.stack().dropna()
    factor.name = 'factor'
    
    quantile_groups = bucketize_by_quantile(factor, group)
    portfolio_weights = get_portfolios_from_quantile_groups(quantile_groups, group)
    nav = net_value(rebalance_dates_M, portfolio_weights, daily_returns)
    
    factor = factor2.copy()
    factor.index = pd.to_datetime(factor.index)
    factor = factor.stack().dropna()
    factor.name = 'factor'

    quantile_groups = bucketize_by_quantile(factor, 1)
    portfolio_weights = get_portfolios_from_quantile_groups(quantile_groups, 1)
    bm_nav = net_value(rebalance_dates_Q, portfolio_weights, daily_returns)
    
    nav_df = pd.concat([nav, bm_nav], axis=1)
    nav_df.columns = ['strategy', 'benchmark']
    nav_df = nav_df.pct_change().dropna()
    nav_df['excess'] = nav_df['strategy'] - nav_df['benchmark']
    nav_df = (nav_df + 1).cumprod()
    return nav_df
 
def easy_BM_backtest(factor1, factor2, start_date, end_date, daily_returns, freq='M', group=1):
    """
    简单回测(基准1和基准2的对比)
    params：
        factor1,2：dataframe，index为date，columns为code，values为因子值
        start_date：开始时间
        end_date：结束时间
        freq：调仓频率，默认月频，根据因子频率来设定
        group：第几组作为多头组合
    return：
        nav_df：策略和基准的净值数据
    """
    rebalance_dates_M = get_period_date(freq, start_date, end_date)
    rebalance_dates_M = [datetime.strptime(dt, '%Y-%m-%d') for dt in rebalance_dates_M]
    
    factor = factor1.copy()
    factor.index = pd.to_datetime(factor.index)
    factor = factor.stack().dropna()
    factor.name = 'factor'

    quantile_groups = bucketize_by_quantile(factor, group)
    portfolio_weights = get_portfolios_from_quantile_groups(quantile_groups, group)
    nav = net_value(rebalance_dates_M, portfolio_weights, daily_returns)
    
    factor = factor2.copy()
    factor.index = pd.to_datetime(factor.index)
    factor = factor.stack().dropna()
    factor.name = 'factor'
    
    quantile_groups = bucketize_by_quantile(factor, group)
    portfolio_weights = get_portfolios_from_quantile_groups(quantile_groups, group)
    bm_nav = net_value(rebalance_dates_M, portfolio_weights, daily_returns)
    
    nav_df = pd.concat([nav, bm_nav], axis=1)
    nav_df.columns = ['strategy', 'benchmark']
    nav_df = nav_df.pct_change().dropna()
    nav_df['excess'] = nav_df['strategy'] - nav_df['benchmark']
    nav_df = (nav_df + 1).cumprod()
    return nav_df
 
def get_preiod_factor(statDate, univ):
    """
    获取单期因子
    param:
        statDate: 日期，‘2019’
        univ: 股票池
    return:
        factor_df，单期因子
    """
    q = query(indicator.code, 
              indicator.statDate,
              income.operating_revenue, #营业收入
              indicator.gross_profit_margin, #销售毛利率
              income.sale_expense, #销售费用
             ).filter(
              indicator.code.in_(univ)
             )
    factor_q_df = get_fundamentals(q, statDate = statDate)

    rd_expenses = sup.run_query(query(sup.STK_FINANCE_SUPPLEMENT
    ).filter(
    sup.STK_FINANCE_SUPPLEMENT.report_date==statDate+'-12-31',
    sup.STK_FINANCE_SUPPLEMENT.report_type==0,
    sup.STK_FINANCE_SUPPLEMENT.code.in_(med_univ))
                               )
    rd_expenses = rd_expenses[['report_date','code','rd_expenses']].dropna()
    rd_expenses['report_date'] = rd_expenses['report_date'].astype(str)

    factor_df = pd.merge(factor_q_df, rd_expenses, 
                           left_on=['statDate', 'code'], right_on=['report_date', 'code'])
    del factor_df['statDate']
    return factor_df
 
def get_idx_chgpct(start_date, end_date, index_univ):
    """
    股票和指数日频收益率
    param:
        start_date: 开始时间
        end_date：结束时间
        index_univ：list，股票和指数的代码
    return:
        idx_df：股票和指数日频收益率
    """
    idx_list = []
    for idx in index_univ:
        idx_price = get_price(idx, 
                              start_date=start_date, 
                              end_date=end_date,
                              frequency='daily',
                              fields=['pre_close', 'close'],
                              fq='pre')
        idx_price = (idx_price
        .assign(idx_chgPct=(idx_price['close'] - idx_price['pre_close']) / idx_price['pre_close'])
        .assign(code=idx)
        )
        idx_price = idx_price[['code', 'idx_chgPct']]
        idx_list.append(idx_price)
        
    idx_df = pd.concat(idx_list, axis=0)
    idx_df.index.name = 'date'
    return idx_df

def factor_ret_plot(factor, forward_return, name, quantile=10, freq='monthly'):
    """
    因子分组，多头/多空收益
    输入：
        factor：dataframe，index为date，columns为code，values为因子值
        forward_return：dataframe，index为date，columns为code，values为forward_return
        name：str，因子名称
        quantile：分组数目
        freq: 'monthly', 'weekly' , 'daily'
    """
    group_return_10 = group_ret(factor, forward_return, n_quantile=quantile, hedging=True)
    group_ret_plot(group_return_10, name+'因子分组表现')

    group_return_10 = group_ret(factor, forward_return, n_quantile=quantile, hedging=False)
    group_return_10.index = pd.to_datetime(group_return_10.index)
    (group_return_10 + 1).cumprod().plot(figsize=(12,5), grid=True, title=name+'因子分组多头净值曲线')

    group_return_10 = group_ret(factor, forward_return, n_quantile=quantile, hedging=True)
    group_return_10.index = pd.to_datetime(group_return_10.index)
    (group_return_10 + 1).cumprod().plot(figsize=(12,5), grid=True, title=name+'因子分组对冲净值曲线')

    group_return_10 = group_ret(factor, forward_return, n_quantile=quantile, hedging=False)
    group_return_10['多空组合'] = group_return_10[10]-group_return_10[1]
    
    cpt_dict = OrderedDict()
    for i in range(1, quantile+1):
        cpt_dict['第%s组多头组合'%str(i)] = group_return_10[i]
    cpt_dict['多空组合'] = group_return_10['多空组合']
    return group_cal_indicators(cpt_dict, freq)

def group_cal_indicators(group_cpt_dict, freq='monthly'):
    """
    N组 多头净值的指标
    输入：
        group_cpt_dict：dict，key为组合名称，值为组合净值：Series，index为日期，value为收益率
        freq: 'monthly', 'weekly' , 'daily'
    """
    group_cal_list = []
    for group_name, group_cpt in group_cpt_dict.items():
        cpt_df = group_cpt.to_frame(group_name).dropna()
        
        cal_ = cal_indicators(cpt_df, freq)
        group_cal_list.append(cal_)
    
    group_cal_df = pd.concat(group_cal_list, axis=1)
    return group_cal_df

def cal_indicators(return_df, freq='daily'):
    """
    计算净值的指标
    freq: 'monthly', 'weekly' , 'daily'
    """
    returns_df_ = return_df.copy()
    returns_df_ = pd.DataFrame(returns_df_, dtype=np.float)
    returns = returns_df_.squeeze()
    
    res = pd.DataFrame(index=['年化收益率', '年化波动率', '夏普比率', '最大回撤'], 
                       columns=returns_df_.columns, data=0.0)
    
    res.loc['年化收益率'] = str(np.round(em.annual_return(returns, period=freq) * 100, 2)) + '%'
    res.loc['年化波动率'] = str(np.round(em.annual_volatility(returns, period=freq) * 100, 2)) + '%'
    res.loc['夏普比率'] = str(np.round(em.sharpe_ratio(returns, risk_free=0, period=freq), 2))
    res.loc['最大回撤'] = str(np.round(em.max_drawdown(returns) * 100, 2)) + '%'
    return res  