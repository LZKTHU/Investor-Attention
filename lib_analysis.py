"""
Created on Mon May 5 09:23:16 2020

@author: 李泽坤
"""
from jqdata import *
from jqdata import finance 

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
from functools import partial
from pandas.tseries.offsets import QuarterEnd, YearEnd
from pandas import Series, DataFrame
import seaborn as sns
import copy
import pickle
from collections import OrderedDict
import warnings

warnings.filterwarnings('ignore')
mpl.rcParams['font.sans-serif']=['SimHei'] 
mpl.rcParams['axes.unicode_minus']=False 


"""
        
## class net_value_analysis

#### class net_value_analysis方法有：cal_indicators，win_prob，plot_ret，plot_hedging

- cal_indicators：计算策略评价指标（年化收益率','年化标准差','贝塔','夏普比率','信息比率','最大回撤）
- win_prob：策略相对基准的胜率（日，周，月，年度的胜率）
- plot_ret：画出策略，基准和超额收益的收益率（年度）
- plot_hedging：画出策略，基准，超额收益，超额收益的回撤

#### class net_value_analysis的参数有：
- 策略净值数据，dataframe，columns为：{'strategy'：策略净值，'benchmark'：基准净值，'excess'：超额收益净值}


## class portfolio_analysis

#### class portfolio_analysis方法有：industry_exposure，cap_exposure

- industry_exposure：相对基准的行业暴露
- cap_exposure：相对基准的市值暴露

#### class portfolio_analysis的参数有：
- 策略持仓数据，dataframe，columns为：{'date'：日期，'code'：股票代码，'weight'：股票权重}

"""


        
class net_value_analysis(object):
    
    # 定义函数中不同的变量
    def __init__(self, nav_excess_df=None):
        self.nav_df = nav_excess_df[['strategy','benchmark']]
        self.nav_excess_df = nav_excess_df
        self.ret_dict = {}


    def cal_indicators(self):
        """计算评价指标
        """
        nav_df = self.nav_df
        df_daily_return = nav_df.pct_change().dropna(how='all', axis=0)
        res = pd.DataFrame(index=['年化收益率','年化标准差','贝塔','夏普比率','信息比率','最大回撤'], columns=
                           df_daily_return.columns, data=0.0)

        res.loc['年化收益率'] = (df_daily_return.mean() * 250).apply(lambda x: '%.2f%%' % (x*100))
        res.loc['年化标准差'] = (df_daily_return.std() * np.sqrt(250)).apply(lambda x: '%.2f%%' % (x*100))
        res.loc['贝塔', 'benchmark'] = 1
        res.loc['贝塔', 'strategy'] = np.round((np.cov(df_daily_return['strategy'],df_daily_return['benchmark'])[0,1] 
                                                                          / np.cov(df_daily_return['benchmark'])), 2)

        res.loc['夏普比率'] = (df_daily_return.mean() / df_daily_return.std() * np.sqrt(250)).apply(lambda x: np.round(x, 2))

        res.loc['信息比率', 'strategy'] = np.round(((df_daily_return['strategy'].mean()-df_daily_return['benchmark'].mean()) 
                                 / (df_daily_return['strategy']-df_daily_return['benchmark']).std() * np.sqrt(250)), 2)
 
        def cal_maxdrawdown(data):
            """计算最大回撤
            """
            if isinstance(data, list):
                data = np.array(data)
            if isinstance(data, pd.Series):
                data = data.values

            def get_mdd(values): 
                dd = [values[i:].min() / values[i] - 1 for i in range(len(values))]
                return abs(min(dd))

            if not isinstance(data, pd.DataFrame):
                return get_mdd(data)
            else:
                return data.apply(get_mdd)
        
        res.loc['最大回撤'] = cal_maxdrawdown(nav_df).apply(lambda x: '%.2f%%' % (x*100))
        return res 
    
    
    def win_prob(self):
        """策略相对基准的胜率
        """
        nav_df = self.nav_df
        start_date = nav_df.index.min().strftime('%Y-%m-%d')
        end_date = nav_df.index.max().strftime('%Y-%m-%d')

        date_list = nav_df.index.tolist()
        week_list = date_list[::5]
        
        def get_month_list(start_date, end_date):
            """获取月末交易日的列表
            """
            trading_dates = get_trade_days(start_date=start_date, end_date=end_date)
            yearmonth = lambda date: (date.year, date.month)
            last_element = lambda iterable: list(iterable)[-1]

            res = [last_element(group) for _, group in itertools.groupby(trading_dates, yearmonth)]
            return res

        def get_year_list(start_date, end_date):
            """获取年初交易日的列表
            """
            trading_dates = get_trade_days(start_date=start_date, end_date=end_date)
            year = lambda date: (date.year)
            last_element = lambda iterable: list(iterable)[0]

            res = [last_element(group) for _, group in itertools.groupby(trading_dates, year)]
            res = res + [trading_dates[-1]]
            return res

        month_list = get_month_list(start_date, end_date)
        year_list = get_year_list(start_date, end_date)

        period_dict = {'日度':date_list, 
                       '周度':week_list, 
                       '月度':month_list,
                       '年度':year_list}
        prob_dict = {}
        ret_dict = {}

        for key,value in period_dict.items():        
            frame_nav = nav_df.loc[value]
            frame_ret = frame_nav.pct_change().dropna()
            frame_ret['hedging'] = frame_ret['strategy'] - frame_ret['benchmark']

            win_prob = 100.0 * len(frame_ret[frame_ret['hedging'] >= 0]) / len(frame_ret)
            prob_dict[key] = str(np.round(win_prob, 2)) + '%'
            ret_dict[key] = frame_ret.applymap(lambda x: np.round(100.0 * x, 2))

        prob_df = pd.Series(data=prob_dict).to_frame('win_prob')
        self.ret_dict = ret_dict
        return prob_df
    
    
    def plot_ret(self, freq='年度'):
        """
        画出策略和基准的收益率
        param:
            freq：数据的频率
        return:
            ret_df：策略和基准的收益率
        """
        ret_dict = self.ret_dict
        fig = plt.figure(figsize=(13,5))
        ax1 = fig.add_subplot(111)

        ret_df = ret_dict[freq].copy()
        ret_df['year'] = ret_df.index
        ret_df['year'][:-1] = [int(dt.strftime("%Y-%m-%d")[:4])-1 for dt in ret_df['year'][:-1]]
        ret_df = ret_df.set_index('year')

        ret_df.plot(kind='bar', ax=ax1, alpha=0.8)
        ax1.set_ylabel(u'return(%)', fontsize=20)
        ax1.set_title(freq+'收益率', fontsize=20)
        return ret_df
    
    def underwater(self, start=None, end=None, 
                 long_only_under_water=False):
        """
        策略收益、超额收益的一定回看期最大回撤幅度、最大回撤区间
        param:
            start：开始时间
            end：结束时间
            long_only_under_water：True代表策略的回撤，False代表画超额收益的回撤
        """
        nav_df = self.nav_excess_df
        cpt_df = nav_df.pct_change().dropna()
        cpt_df.iloc[0] = 0
        cpt_df = cpt_df.loc[start:end]

        df_cum_rets = (cpt_df['excess'] + 1).cumprod()
        cpt_df['excess_cum'] = df_cum_rets - 1
        cpt_df['strategy_cum'] = (cpt_df['strategy'] + 1).cumprod() - 1
        cpt_df['benchmark_cum'] = (cpt_df['benchmark'] + 1).cumprod() - 1

        if long_only_under_water == True:
            df_cum_rets = (cpt_df['strategy'] + 1).cumprod()

        running_max = np.maximum.accumulate(df_cum_rets)
        underwater = -((running_max - df_cum_rets) / running_max)
        underwater.index = cpt_df.index
        
        worst_underwater_end = np.argmin(underwater)
        worst_underwater_begin = np.argmax(running_max[:np.argmin(underwater)])
        worst_underwater = min(underwater)
        
        index = ["最大回撤开始日期",  '最大回撤结束日期', '最大回撤幅度']
        result = [worst_underwater_begin.strftime('%Y年%m月%d日'), worst_underwater_end.strftime('%Y年%m月%d日'), format(worst_underwater, '.2%')]
        result_dict = {'数据':result}
        dataframe = pd.DataFrame(result_dict)
        dataframe.index = index
        
        return dataframe
        
    def plot_hedging(self, start=None, end=None, 
                 long_only_under_water=False, title='hedging & underwater'):
        """
        画出策略，基准，超额收益，超额收益的回撤
        param:
            start：开始时间
            end：结束时间
            long_only_under_water：是否画出策略的回撤，False代表画超额收益的回撤
        """
        nav_df = self.nav_excess_df
        cpt_df = nav_df.pct_change().dropna()
        cpt_df.iloc[0] = 0
        cpt_df = cpt_df.loc[start:end]

        df_cum_rets = (cpt_df['excess'] + 1).cumprod()
        cpt_df['excess_cum'] = df_cum_rets - 1
        cpt_df['strategy_cum'] = (cpt_df['strategy'] + 1).cumprod() - 1
        cpt_df['benchmark_cum'] = (cpt_df['benchmark'] + 1).cumprod() - 1

        if long_only_under_water == True:
            df_cum_rets = (cpt_df['strategy'] + 1).cumprod()

        running_max = np.maximum.accumulate(df_cum_rets)
        underwater = -((running_max - df_cum_rets) / running_max)
        underwater.index = cpt_df.index

        fig = plt.figure(figsize=(15, 6))
        fig.set_tight_layout(True)
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()

        ax1.grid(axis='y', ls='--')
        ax1.fill_between(underwater.index, 0, np.array(underwater), color='grey', alpha=0.3, label='underwater')
        ax2.plot(cpt_df.index, cpt_df['excess_cum'], lw=2, color='orange', label='hedging')
        ax2.plot(cpt_df.index, cpt_df['strategy_cum'], lw=2, color='#00BFFF', label='strategy')
        ax2.plot(cpt_df.index, cpt_df['benchmark_cum'], color='grey', label='benchmark')

        if (start != None) & (end != None):
            ax2.set_title(title+'  ( '+start+' —— '+end+' )', fontsize=16)
        else:
            ax2.set_title(title, fontsize=16)


class portfolio_analysis(object):
    
    # 定义函数中不同的变量
    def __init__(self, portfolio_df=None):
        # portfolio_df：dataframe，持仓股票组合，colums=[date,code]
        self.portfolio_df = portfolio_df
        
    
    def industry_exposure(self, date, index_id='000905.XSHG'):
        """
        相对基准的行业暴露
        param：
            date：日期
            index_id：基准代码
        """
        univ = self.portfolio_df['code'].tolist()
        
        def get_industry_sw(tradeDate_list, univ):
            """
            获取申万一级的行业数据
            param：
                univ：list，持仓股票列表
                tradeDate_list：日期列表
            """
            data_dict = OrderedDict()
            for date in tradeDate_list:
                temp = get_industry(security=univ, date=date)

                temp_dict = OrderedDict()
                for key, value in temp.items():
                    try:
                        temp_dict[key] = temp[key]['sw_l1']['industry_name'][:-1]
                    except:
                        continue
                data_dict[date] = pd.Series(temp_dict)
            data = pd.concat(data_dict, names=['date', 'code']).to_frame('industry')
            return data

        bm_w = get_index_weights(index_id, date=date)
        bm_w = bm_w.reset_index()
        bm_w = bm_w[['date','code','weight']]
        bm_w['date'] = [dt.strftime('%Y-%m-%d') for dt in bm_w['date']]

        bm_univ = bm_w['code'].tolist()
        bm_ind =get_industry_sw([date], bm_univ)
        bm_ind = bm_ind.reset_index()
        bm_ind =bm_ind.merge(bm_w, on=['date','code']).groupby(by='industry').sum() / 100
        bm_ind.columns = ['benchmark_weight']

        target_ind = get_industry_sw([date], univ)
        target_ind = target_ind.merge(self.portfolio_df, on=['date','code'])
        target_ind = target_ind.groupby(by='industry').sum()
        target_ind.columns = ['strategy_weight']

        cum_ind = pd.concat([bm_ind, target_ind], axis=1)
        cum_ind = cum_ind.fillna(0)

        fig = plt.figure(figsize=(16,6))
        ax = fig.add_subplot(111)
        ax = cum_ind.plot(kind='bar', ax=ax)
        ax.legend(loc='best')
        ax.set_xlabel(u'行业名称', fontsize=15)
        ax.set_ylabel(u'权重', fontsize=15)
        ax.grid(False)

        
    def cap_exposure(self, date, index_id='000905.XSHG'):
        """
        相对基准的市值暴露
        param：
            date：日期
            index_id：基准代码
        """
        univ = self.portfolio_df['code'].tolist()

        def get_mkt(univ, date):
            """获取股票市值数据
            """
            q = query(
                valuation.code,
                valuation.market_cap,
                valuation.day
            ).filter(
                valuation.code.in_(univ),
            )
            data = get_fundamentals(q, date=date)
            data = data.rename(columns={'day':'date'})
            return data

        bm_w = get_index_weights(index_id, date=date)
        bm_w = bm_w.reset_index()
        bm_w = bm_w[['date','code','weight']]
        bm_w['date'] = [dt.strftime('%Y-%m-%d') for dt in bm_w['date']]
        bm_w['weight'] = bm_w['weight'] / 100

        bm_univ = bm_w['code'].tolist()
        bm_mkt = get_mkt(bm_univ, date)
        bm_mkt = bm_mkt.merge(bm_w, on=['date','code'])
        bm_mkt['weight'] = bm_mkt['weight'] * bm_mkt['market_cap']

        target_mkt = get_mkt(univ, date)
        target_mkt = target_mkt.merge(self.portfolio_df, on=['date','code'])
        target_mkt['weight'] = target_mkt['weight'] * target_mkt['market_cap']

        d = {'strategy_market_cap': target_mkt['weight'].sum(), 
             'benchmark_market_cap': bm_mkt['weight'].sum()}

        data_df = pd.DataFrame(data=d, index=[date])
        data_df.index.name = 'date'
        
        fig = plt.figure(figsize=(15,10))
        fig.set_tight_layout(True)
        ax = fig.add_subplot(221)
        ax = data_df.plot(kind='bar', ax=ax, colors=['#FF6347','grey'], alpha=0.5)
        ax.legend(loc='best')
        ax.set_xlabel(u'时间', fontsize=15)
        ax.set_ylabel(u'市值（亿）', fontsize=15)
        ax.grid(False)
        ax.set_title('持仓和基准市值', fontsize=15)
        
        ax2 = fig.add_subplot(222)
        ax2 = sns.distplot(target_mkt['market_cap'], kde=True, rug=True, 
                           color='#FF6347', label='strategy_market_cap')
        ax2 = sns.distplot(bm_mkt['market_cap'], kde=True, rug=True, 
                           color='grey', label='benchmark_market_cap')
        ax2.legend()
        ax2.grid(True)
        ax2.set_xlabel("probability", fontsize=15)
        ax2.set_title('持仓和基准市值分布', fontsize=15)
        
        listBins = [0, 100, 300, 500, 700, 1000, 1000000]
        listLabels = ['0_100','100_300','300_500','500_700','700_1000','1000以上']

        target_mkt['fenzu'] = pd.cut(target_mkt['market_cap'], bins=listBins, labels=listLabels, include_lowest=True)
        target_mkt = target_mkt.groupby('fenzu').count()

        ax3 = fig.add_subplot(223)
        ax3 = target_mkt['market_cap'].plot(kind='bar', ax=ax3, color='grey', alpha=0.5)
        ax3.legend(loc='best')
        ax3.set_xlabel(u'市值区间（亿）', fontsize=15)
        ax3.set_ylabel(u'数量', fontsize=15)
        ax3.set_title('持仓市值区间', fontsize=15)
        ax3.grid(False)