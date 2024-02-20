from finlab import data
import finlab
import pickle
import os
import plotly.express as px
from finlab.backtest import sim
from finlab.tools.event_study import create_factor_data
import tqdm
import numpy as np 
import pandas as pd
from finlab.dataframe import FinlabDataFrame
import cufflinks as cf

path = "D:/trade_data/finlab_db_m/"


"""
程式碼傷眼滲入
"""

#資料加載------------------------------------------------------------------------------------------------          

def often_use_list():
    data_set_list = ["price:收盤價",\
                 "dividend_tse:最近一次申報每股 (單位)淨值",\
                 "price_earning_ratio:股價淨值比",\
                 'fundamental_features:營業利益率',\
                 "monthly_revenue:當月營收",\
                 "monthly_revenue:上月營收",\
                 "monthly_revenue:去年當月營收",\
                 "financial_statement:每股盈餘",\
                 "price:成交股數",\
                 "financial_statement:股本",\
                 "etl:market_value",\
                 "financial_statement:投資活動之淨現金流入_流出",\
                 "financial_statement:營業活動之淨現金流入_流出",\
                 "fundamental_features:經常稅後淨利",\
                 "financial_statement:股東權益總額",\
                 "fundamental_features:營業利益成長率",\
                 "financial_statement:研究發展費",\
                 
                ]
    return data_set_list

def load_cloud_save(path:str, dataset:str) :
    dataset_df = data.get(dataset)
    data_set_name_re = dataset.replace(":","_")
    manual_path =path+data_set_name_re+".pickle"
    dataset_df.to_pickle(manual_path)
        
        
        
        
        
def load_prem(path:str ,dataset:str) :
    data_set_name_re = dataset.replace(":","_")
    manual_path =path+data_set_name_re+".pickle"
    dataset_df  = pickle.load(open(manual_path, 'rb'))
    return dataset_df





def co_get(dataset):
    if not os.path.isdir(path):
        raise OSError("資料夾路徑錯誤")
    try:
        dataset_df = load_prem(path,dataset)
        print("從地端載入: "+"\""+dataset+"\"")
        return dataset_df
    except :
        print("地端資料庫未發現資料:"+"\""+dataset+"\""+",改為從finlab下載...")
        load_cloud_save(path,dataset)
        load_prem(path,dataset)
        
        
        
#因子分析------------------------------------------------------------------------------------------------              
        
def co_event_analysis_real_trade(buy,p=0.07):
    """
    利用回測軟體+機率隨機賣出,模擬事件發生後的報酬率
    機率隨機賣出:每一天隨機賣掉的機率為p = 0.07
    """
    buy = FinlabDataFrame(buy)
    random_sell = pd.DataFrame(np.random.choice(a=[False, True], p=[1-p, p],size=(buy.shape[0], buy.shape[1]))  , columns=buy.columns,index=buy.index)
    random_sell

    # 部位
    position = buy.hold_until(FinlabDataFrame(random_sell))
    report = sim(position , resample="D", upload=False, position_limit=1/3, fee_ratio=0,tax_ratio=0, trade_at_price='open')
    report_trades = report.get_trades()
    #去除端值(持有區間最高30天)
    report_trades_clean = report_trades.drop(report_trades.loc[report_trades["period"]>30].index)
    # 計算平均
    report_trades_groupby = report_trades_clean.groupby("period").mean().reset_index()

    ##繪圖
    fig = px.scatter(report_trades_groupby, x="period", y="return",color='return',trendline="lowess")
    fig.show()
    
    
    
    
    
def co_event_analysis(buy:"dataframe"):
    """
    用finlab原始程式碼改的,用以分析事件發生前後之報酬率變化
    注意:
    
    1.記憶體量不夠可能會錯誤
    2.cross over
    
    參考:
    https://www.finlab.tw/event-study-usage/
    https://doc.finlab.tw/reference/tools/
    """
    adj_close = co_get('etl:adj_close')
    factor_data = create_factor_data(buy, adj_close, event=buy)
    buy_time_distribution = pd.DataFrame(buy.sum(axis=1)).reset_index() 
    buy_time_distribution.rename(columns = {0:'number of times'}, inplace = True)
    buy_time_distribution
    fig1 = px.area(buy_time_distribution, x="date", y="number of times",color="number of times",
                 title="事件發生次數與日期分布")
    fig1.show()
    
    #用加權指數當成benchmark,排除加權指數時間變因
    benchmark = co_get('benchmark_return:發行量加權股價報酬指數')
    benchmark_pct = benchmark.reindex(adj_close.index, method='ffill').pct_change()
    stock_pct = adj_close.pct_change()
    def get_period(df, date, sample):
        i = df.index.get_loc(date)
        return df.iloc[i+sample[0]: i+sample[1]].values
    
    #轉換成,獨立事件與時間報酬率
    ret = []
    sample_period=(-40, -20) #
    estimation_period=(-15, 30)# 觀察事件前15日與後30日變化
    for date, sid in tqdm.tqdm(factor_data.index):

        X1, Y1 = get_period(benchmark_pct, date, sample_period)[:,0], \
            get_period(stock_pct[sid], date, sample_period)
        X2, Y2 = get_period(benchmark_pct, date, estimation_period)[:,0], \
            get_period(stock_pct[sid], date, estimation_period)

        # Run CAPM
        cov_matrix = np.cov(Y1, X1)
        beta = cov_matrix[0, 1] / cov_matrix[1, 1]
        AR = np.array(Y2) - beta * X2
        ret.append(AR)
    #計算事件發生日前後的日報酬率變化
    ret = pd.DataFrame(ret, columns=range(*estimation_period))
    ret_df = pd.DataFrame(ret.mul(100).mean()).reset_index() 
    ret_df_re = ret_df.rename(columns = {"index":"days",0:"return"})
    ret_df_re
    fig2 = px.bar(ret_df_re, x="days", y="return",color="return",
                 title="事件發生日前後的日報酬率變化")
    # fig.add_trace(go.Scatter(
    #     x=list(ret_df_re["days"]),
    #     y=list(ret_df_re["return"]),
    #     xperiod="M1",
    #     xperiodalignment="middle",
    #     hovertemplate="%{y}%{_xother}"
    # ))
    fig2.show()
    
    #計算累計報酬率,並將事件發生日作基準點
    accum_ret_df = pd.DataFrame(ret.mul(100).cumsum(axis=1).mean()).reset_index() 
    accum_ret_df_re = accum_ret_df.rename(columns = {"index":"days",0:"return"})
    accum_ret_df_re["return_accumulated"] = accum_ret_df_re["return"] -accum_ret_df_re.at[15,"return"]
    std = ret.mul(100).cumsum(axis=1).std() * 0.1
    accum_ret_df_re
    fig3 = px.line(accum_ret_df_re, x="days", y="return_accumulated",
             title="累計報酬率,以事件發生日作基準點")

    fig3.show()
    return accum_ret_df_re

#交易紀錄視覺化------------------------------------------------------------------------------------------------ 
"""
#步驟
1.拿全部kbar,資料量大,只加載1次,分開做
2.拿取總交易紀錄,並觀察要查詢哪一筆
3.選取指定股票的kbar
4.選取指定股票交易紀錄
5.印出指定的股票所有交易紀錄(指定股票可能成交不只一次)


目前無法出場的交易紀錄會抱錯,待處理

"""


def co_get_all_kbar():
    """
    把調整後的每日open,high,low,close抓下來,組成kbar df
    """
    adj_open = co_get('etl:adj_open')
    adj_high = co_get('etl:adj_high')
    adj_low = co_get('etl:adj_low')
    adj_close = co_get('etl:adj_close')
    vol = co_get('price:成交股數')/1000
    kbar_df= pd.concat([adj_open,adj_high,adj_low,adj_close,vol],axis=1)
    return kbar_df

def co_get_stock_kbar(stock_name,kbar_df):
    stock_kbar = kbar_df[stock_name] # 選出指定的股票
    stock_kbar.columns = ["open", "high", "low","close","volume"] # 調整欄位名稱至cf所需
    stock_kbar = stock_kbar.dropna() # 若有NAN,則去除該列
    return stock_kbar

def get_stock_trades(stock_name,all_trades):
    stock_trades = all_trades[all_trades["stock_id"].str.contains(stock_name)]
    return stock_trades

# def co_get_stock_kbar_range(stock_trade,day_range=100):
#     #抓出進出場時間
#     entry_date = stock_trades.iat[0,1]
#     exit_date = stock_trades.iat[0,2]
#     #進出場區間往外擴張day_range = 300天
   
#     entry_date_range = entry_date-pd.Timedelta(days = day_range)
#     exit_date_range = entry_date+pd.Timedelta(days = day_range)
#     #節選出想要印出的股票區間

#     stock_kbar_range = pd.DataFrame(stock_kbar[entry_date_range.strftime('%Y-%m-%d'):exit_date_range.strftime('%Y-%m-%d')])
#     return stock_kbar_range, entry_date.strftime('%Y-%m-%d') , exit_date.strftime('%Y-%m-%d')


def co_plot_trade_fig(stock_name,stock_trades,stock_kbar,day_range):
    """
    把該股票交易紀錄依序印出
    """
    cf.set_config_file(theme='pearl',sharing='public',offline=True)# 設定cf
    for i in range(stock_trades.shape[0]):
        stock_trade = stock_trades.iloc[i:i+1]
        entry_date = stock_trade.iat[0,1]
        exit_date = stock_trade.iat[0,2]

        #進出場區間往左擴張day_range,往右擴張day_range/3(進場前的資訊比較重要)
        entry_date_range = entry_date-pd.Timedelta(days = day_range)
        exit_date_range = exit_date+pd.Timedelta(days = day_range/3)

        #節選出想要印出的股票區間
        stock_kbar_range = pd.DataFrame(stock_kbar[entry_date_range.strftime('%Y-%m-%d'):exit_date_range.strftime('%Y-%m-%d')])

        #將entry_date與exit_date從timestamp改成str以符合cf格式
        entry_date = entry_date.strftime('%Y-%m-%d')
        exit_date = exit_date.strftime('%Y-%m-%d')

        #畫圖
        qf=cf.QuantFig(stock_kbar_range,title=stock_name,legend='top',name='kbar')
        qf.add_volume()
        qf.add_annotations( {'x': entry_date,'text': 'buy'})
        qf.add_annotations( {'x': exit_date,'text': 'sell'} )
        qf.add_ema()
        qf.add_trendline(entry_date,exit_date,on='close')
        qf.iplot()

def co_trade_visual(stock_name,all_trades,kbar_df,day_range):
    stock_kbar = co_get_stock_kbar(stock_name,kbar_df) #3.選取指定股票的kbar
    stock_trades = get_stock_trades(stock_name,all_trades) #4.選取指定股票交易紀錄
    co_plot_trade_fig(stock_name,stock_trades,stock_kbar,day_range) #5. 印出指定的股票所有交易紀錄

