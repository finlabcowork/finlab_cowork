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
from sklearn.linear_model import LinearRegression
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


#eps簡易預測模型------------------------------------------------------------------------------------------------ 

def co_eps_predict_simple(stock_name,month_revenue_pred,gross_profit_ratio_pred):
    "利用線性迴歸直接估算營收與eps"
    #載資料
    close = co_get('price:收盤價') 
    revenue_quarter = co_get('financial_statement:營業收入淨額')
    gross_profit = co_get('financial_statement:營業毛利')
    gross_profit_ratio = co_get('fundamental_features:營業毛利率')
    eps = co_get('financial_statement:每股盈餘')
    monthly_revenue = co_get('monthly_revenue:當月營收')
    
    
    #合併df
    revenue_gross_eps_concat = pd.concat([revenue_quarter[stock_name],gross_profit[stock_name],gross_profit_ratio[stock_name],eps[stock_name]],axis=1)
    revenue_gross_eps_concat.columns = ["revenue_quarter", "gross_profit","gross_profit_ratio", "eps"]

    #基礎資料
    print("目前最新資料為: ",revenue_gross_eps_concat.index[-1],"查詢個股為:",stock_name)
    
    #營收圖表,觀察歷史變動率與是否適合
    coefficient_of_variation = monthly_revenue[stock_name].std()/monthly_revenue[stock_name].mean()
    coefficient_of_variation_TW_mean = (monthly_revenue.dropna(axis=1).std()/monthly_revenue.dropna(axis=1).mean()).mean()
    coefficient_pr = round((1-sum(monthly_revenue.dropna(axis=1).std()/monthly_revenue.dropna(axis=1).mean()>coefficient_of_variation)/monthly_revenue.dropna(axis=1).shape[1])*100,2)
    
    fig = px.line(monthly_revenue[stock_name].reset_index(), x='date', y=stock_name,title = stock_name+"營收變化")
    fig.show()
    
    
    print("營收變異係數為",round(coefficient_of_variation,3))
    print("台股營收平均變異係數為",round(coefficient_of_variation_TW_mean,3))
    print("此股票營收波動度大於"+str(coefficient_pr)+"%台灣股票")
    print("過去四個月,"+'平均營收為: '+str((revenue_gross_eps_concat['revenue_quarter'][-1]/4/100000))+"億元")
    print("過去四個月,"+'毛利為: ',str(round(revenue_gross_eps_concat['gross_profit_ratio'][-1],2)),"%")
    print("-------------------------------------------------------------------------------------------------------------")

    #營收與毛利迴歸,採用最近4年
    "雖然固定成本理當算在營業費用中,但實際觀察各家企業,還是能後發現,營業成本還是有許多固定金額,因此透\
    過線性迴歸抓出[固定金額]以及[在當前營收水準下每多增加1元營收,能夠轉變成多少營業毛利(邊際毛利)]"


    #營收與毛利圖
    fig = px.scatter(revenue_gross_eps_concat[-16:], x="revenue_quarter", y="gross_profit",\
                     title=stock_name+"過去4年營收與毛利圖,呈現的愈直表示兩者關聯度高,代表用營收來推估毛利會愈準確",\
                    color = "gross_profit",trendline="lowess")
    fig.show()
    #營收與毛利統計
    revenue_quarter_4y = pd.DataFrame(revenue_gross_eps_concat['revenue_quarter'][-16:])
    gross_profit_4y = pd.DataFrame(revenue_gross_eps_concat['gross_profit'][-16:])

    lm1 = LinearRegression()
    lm1.fit(revenue_quarter_4y, gross_profit_4y)

    print('營收對毛利迴歸係數:',lm1.coef_[0][0],",表示邊際毛利率為:"+str(round(lm1.coef_[0][0]*100,1))+"%") 
    print('營收對毛利迴歸截距項:',lm1.intercept_[0],",表示營業成本(月)的固定金額為:"+str(round(-lm1.intercept_[0]/4/100000,1))+"億元")
    print('Pearson套件相關係數:'+str(round(revenue_quarter_4y["revenue_quarter"].corr(gross_profit_4y["gross_profit"],method='pearson'),2)))
    print("-------------------------------------------------------------------------------------------------------------------------------")
    #營收與eps迴歸,採用最近4年
    #營收與eps圖
    revenue_gross_eps_concat = pd.concat([revenue_quarter[stock_name],gross_profit[stock_name],gross_profit_ratio[stock_name],eps[stock_name]],axis=1)
    revenue_gross_eps_concat.columns = ["revenue_quarter", "gross_profit","gross_profit_ratio", "eps"]
    revenue_gross_eps_concat

    fig = px.scatter(revenue_gross_eps_concat[-16:], x="revenue_quarter", y="eps",\
                     title=stock_name+"過去4年營收與eps圖,呈現的愈直表示兩者關聯度高,代表用營收來推估eps會愈準確",\
                    color = "eps",trendline="lowess")
    fig.show()

    #營收與eps統計
    revenue_quarter_4y = pd.DataFrame(revenue_gross_eps_concat["revenue_quarter"][-16:])
    eps_4y =  pd.DataFrame(revenue_gross_eps_concat["eps"][-16:])
    
    lm2 = LinearRegression()
    lm2.fit(revenue_quarter_4y, eps_4y)

    print('營收對eps迴歸係數:',lm2.coef_[0][0],",表示月營收提升1百萬,月eps提升"+str(round(lm2.coef_[0][0]/4*1000000,3))+"元,","換算年eps:"+str(round(lm2.coef_[0][0]/4*1000000*12,3)),"元") 
    print('Pearson套件相關係數:'+str(round(revenue_quarter_4y["revenue_quarter"].corr(eps_4y["eps"],method='pearson'),2)))
    print("-------------------------------------------------------------------------------------------------------------------------------")

    #毛利與eps迴歸,採用最近4年
    #毛利率與eps圖
    fig = px.scatter(revenue_gross_eps_concat[-16:], x="gross_profit_ratio", y="eps",\
                     title=stock_name+"過去4年毛利率與eps圖,呈現的愈直表示兩者關聯度高,代表用毛利率來推估eps會愈準確",\
                    color = "eps",trendline="lowess")
    fig.show()

    #毛利率與eps圖統計
    gross_profit_ratio_4y = pd.DataFrame(revenue_gross_eps_concat["gross_profit_ratio"][-16:])
    eps_4y =  pd.DataFrame(revenue_gross_eps_concat["eps"][-16:])

    lm3 = LinearRegression()
    lm3.fit(gross_profit_ratio_4y, eps_4y)

    print('毛利對eps迴歸係數:',lm3.coef_[0][0],",表示毛利提升1%,季eps提升"+str(round(lm3.coef_[0][0],3))+"元","換算年eps:"+str(round(lm3.coef_[0][0]*4,3))+"元") 
    print('Pearson套件相關係數:'+str(round(gross_profit_ratio_4y["gross_profit_ratio"].corr(eps_4y["eps"],method='pearson'),2)))
    print("-------------------------------------------------------------------------------------------------------------------------------")
    
    #營收+毛利率與eps圖
    revenue_quarter_4y = pd.DataFrame(revenue_gross_eps_concat['revenue_quarter'][-16:])
    gross_profit_ratio_4y = pd.DataFrame(revenue_gross_eps_concat["gross_profit_ratio"][-16:])
    eps_4y =  pd.DataFrame(revenue_gross_eps_concat["eps"][-16:])
    concat_revenue_gross = pd.concat([revenue_quarter_4y,gross_profit_ratio_4y],axis=1)
    concat_revenue_gross_eps = pd.concat([revenue_quarter_4y,gross_profit_ratio_4y,eps_4y],axis=1)
    
    fig = px.scatter_3d(concat_revenue_gross_eps, x='revenue_quarter', y='gross_profit_ratio', z='eps',
                        color='eps')
    fig.show()
    
    #營收+毛利率與eps,複迴歸
    lm4 = LinearRegression()
    lm4.fit(concat_revenue_gross, eps_4y)
    
    #複迴歸預測
    eps_pred_quarterly = lm4.predict(np.array([[month_revenue_pred*4*1000,gross_profit_ratio_pred]]))[0][0]#季eps
    eps_pred_year = eps_pred_quarterly*4
    eps_pred_month = eps_pred_year/12
    pe_ratio = close[stock_name][-1]/eps_pred_year
    eps_growth_ratio_quarterly =  eps_pred_quarterly/eps_4y[-1:]["eps"][0]-1
    
    pred_df = pd.DataFrame([coefficient_of_variation,coefficient_of_variation_TW_mean,coefficient_pr,eps_pred_month,eps_pred_quarterly,eps_pred_year,pe_ratio,round(eps_growth_ratio_quarterly*100,2)]).transpose()
    pred_df.columns = ["營收變異係數","台股營收平均變異係數","營收變異係數pr值(該公司營收變異大於多少%公司)","預估月eps","預估季eps", "預估年eps","預估本益比","預估季eps成長率(%)"]
    return pred_df
