from finlab import data
import finlab
import pickle
import os
import plotly.express as px
from finlab.backtest import sim


path = "D:/trade_data/finlab_db_m/"


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
        
        
def co_event_analysis_simple(buy,vol):
    """
    正確做法應該是要追蹤符合條件的股票未來每日的報酬率變化,但較為麻煩且耗費效能
    改為用買入之後,使用finlab的回測套件隨機出場,並繪出持股持時間與報酬分布圖,僅限於樣本數較多之事件
    """
    sell = (vol%23 ==0)
    position = buy.hold_until(sell)
    report = sim(position , resample="D", upload=False, position_limit=1/3, fee_ratio=0,tax_ratio=0, trade_at_price='open')
    report_trades = report.get_trades()
    #去除端值(持有區間最高30天)
    report_trades_clean = report_trades.drop(report_trades.loc[report_trades["period"]>30].index)
    # 計算平均
    report_trades_groupby = report_trades_clean.groupby("period").mean().reset_index()

    ##繪圖
    fig = px.scatter(report_trades_groupby, x="period", y="return",color='return',trendline="lowess")
    fig.show()
    
def co_event_analysis(buy,vol):