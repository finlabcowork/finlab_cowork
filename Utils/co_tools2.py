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
from datetime import datetime
# import df_type

# db_path = "/home/sb0487/trade/finlab/finlab_db" #資料儲存路徑



"""
程式碼傷眼滲入
"""


"""
改動:
    path由外部輸入
    判斷資料是否為最新,若非,則自動更新
    spark與 cudf 語法都不太一樣,目前先用findf即可
"""
class Codata:


    def __init__(self, db_path,auto_update = False,df_type = "findf"):
        print("初始化...")
        self.db_path = db_path
        self.auto_update = auto_update
        self.df_type = df_type
        self.day_update_d = self.get_update_date_d()
        self.day_update_m = self.get_update_date_m()
        self.day_update_q = self.get_update_date_q()
        
        print(f"當前日資料最新日期為:{self.day_update_d}")
        print(f"當前月資料最新日期為:{self.day_update_m}")
        print(f"當前季資料最新日期為:{self.day_update_q}")

    def get_update_date_d(self):
        day_update_d = data.get("price:收盤價").index[-1]
        return day_update_d
        
    def get_update_date_m(self):
        day_update_m = data.get("monthly_revenue:當月營收").index[-1]
        return day_update_m
        
    def get_update_date_q(self):
        day_update_q = data.get("financial_statement:營業成本").index[-1]
        return day_update_q
        
    def get_file_path(self,file_name): 
        return os.path.join(self.db_path, file_name.replace(":", "_") + ".pickle")

    def save_file(self,file_df,file_name) :
        file_df.to_pickle(self.get_file_path(file_name))
    
    def load_local(self,file_name): #從地端讀取檔案
        #不把df型態判別寫在這邊理由是:不同df種類自動更新日期判別語法不同,且pickle只有pd能讀,其他也是要轉換
        with open(self.get_file_path(file_name), 'rb') as file:
            file_df = pickle.load(file)
        return file_df
    def ouput_type_df(self,file_df):
        if self.df_type == "findf":
            type_df = file_df
        elif self.df_type == "cudf":
            import cudf
            type_df = df_type.DataFrame(file_df)
        elif self.df_type == "sparkdf":
            from pyspark.sql import SparkSession
            spark = SparkSession.builder.appName("Pandas to Spark DataFrame").getOrCreate()
            type_df = spark.createDataFrame(file_df)
        return type_df
        

    def check_update(self,file_df,file_name): #檢查更新日期
        if type(file_df.index[-1]) == int:
            print("自動更新只支援日,月,季資料")
            file_update_df = file_df
        elif file_df.index[-1] == self.day_update_d or file_df.index[-1] ==  self.day_update_q:
            print(f"目前資料已是最新:{file_df.index[-1]}")
            file_update_df = file_df

        elif file_df.index[-1] == self.day_update_m and ((file_df.index[-1]-file_df.index[1]).days)/len(file_df)>3:
            print(f"目前資料已是最新:{file_df.index[-1]}")
            file_update_df = file_df
        else :
            file_df = data.get(file_name)
            self.save_file(file_df,file_name)
            file_update_df = self.load_local(file_name)
            print(f"資料更新至{file_update_df.index[-1]}")
        return file_update_df
        
    # @classmethod
    def get(self, file_name):
        if not os.path.isdir(self.db_path):
            raise OSError("資料夾路徑錯誤")
        try:
            file_df = self.load_local(file_name)
            print(f"從地端載入: \"{file_name}\"")
            if self.auto_update:
                file_update_df = self.check_update(file_df,file_name)
            #選擇df輸出型態
            type_df = self.ouput_type_df(file_update_df)
            return type_df
            
        except FileNotFoundError as e:
            print(f"地端資料庫未發現資料: \"{file_name}\", 改為從finlab下載...")
            # print(f"錯誤訊息: {e}")
            file_df = data.get(file_name)
            self.save_file(file_df,file_name)
            file_df = self.load_local(file_name)
            
            #選擇df輸出型態
            type_df = self.ouput_type_df(file_df)
            return type_df
