from finlab import data
import finlab
import pickle
import os

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