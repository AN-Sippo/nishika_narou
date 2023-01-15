import lightgbm as lgb 
import pandas as pd 
import matplotlib.pyplot as plt 
import re

from utils.make_days import make_days

class numerfical_dataset():
    """
    sklearn.tarin_test_splitを使いたいので、初期化の入力をpd.DataFrameにすることにした。
    自作データセットだとtrain_test_splitなんかどう使えばええんやと思っていたが、先に分割してから自作データセットにぶち込めばええんやと気がついた。
    使うカラムはself.train_coumns_reで(ハードコード)指定している。
    """
    def __init__(self,df:pd.DataFrame):
        self.data = df 
        self.data = make_days(self.data,"general_firstup")

        # カラム名の整理
        self.train_columns_re = re.compile(r"(days|biggenre|genre|novel_type|end|isstop|isr15|isbl|isgl|istenni|istensei|iszankoku|pc_or_k|title_.+|story_.+)")
        # self.train_columns_re = re.compile(r"fav_novel_cnt_bin")
        self.train_columns = ["days","biggenre","genre","novel_type","end","isstop","isr15","isbl","isgl","iszankoku","istensei","istenni","pc_or_k"]
        self.target_column = "fav_novel_cnt_bin"

        self.is_train = self.target_column in self.data.columns 


        # have datas practical
        self.train_data = self.data.filter(regex=self.train_columns_re)
        print(f"picked {len(self.train_data.columns)} columns")

        if self.is_train:
            self.target_data = self.data[[self.target_column]]

        # mean-normilization
        # eps = 1e-8
        # for column in self.train_data:
        #     mean = self.train_data[column].mean()
        #     std = self.train_data[column].std()
        #     self.train_data.loc[:,column] = self.train_data.loc[:,column].apply(lambda x:(x - mean) / (std + eps))

    def __len__(self):
        return len(self.train_data)
    
    def __getitem__(self,idx:int)->dict:
        data = self.train_data.loc[idx,:].values

        if self.is_train:
            label = self.target_data.loc[idx,self.target_column]
            target = label
            # labelのワンホットエンコーディング
            # target = [0,0,0,0,0]
            # target[label]+=1
        else:
            target = None

        ans = {"data":data,"label":target}
        return ans 

    def getall(self):
        """
        return train(list),label(list) で、持っているすべてのデータを返却する。
        """

        # data = []
        # label = []
        # for i in range(len(self)):
        #     data.append(self[i]["data"])
        #     label.append(self[i]["label"])
        # return data,label 

        return self.train_data,self.target_data
            
    def getitems(self):
        """
        (X,y)のリストを返す。
        valid_setに渡す用。
        """
        res = []
        for i in range(len(self)):
            res.append((self[i]["data"],self[i]["label"]))
        return res
