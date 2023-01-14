import pandas as pd 
from datetime import datetime
strptime = datetime.strptime

def make_days(df:pd.DataFrame,column:str)->pd.DataFrame:
    """
    日付として
    2021-08-12 08:06:53
    のフォーマットで与えられている情報を特定の日付までの経過日数として数値化する。
    pandas.df型を第一引数に。
    この処理を使うカラムを第二引数に指定する。
    """
    base_date = "2023-1-1"

    if column != "general_firstup":
        raise "specified column name is not date"

    df = df.copy()

    # general_firstupが出版日のカラム名
    # 2021-08-12 08:06:53 (例)
    df["days"] = df[column].apply(lambda x:(strptime(base_date,"%Y-%m-%d") - strptime(x,"%Y-%m-%d %H:%M:%S")).days)
    return df 
