import pandas as pd
import os

def init_encode(csv_l:list):
    """
    {writer:著者番号}
    の組み合わせを作成する。
    csv_lに、train,test含むすべてのデータが格納されたcsvのパスを渡す。
    csvファイルには[writer]カラムがあることが絶対条件
    """
    writer_l = []
    count = 0
    for csv_path in csv_l:
        df = pd.read_csv(csv_path)["writer"]
        writer_l += df.to_list()

    writer_l = list(set(writer_l))
    df_l = []
    for writer_n,writer in enumerate(writer_l):
        df_l.append([writer,writer_n])


    writer_csv = pd.DataFrame(df_l,columns=["writer","writer_n"])
    writer_csv = writer_csv.set_index("writer",drop=True)
    writer_csv.to_csv("writer.csv")



def encode_writer(df:pd.DataFrame,column:str="writer",database_path:str="writer.csv")->pd.DataFrame:
    """
    pd.DataFrameを渡すと、[writer]カラムを参照して、その著者番号である[writer_n]カラムを追加して返してくれる。
    database_pathには[[writer,writer_n],.....]の形式で著者番号が格納されている。
    """
    database_path = os.path.join(os.path.dirname(__file__),database_path)
    df = df.copy()
    database = pd.read_csv(database_path)
    database = database.set_index("writer",drop=True)


    df["writer_n"] = df["writer"].apply(lambda writer:database.at[writer,"writer_n"])
    return df 






if __name__ == "__main__":
    init_encode(["../datas/test.csv","../datas/train.csv"])

    # test
    # df = pd.read_csv("../datas/train.csv")
    # print(encode_writer(df)[["ncode","writer","writer_n"]].head())