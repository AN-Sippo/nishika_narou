import pandas as pd 
import torch
import datetime
from torch.utils.data import Dataset,DataLoader
from torch import nn 
import matplotlib.pyplot as plt 
import re
strptime = datetime.datetime.strptime 

mode = "train"

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

class numerfical_dataset(Dataset):
    """
    使いたいデータが入ったcsvファイルのパスを指定すれば、Dataset[idx]で、{x:torch.tensor,y:torch.tensor}を返す。
    使うカラムはself.train_coumns_reで指定している。
    """
    def __init__(self,csv_path:str):
        self.data = pd.read_csv(csv_path)
        self.data = make_days(self.data,"general_firstup")

        # カラム名の整理
        self.train_columns_re = re.compile(r"(days|biggenre|genre|novel_type|end|isstop|isr15|isbl|isgl|istenni|istensei|iszankoku|pc_or_k|title_\d+|story_\d+)")
        self.train_columns = ["days","biggenre","genre","novel_type","end","isstop","isr15","isbl","isgl","iszankoku","istensei","istenni","pc_or_k"]
        self.target_column = "fav_novel_cnt_bin"

        self.is_train = self.target_column in self.data.columns 


        # have datas practical
        self.train_data = self.data.filter(regex=self.train_columns_re)
        print(f"picked {len(self.train_data)} columns")

        if self.is_train:
            self.target_data = self.data[[self.target_column]]

        # mean-normilization
        eps = 1e-8
        for column in self.train_data:
            mean = self.train_data[column].mean()
            std = self.train_data[column].std()
            self.train_data.loc[:,column] = self.train_data.loc[:,column].apply(lambda x:(x - mean) / (std + eps))

    def __len__(self):
        return len(self.train_data)
    
    def __getitem__(self,idx:int)->dict:
        data = self.train_data.loc[idx,:]
        data = torch.Tensor(data)

        if self.is_train:
            label = self.target_data.loc[idx,self.target_column]
            target = [0,0,0,0,0]
            target[label]+=1
            target = torch.Tensor(target)
        else:
            target = None

        ans = {"data":data,"label":target}
        return ans 


class numerfical_model(torch.nn.Module):
    def __init__(self):
        super(numerfical_model,self).__init__() 
        self.linear_stack = nn.Sequential(
            nn.Linear(413,100),
            nn.BatchNorm1d(100),
            nn.Sigmoid(),
            nn.Linear(100,10),
            nn.BatchNorm1d(10),
            nn.Sigmoid(),
            nn.Linear(10,5),
            nn.Softmax(dim=1)
        )
    
    def forward(self,x):
        logits = self.linear_stack(x)
        return logits 


def train_loop(train_dataloader:DataLoader,loss_fn:torch.nn,optimizer:torch.optim.Optimizer):
    # const
    full_size = len(train_dataloader.dataset)
    batch_size = train_dataloader.batch_size
    len_ = len(train_dataloader)

    global loss_y,loss_x

    for i,data in enumerate(train_dataloader):
        #forward
        x = data["data"]
        y = data["label"]

        pred = model(x)
        loss = loss_fn(pred,y)

        #backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss,current = loss.item(),i * batch_size + epoch * full_size
        loss_y.append(loss)
        loss_x.append(current)
        if i % 200 == 0:
            print(f'loss:{loss:>7f} [{current:>5d}/{full_size * epochs}]')

if mode == "train":
    #--------hyper parameters----------------
    # lr_l = [0.001]
    lr_l = [0.001,0.003,0.01,0.03,0.1,0.3]
    epochs_l = [3]
    mini_batch_size_l = [64]
    #----------------------------------------

    training_data = numerfical_dataset('with_keyword_train.csv')
    model = numerfical_model()
    loss_fn = torch.nn.CrossEntropyLoss()

    ax1 = plt.subplot(3,3,1)
    ax2 = plt.subplot(3,3,2)
    ax3 = plt.subplot(3,3,3)
    ax4 = plt.subplot(3,3,4)
    ax5 = plt.subplot(3,3,5)
    ax6 = plt.subplot(3,3,6)
    ax7 = plt.subplot(3,3,7)
    ax_l = [ax1,ax2,ax3,ax4,ax5,ax6,ax7]

    count = 0
    for lr in lr_l:
        for epochs in epochs_l:
            for mini_batch_size in mini_batch_size_l:
                train_dataloader = DataLoader(training_data,batch_size=mini_batch_size,shuffle=True)
                optimzer = torch.optim.Adam(model.parameters(),lr=lr)

                loss_y = []
                loss_x = []
                for epoch in range(epochs):
                    train_loop(train_dataloader,loss_fn=loss_fn,optimizer=optimzer)
                ax_l[count].plot(loss_x,loss_y,label = f"lr={lr} batch_size={mini_batch_size} epochs={epochs}")
                ax_l[count].legend()
                count += 1

    plt.show()
    torch.save(model,"numerfical_model.pth")

if mode == "predict":
    # load a model
    model = torch.load("numerfical_model.pth")
    model.eval()
    
    test_dataset = numerfical_dataset("test.csv")
    raw_test_dataset = pd.read_csv("test.csv")

    # make a dataset for submission
    out_df = pd.DataFrame()
    out_df["ncode"] = raw_test_dataset.loc[:,"ncode"]
    out_df["proba_0"] = 0
    out_df["proba_1"] = 0
    out_df["proba_2"] = 0
    out_df["proba_3"] = 0
    out_df["proba_4"] = 0

    # predict
    for idx in range(len(test_dataset)):
        data = test_dataset[idx]
        x = data["data"]
        x = x.unsqueeze(0)

        # print(x.shape)
        model.eval()
        logits = model(x)
        logits = logits.detach().numpy().tolist()[0]
        out_df.loc[idx,["proba_0","proba_1","proba_2","proba_3","proba_4"]] = logits
    

    out_df = out_df.set_index("ncode")
    out_df.to_csv("out.csv")

