import torch 
from torch import nn
import pandas as pd 
from  transformers import BertModel
from  transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from torch.utils.data import DataLoader
import numpy as np 
from matplotlib import pyplot as plt 
from copy import deepcopy

#自作部分
from datasets_for_bert import dataset_for_bert

model_name = "cl-tohoku/bert-base-japanese"
text_columns_l = ["title","story"]

mode = "predict"

class bert_model_for_classification(nn.Module):
    def __init__(self,n_classes:int=5,model_name:str="cl-tohoku/bert-base-japanese",):
        super(bert_model_for_classification,self).__init__()
        
        self.bert_model:nn.Module = BertModel.from_pretrained(model_name)
        self.dense:nn.Module = nn.Linear(768,5)
        self.n_classes = n_classes
    
    def forward(self,X:torch.Tensor)->torch.Tensor:
         x1:BaseModelOutputWithPoolingAndCrossAttentions = self.bert_model.forward(**X)
         x2:torch.FloatTensor = x1.pooler_output
         x3 = self.dense(x2)
        #  z3 = nn.Softmax(dim=1)(x3) crossentropylossではやらなくていいらしい
         return x3

    #今回は、5クラス分類でBERTをfine-tuneして、BERTの予測値のみ(クラス分類はしてないもの)を最終的に特徴量として使うので、
    #Linearレイヤを通していないアウトプットを取得するインターフェースが必要
    def predict(self,X:torch.Tensor,return_class=False):
        x = self.bert_model.forward(X)
        return x 


#hyper parameter はtrain_loopに渡して調整できるようにしておく。
def train_loop(model:bert_model_for_classification,lr:int=0.01,batch_size=64,epochs=1):
    loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    dataloader:DataLoader = DataLoader(dataset=dataset,batch_size=batch_size)

    len_all_data = len(dataloader)
    loss_y = []
    loss_x = []

    for epoch in range(epochs):
        for batch,data in enumerate(dataloader):
            X = {"input_ids":data["input_ids"],"attention_mask":data["attention_mask"],"token_type_ids":data["token_type_ids"]}
            y = data["label"]

            pred = model.forward(X)
            loss = loss_fn(pred,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                print(f"finished [{epoch * len_all_data + batch}/{epochs * len_all_data}] loss:{loss.item()}")
            
            loss_y.append(loss.item())
            loss_x.append(epoch * len_all_data + batch)
    
    return loss_x,loss_y 

#---------- max_lengthを決める-------------
percentile = 0.9
data_for_max_length_path = "datas/train.csv"
columns_to_quent = {}

max_length_l = (pd.read_csv(data_for_max_length_path)[text_columns_l].applymap(len)).quantile(percentile)
for column_name,quen in zip(text_columns_l,max_length_l):
    columns_to_quent[column_name] = int(quen) + 2

#-----------------------------------------

if mode == "train":
    lr_l = [0.01,0.003] #0.003の方が良さそう。0.001も要検討。
    batch_size_l = [64]
    epochs_l = [1]

    save_models = {}
    for text_column in text_columns_l:
        save_models[text_column] = None

    for text_column in text_columns_l:
        dataset = dataset_for_bert("datas/train.csv",max_length=columns_to_quent[text_column],column=text_column)

        fig,ax_l = plt.subplots(len(lr_l),len(batch_size_l))

        model = bert_model_for_classification()

        min_loss = 10000
        for lr_idx,lr in enumerate(lr_l):
            for batch_size_idx,batch_size in enumerate(batch_size_l):
                for epochs in epochs_l:
                    loss_x,loss_y = train_loop(model=model,lr=lr,batch_size=batch_size,epochs=epochs)
                    ax = ax_l[lr_idx]
                    ax.plot(loss_x,loss_y)
                    ax.set_label(f"lr={lr} batch_size={batch_size}")
                    ax.set_yticks([0,0.5,1])
                    ax.legend()
                    if min_loss > loss_y[-1]:
                        temp_model = model 
                        min_loss = loss_y[-1]
        fig.show()
        torch.save(temp_model,f"models/bert{text_column}.pth")

if mode == "predict":
    with torch.no_grad():
        bert_output_size:int = 768
        output_path:str = "datas/bert.csv"
        df = pd.read_csv("datas/test.csv")

        # bertカラムの初期化
        for i in range(bert_output_size):
            df[f"bert_title{i}"] = 0 
        for i in range(bert_output_size):
            df[f"bert_story{i}"] = 0 

        # modelに突っ込むためのデータ整形
        test_datas_n = 8522 #test.csvが持ってるrowの数(2,4261)
        batch_size = 1000

        for text_column in text_columns_l:
            # model:bert_model_for_classification = torch.load(f"models/bert{text_column}.pth")
            model:bert_model_for_classification = bert_model_for_classification()
            model.eval()
            dataset = dataset_for_bert("datas/test.csv",max_length=columns_to_quent[text_column],column=text_column,n_class=5)
            dataloader = DataLoader(dataset,shuffle=False,batch_size=batch_size)
            for batch,data in enumerate(dataloader):
                data.pop("label")
                # training終わってから、predictメソッドがreturnしてないことに気がついた。
                # 再学習は時間がかかりすぎるので、とりあえず直接内部のbert呼び出す。
                res:torch.Tensor = model.bert_model.forward(**data).pooler_output
                df.loc[batch*batch_size:min(test_datas_n,batch_size*(batch+1)-1),[f"bert_{text_column}{i}" for i in range(bert_output_size)]] = res.detach().numpy()
                print(f"finished [{min(test_datas_n,batch_size*(batch+1))}/{test_datas_n}]")
        df.set_index("ncode",drop=True)
        print(df.head())
        df.to_csv(output_path)