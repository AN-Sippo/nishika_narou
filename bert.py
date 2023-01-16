import torch 
from torch import nn
import pandas as pd 
from  transformers import BertModel
from  transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from torch.utils.data import DataLoader
import numpy as np 
from matplotlib import pyplot as plt 

#自作部分
from datasets_for_bert import dataset_for_bert

model_name = "cl-tohoku/bert-base-japanese"
dataset = dataset_for_bert("datas/train.csv",max_length=15,column="title")


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

            if batch % 1 == 0:
                print(f"finished [{epoch * len_all_data + batch}/{epochs * len_all_data}] loss:{loss.item()}")
            
            loss_y.append(loss.item())
            loss_x.append(epoch * len_all_data + batch)
    
    return loss_x,loss_y 

#---------- max_lengthを決める-------------
text_columns = ["title","story"]
percentile = 0.9
data_for_max_length_path = "datas/train.csv"
columns_to_quent = {}

max_length_l = (pd.read_csv(data_for_max_length_path)[text_columns].applymap(len)).quantile(percentile)
for column_name,quen in zip(text_columns,max_length_l):
    columns_to_quent[column_name] = quen + 2

print(columns_to_quent)
#-----------------------------------------
# lr_l = [0.01]
# batch_size_l = [64]
# epochs_l = [1]

# fig,ax_l = plt.subplots(len(lr_l),len(batch_size_l))

# model = bert_model_for_classification()
# for lr_idx,lr in enumerate(lr_l):
#     for batch_size_idx,batch_size in enumerate(batch_size_l):
#         for epochs in epochs_l:
#             loss_x,loss_y = train_loop(model=model,lr=lr,batch_size=batch_size,epochs=epochs)
#             ax = ax_l[lr_idx][batch_size_idx]
#             ax.plot(loss_x,loss_y)
#             ax.set_label(f"lr={lr} batch_size={batch_size}")
#             ax.set_yticks([0,0.5,1])
#             ax.legend()



