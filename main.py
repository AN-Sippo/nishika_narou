import pandas as pd 
from  transformers import BertModel,BertTokenizer,BatchEncoding
from torch.utils.data import Dataset,DataLoader
import torch 
from torch import nn 
from torchvision import transforms 
import numpy as np 
from matplotlib import pyplot as plt 

#自作部分
from dataset import text_dataset_tokenize_first

model_name = "cl-tohoku/bert-large-japanese"
device = 'cuda:0' if torch.cuda.is_available() else "cpu"

tokenizer = BertTokenizer.from_pretrained(model_name)

#今は使ってない
class text_dataset(Dataset):
  def __init__(self,path:str):
    self.path = path 
    self.datas = pd.read_csv(self.path)


    # transforms
    self.transform = None
    self.target_transform = None

    #tokenizeしてから格納 map{str:torch.tensor}
   
    # self.titles = self.titles.to(device)
    # self.stories = self.stories.to(device)

    #labelsはtensor化してから格納
    self.labels = torch.tensor(self.datas["fav_novel_cnt_bin"].tolist())
    self.labels = self.labels.to(device)


  def _transform(self,x:str)->BatchEncoding:
    res =  tokenizer(x,truncation='longest_first',padding="max_length",max_length=128,return_tensors="pt")
    return res 

  def _target_transform(self,y:int)->torch.Tensor:
    res = torch.tensor(y,dtype=torch.float16).to(device)
    return res

  def __len__(self):
    return len(self.labels)

  def __getitem__(self,idx):
    # DataLoaderに渡すため、__get_item__が返すmapobjectのvalueはtorch.tensorであるべき？
    token_titles = tokenizer(self.datas["title"].tolist(),truncation="longest_first",padding="max_length",max_length=128,return_tensors="pt")
    token_stories = tokenizer(self.datas["story"].tolist(),truncation="longest_first",padding="max_length",max_length=128,return_tensors="pt")

    title_input_ids = token_titles["input_ids"][0]
    title_attention_mask = token_titles["attention_mask"][0]
    title_token_type_ids = token_titles["token_type_ids"][0]

    story_input_ids = token_stories["input_ids"][0]
    story_attention_mask = token_stories["attention_mask"][0]
    story_token_type_ids = token_stories["token_type_ids"][0]

    label = self.labels[idx]

    if self.transform:
      title = self.transform(title)
      story = self.transform(story)

    if self.target_transform:
      label = self.target_transform(label)

    return {"title_input_ids":title_input_ids,"title_attention_mask":title_attention_mask,"title_token_type_ids":title_token_type_ids,
            "story_input_ids":story_input_ids,"story_attention_mask":story_attention_mask,"story_token_type_ids":title_token_type_ids,
            "label":label
            }

train_dataset = text_dataset_tokenize_first("train.csv",tokenizer)
train_loader = DataLoader(train_dataset,batch_size=64,shuffle=True)

class first_network(nn.Module):
  def __init__(self):
    super(first_network,self).__init__()  
    self.bert_model = BertModel.from_pretrained(model_name)

    self.linear = nn.Linear(2048,100)
    self.linear2 = nn.Linear(100,10)
    self.linear3 = nn.Linear(10,1)
    self.me = "sippo"
  
  def forward(self,x):
    x1,x2 = x 
    x1 = self.bert_model.forward(**x1).pooler_output
    x2 = self.bert_model.forward(**x2).pooler_output
    z1 = torch.cat([x1,x2],dim=1)
    x1 = self.linear(z1) 
    z1 = nn.ReLU()(x1)
    x1 = self.linear2(z1)
    z1 = nn.ReLU()(x1)
    x1 = self.linear3(z1)
    z1 = nn.ReLU()(x1)

    return z1 

model = first_network()
model = model.to(device)

lr = 0.03 
epochs = 1

loss_fn = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(),lr=lr)
loss_stack = []

def train_loop(dataloader,model,loss_fn,optimizer):
  for idx,data in enumerate(dataloader):

    #デバッグ用
    if idx == 0:
        print("data loading ...")

    #DataLoaderからの返り値を整形
    title_input_ids,title_attention_mask,title_token_type_ids = data["title_input_ids"],data["title_attention_mask"],data["title_token_type_ids"]
    story_input_ids,story_attention_mask,story_token_type_ids = data["story_input_ids"],data["story_attention_mask"],data["story_token_type_ids"]
    x = [
        {"input_ids":title_input_ids,"attention_mask":title_attention_mask,"token_type_ids":title_token_type_ids},
         {"input_ids":story_input_ids,"attention_mask":story_attention_mask,"token_type_ids":story_token_type_ids}
         ]

    y = data["label"]

    #デバッグ用。処理が進んでいるかの確認
    if idx == 0:
      print("let's predict!")

    pred = model(x)
    loss = loss_fn(pred,y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if idx < 6:
      print(f"calc:{idx}batch \n loss:{loss.item()}")
    if idx % 20 == 0:
      print(f"calc:{idx} \n loss:{loss.item()}")
      loss_stack.append(loss.item())
  print(f'epoch finished!! \n loss:{loss.item()}')

for epoch in range(epochs):
  train_loop(train_loader,model,loss_fn,optimizer)
  print(loss_stack)