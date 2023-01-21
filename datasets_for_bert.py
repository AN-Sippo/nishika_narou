import torch 
from torch import nn 
from transformers import BertTokenizer 
from torch.utils.data import Dataset,DataLoader
import pandas as pd 
import re 


model_name:str="cl-tohoku/bert-base-japanese"
tokenizer = BertTokenizer.from_pretrained(model_name)

URL_PATTERN = re.compile(r'http[\w:./\d]+') # URL
DATE_PATTERN = re.compile(r'\d+/\d+/\d+') # DATE

class dataset_for_bert(Dataset):
    def __init__(self,csv_path:str,max_length:int,column:str,n_class:int=5) -> None:
        #have data
        self.path = csv_path 
        data = pd.read_csv(self.path)
        self.n_class = n_class
        self.max_length = max_length

        #columns
        self.target_column = "fav_novel_cnt_bin"
        self.train_column = column

        #to use same dataset for train and predict
        self.is_train = (self.target_column in data.columns)

        #regex 
        data = data[self.train_column].apply(lambda x:URL_PATTERN.sub("[UNK]",x))

        #train_data
        self.train_data = data.tolist()
        self.train_data = tokenizer(self.train_data,truncation=True,padding=True,max_length=self.max_length,return_tensors="pt",return_length=True)
        #padding,truncationがうまく働く自信がないのでエラーハンドリングしておく。
        for input_id in self.train_data["input_ids"]:
            if len(input_id) != self.max_length:
                raise ValueError(f"expected length {self.max_length} but tokenizer returned length{len(input_id)}")

        
        #target_data
        if self.is_train:
            self.target_data = torch.Tensor(data[self.target_column].tolist())
        else:
            self.target_data = None 

    def __len__(self):
        return len(self.train_data["input_ids"])
    
    def __getitem__(self,idx:int):
        data = self.train_data
        input_ids = data["input_ids"][idx]
        attention_mask = data["attention_mask"][idx]
        token_type_ids = data["token_type_ids"][idx]
        if self.is_train:
            res = [0 for _ in range(self.n_class)]
            label = int(self.target_data[idx].item())
            res[label]+=1
            res = torch.Tensor(res)
        else:
            res = torch.Tensor(0)

        
        #test.csvに対して予測を行うときに、datasetからの返り値をそのままbertに突っ込めたら楽なので。
        return {"input_ids":input_ids,"attention_mask":attention_mask,"token_type_ids":token_type_ids,"label":res}

    def getitem_without_label(self,idx:int):
        """
        test.csvに対して予測を行うときに、datasetからの返り値をそのままbertに突っ込めたら楽なので。
        使い方は、__getitem__と同じ。
        ただ、帰ってくるdictにlabelがない。
        """
        data = self.train_data
        input_ids = data["input_ids"][idx]
        attention_mask = data["attention_mask"][idx]
        token_type_ids = data["token_type_ids"][idx]
        if self.is_train:
            res = [0 for _ in range(self.n_class)]
            label = int(self.target_data[idx].item())
            res[label]+=1
            res = torch.Tensor(res)
        else:
            label = None 
        
        return {"input_ids":input_ids,"attention_mask":attention_mask,"token_type_ids":token_type_ids}

if __name__ == "__main__":
    dataset = dataset_for_bert("datas/train.csv",15,"title")
    print(dataset[0])


        



    
    




