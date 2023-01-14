from torch.utils.data import Dataset
import torch 
from transformers import BatchEncoding,BertTokenizer
import pandas as pd 

device = "cpu" #ローカルで実行していたため。

class text_dataset_tokenize_first(Dataset):
  def __init__(self,csv_path:str,tokenizer:BertTokenizer):
    self.path = csv_path 
    self.datas = pd.read_csv(self.path)
    self.tokenizer = tokenizer 


    # transforms
    self.transform = None
    self.target_transform = None

    #tokenizeしてから格納 map{str:torch.tensor}
   
    self.titles = tokenizer(self.datas["title"].tolist(),truncation="longest_first",padding="max_length",max_length=128,return_tensors="pt")
    self.titles = self.titles.to(device)

    self.stories = tokenizer(self.datas["story"].tolist(),truncation="longest_first",padding="max_length",max_length=128,return_tensors="pt")
    self.stories = self.stories.to(device)

    #labelsはtensor化してから格納
    self.labels = torch.tensor(self.datas["fav_novel_cnt_bin"].tolist(),dtype=torch.float32)
    self.labels = self.labels.to(device)


  def __len__(self):
    return len(self.labels)

  def __getitem__(self,idx):
    # DataLoaderに渡すため、__get_item__が返すmapobjectのvalueはtorch.tensorであるべき？
    
    title_input_ids = self.titles["input_ids"][idx]
    title_attention_mask = self.titles["attention_mask"][idx]
    title_token_type_ids = self.titles["token_type_ids"][idx]

    story_input_ids = self.stories["input_ids"][idx]
    story_attention_mask = self.stories["attention_mask"][idx]
    story_token_type_ids = self.stories["token_type_ids"][idx]

    label = self.labels[idx]

    if self.transform:
      title = self.transform(title)
      story = self.transform(story)

    if self.target_transform:
      label = self.target_transform(label)

    return {"title_input_ids":title_input_ids,"title_attention_mask":title_attention_mask,"title_token_type_ids":title_token_type_ids,
            "story_input_ids":story_input_ids,"story_attention_mask":story_attention_mask,"story_token_type_ids":story_token_type_ids,
            "label":label
            }
