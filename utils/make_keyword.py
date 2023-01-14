import pandas as pd 
import re 
from transformers import BertTokenizer
from collections import defaultdict as dd

train_data = pd.read_csv("../datas/train.csv")
URL_PATTERN = re.compile(r'http[\w:./\d]+') # URL
DATE_PATTERN = re.compile(r'\d+/\d+/\d+') # DATE
HIRAGANA_PATTERN = re.compile('[\u3041-\u309F]+') # ひらがな
KATAKANA_PATTERN = re.compile('[\u30A1-\u30FF]+') # カタカナ
EIGO_PATTERN = re.compile('[\u0041-\u005A]+|[\u0061-\u007A]+') # 英語
SUUZI_PATTERN = re.compile('[\u0030-\u0039]+') # 数字

model_name = "cl-tohoku/bert-large-japanese"
tokenizer = BertTokenizer.from_pretrained(model_name)

# print(tokenizer.tokenize(train_data.at[0,"story"]))

class token_transform:
    # delete処理はループ回しながらだとインデックスがずれて、考えるのが面倒くさくなった。
    # ので、削除したい時はとりあえず[UNK]トークンに置き換えておいて、delete_unkで安全に消すことにした。
    def delete_hash(self,tokens:list):
        for idx,token in enumerate(tokens):
            if len(token) >= 3:
                if token[0] == token[1] == "#":
                    tokens[idx] = token[2:]
                elif token[0] == "#":
                    tokens[idx] = token[1:]

            elif len(token) == 2:
                if token[0] == token[1] == "#":
                    tokens[idx] = "[UNK]"
                elif token[0] == "#":
                    tokens[idx] = token[1:]

            elif len(token) == 1:
                if token[0] == "#":
                    tokens[idx] = "[UNK]"
        return tokens 

    def delete_unk(self,tokens:list):
        count = 0

        for token in tokens:
            if token == "[UNK]":
                count += 1
        for _ in range(count):
            tokens.remove("[UNK]")
        return tokens 
        
    
    def delete_unrelated(self,tokens:list):
        unrelated_words = set(["「","」","[","]","『","』","。","、",",","(",")","."," ,",", ",", "])

        for idx,token in enumerate(tokens):
            # ひらがな一文字は流石に関係なしと見ていいと思う
            if len(token) == 1:
                if ord(token) >= 12354 and ord(token) <= 12435:
                    tokens[idx] = "[UNK]"
            # 上記ありがちな符号たち。
            if token in unrelated_words:
                tokens[idx] = "[UNK]"
        return tokens 

    def transform(self,tokens:list):
        tokens = self.delete_hash(tokens)
        tokens = self.delete_unrelated(tokens)
        tokens = self.delete_unk(tokens)
        return tokens 



def make_keyword(df:pd.DataFrame,column:str,cutoff_fav:int=1,cutoff_rank:int=200)->list:
    """
     対象カラム名と,データを受け取って、トップトークンリストを返す
        column:トークン作成の対象とするカラム名。
        cutoff_fav:favがいくつ以上を、トップトークン作成の対象とするか。
        cutoff_rank:トークン全体のうち、上から何番目までをトップトークンとするか。(1-...)
    """
    token_transformer = token_transform()
    token_dic = dd(int)

    df_over_cutoff = df[df["fav_novel_cnt_bin"] >= cutoff_fav]
    for sentense in df_over_cutoff[column]:
        for token in token_transformer.transform(tokenizer.tokenize(sentense)):
            token_dic[token] += 1
    token_l = list(token_dic.items())
    token_l = list(map(lambda x:x[0],sorted(token_l,key=lambda x:x[1],reverse=True)[:cutoff_rank]))
    return token_l 

title_cutoff_rank = 200
story_cutoff_rank = 200

title_keywords = make_keyword(train_data,"title",cutoff_rank=title_cutoff_rank)
story_keywords = make_keyword(train_data,"story",cutoff_rank=story_cutoff_rank)


# この初期化は遅いらしい。（どうしたらいいのかよくわからん）
for i in range(title_cutoff_rank):
    train_data[f"title_{i}"] = 0
for i in range(story_cutoff_rank):
    train_data[f"story_{i}"] = 0

token_transformer = token_transform()
for idx in range(len(train_data)):
    #title
    tokens = set(token_transformer.transform(tokenizer.tokenize(train_data.at[idx,"title"])))
    for key_idx,key_token in enumerate(title_keywords):
        if key_token in tokens:
            train_data.at[idx,f"title_{key_idx}"] = 1
    
    #story
    tokens = set(token_transformer.transform(tokenizer.tokenize(train_data.at[idx,"story"])))
    for key_idx,key_token in enumerate(story_keywords):
        if key_token in tokens:
            train_data.at[idx,f"story_{key_idx}"] = 1
    
    if idx % 100 == 0:
        print(f"{idx} / {len(train_data)} done!")

# print(train_data.head())
train_data = train_data.set_index("ncode",drop=True)
train_data.to_csv("../datas/with_keyword_train.csv")
        
        
