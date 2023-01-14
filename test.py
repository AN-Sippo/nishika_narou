import lightgbm as lgb 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
import numpy as np 
from collections import OrderedDict

# 自作部分
from  dataset_for_lightgbm import numerfical_dataset
from utils.loss_funcs import accuracy

# -----herper parameters-----------
mode = "train"
n_estimators_l = [1000]
lr_l = [0.001,0.003,0.01,0.03,0.1,0.3]
random_state = 34
# ---------------------------------

def train_loop(n_estimators:int,lr:int,X_train,y_train,eval_sets,callbacks:list=None,eval_metric:list=[accuracy])->OrderedDict:
    """
    model.evals_result_で、{"valid_0":OrderedDict}が帰ってくる。
    この中身のOrderedDictをそのまま返す。({"multi_logloss":list,"accuracy":list})
    """
    model = lgb.LGBMClassifier(n_estimators=n_estimators,boosting_type="goss",learning_rate=lr,random_state=random_state)
    model.fit(X_train,y_train,eval_set=eval_sets,callbacks=callbacks,eval_metric=accuracy)
    return model.evals_result_["valid_0"]

df = pd.read_csv("datas/with_keyword_train.csv")
df_train,df_eval = train_test_split(df,test_size=0.3)
df_train = df_train.reset_index(drop=True)
df_eval = df_eval.reset_index(drop=True)

train_dataset = numerfical_dataset(df_train)
test_dataset = numerfical_dataset(df_eval)


X_train,y_train = train_dataset.getall()
X_eval,y_eval = test_dataset.getall()


#get categorical indices numbers 
_categorical_l = ["writer_n","biggenre","genre","novel_type","end","isstop","isbl",'isgl','isr15','istenni','istensei','iszankoku','is']
categorical_l = []
for name in _categorical_l:
    if name in X_train.columns:
        categorical_l.append(X_train.columns.get_loc(name))

X_train,y_train = X_train.values,y_train.values
X_eval,y_eval = X_eval.values,y_eval.values

y_train = np.reshape(y_train,(-1))
y_eval = np.reshape(y_eval,(-1))

eval_sets = [(X_eval,y_eval)]

callbacks = []
callbacks.append(lgb.early_stopping(stopping_rounds=30))
callbacks.append(lgb.log_evaluation())


fig,axs = plt.subplots(6,6)
fig.set_figheight(13)
fig.set_figwidth(13)

for i,lr in enumerate(lr_l):
    for j,n_estimators in enumerate(n_estimators_l):
        ax = axs[i][j]

        loss = train_loop(n_estimators=n_estimators,lr=lr,X_train=X_train,y_train=y_train,eval_sets=eval_sets,callbacks=callbacks)

        y_multi_logloss = np.array(loss["multi_logloss"])
        y_accuracy = np.array(loss["accuracy"])

        both_x = np.arange(0,y_multi_logloss.size,1)

        # ax.plot(both_x,y_multi_logloss,label="multi_logloss",color="red")
        ax.plot(both_x,y_accuracy,label="accuracy",color="blue")
        ax.legend()
        ax.set_yticks([0,0.25,0.5,0.75,1])
        ax.grid(axis="both")



plt.show()