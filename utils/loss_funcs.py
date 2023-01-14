import numpy as np
def accuracy(y_true:np.ndarray,y_pred:np.ndarray):
    """
    引数は共にnumpy配列
    y_true:正解ラベル。(n_samples,)の1次元配列
    y_pred:予測値。何も変換されてない生の予測値らしい(?)分類問題では(n_samples,n_classes)で、softmaxされてなさそう。

    返り値は [名前,値,大きい方が良いか]
    """
    #必要情報の取得
    n_samples = y_true.size 
    n_labels = 5

    #生の予測値を予測ラベルに変換
    y_pred_labels = y_pred.reshape(n_labels,n_samples).argmax(axis=0)

    #形状に自信がないので、エラー処理。
    if y_true.size != y_pred_labels.size:
        raise ValueError(f"size mismatch. y_true is {y_true.size}, but y_pred_labels is {y_pred_labels.size}")
    
    #accuracy 
    res = np.sum(y_true == y_pred_labels) / n_samples
    res = round(res,5)

    return "accuracy",res,True
