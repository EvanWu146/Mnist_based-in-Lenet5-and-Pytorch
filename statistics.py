import torch
import numpy as np

def digital_num(data):
    cal_dict = {}
    for i in range(0, 10):
        cal_dict[i] = 0
    for batch_idx, (data, target) in enumerate(data):
        for x in target:
            cal_dict[x.item()] = cal_dict[x.item()]+1
    print(cal_dict)


def get_basic_indicator(target, pred):
    res = {'TP':{}, 'FP': {}, 'TN': {}, 'FN': {}}

    for k in res.keys():
        for i in range(0, 10):
            res[k][i] = 0  # 初始化各项基本指标

    if target.size() == pred.size():
        for i in range(0, target.shape[0]):
            t = target[i].item()
            p = pred[i].item()
            if t == p:
                res['TP'][t] = res['TP'][t] + 1  # 被正确分类
                for j in range(0, 10):
                    if j != t:
                        res['TN'][j] = res['TN'][j] + 1
                        # 对任何其它类别来说是正确被分入负类
            else:
                res['FN'][t] = res['FN'][t] + 1  # 被误分类为负样本
                res['FP'][p] = res['FP'][p] + 1

    return res



def macro_Avg(target, pred):
    raw = get_basic_indicator(target, pred)
    Accuracy = []
    Precision = []
    Recall = []
    F1 = []
    for i in range(0, 10):
        Accuracy.append(
            (raw['TP'][i] + raw['TN'][i])/
            (raw['TP'][i] + raw['TN'][i] + raw['FP'][i] + raw['FN'][i])
        )
        Precision.append(
            raw['TP'][i] /
            (raw['TP'][i] + raw['FP'][i])
        )
        Recall.append(
            raw['TP'][i] /
            (raw['TP'][i] + raw['FN'][i])
        )
        F1.append(
            (2 * Precision[i] * Recall[i]) /
            (Precision[i] + Recall[i])
        )

    print("The Macro-Average is:")
    print("Accuracy: {:.4f}".format(np.mean(Accuracy)))
    print("Precision: {:.4f}".format(np.mean(Precision)))
    print("Recall: {:.4f}".format(np.mean(Recall)))
    print("F1 score: {:.4f}".format(np.mean(F1)))

    pass


def micro_Avg(target, pred):
    raw = get_basic_indicator(target, pred)

    TP = []
    for k in raw['TP']:
        TP.append(raw['TP'][k])
    FP = []
    for k in raw['FP']:
        FP.append(raw['FP'][k])
    TN = []
    for k in raw['TN']:
        TN.append(raw['TN'][k])
    FN = []
    for k in raw['FN']:
        FN.append(raw['FN'][k])

    _TP = np.mean(TP)
    _FP = np.mean(FP)
    _TN = np.mean(TN)
    _FN = np.mean(FN)

    print("\nThe Micro-Average is:")
    print("Accuracy: {:.4f}".format(
        (_TP + _TN) /
        (_TP + _TN + _FN + _FP)
    ))
    P = _TP / (_TP + _FP)
    R = _TP / (_TP + _FN)

    print("Precision: {:.4f}".format(P))
    print("Recall: {:.4f}".format(R))
    print("F1 score: {:.4f}".format(
        (2 * P * R) /
        (P + R)
    ))
    pass

if __name__ == '__main__':
    # get_basic_indicator()
    pass

