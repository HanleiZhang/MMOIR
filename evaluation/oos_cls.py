import math
import numpy as np
from scipy.stats import norm as dist_model
from utils.metrics import OID_Metrics
import logging

def classify_doc(args, y_prob, mu_stds):    #如果一个文档在任何类别上的最大预测概率高于根据该类别标准差计算出的阈值，则将其分类为该类别。否则，将其分类为异常类别。

    logger = logging.getLogger(args.logger_name)
    thresholds = {}
    for col in range(args.num_labels):
        threshold = max(0.5, 1 - args.scale * mu_stds[col][1])
        thresholds[col] = threshold
    thresholds = np.array(thresholds)
    logger.info('Probability thresholds of each class: %s', thresholds)
    
    y_pred = []
    for p in y_prob:
        max_class = np.argmax(p)
        max_value = np.max(p)
        threshold = max(0.5, 1 - args.scale * mu_stds[max_class][1])

        if max_value > threshold:
            y_pred.append(max_class)
        else:
            y_pred.append(args.ood_label_id)

    return np.array(y_pred)
    
def fit(prob_pos_X):     #根据dist_model模型得到均值和标准差
    prob_pos_X=[value if math.isfinite(value) else 0.0 for value in prob_pos_X]   #math.isfinite(value) 检查 value 是否是一个有限数（即不是无穷大或NaN）。如果 value 是有限数，它将被保留；如果不是，它将被替换为 0.0。
    prob_pos = [p for p in prob_pos_X] + [2 - p for p in prob_pos_X]
    pos_mu, pos_std = dist_model.fit(prob_pos)
    return pos_mu, pos_std

def cal_mu_std(y_prob, trues, num_labels):   #这个函数用于分析分类问题中每个类别的预测概率分布。

    mu_stds = []
    for i in range(num_labels):
        pos_mu, pos_std = fit(y_prob[trues == i, i])   #y_prob[trues == i, i] 是一个切片操作，它从 y_prob 中选取那些实际类别标签等于当前索引 i 的样本的第 i 列的概率值。例如，如果 i 为0，这个表达式将选取所有实际标签为0的样本的第0列概率值。
        mu_stds.append([pos_mu, pos_std])

    return mu_stds

def doc_classification(args, inputs):
    
    oid_metrics = OID_Metrics(args)

    mu_stds = cal_mu_std(inputs['y_logit_train'], inputs['y_true_train'], args.num_labels)    #每个元素是一个包含两个值的列表，分别表示均值和标准差。
    y_pred = classify_doc(args, inputs['y_logit_test'], mu_stds)
    oid_test_results = oid_metrics(inputs['y_true_test'], y_pred, show_results = True)

    return oid_test_results