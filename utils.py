from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score
from sklearn.metrics import f1_score
from sklearn.manifold import TSNE
from scipy.special import softmax
from sklearn.metrics import confusion_matrix
from multiprocessing import Pool
import time
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
import numpy as np

def calculate_micro_macro_auprc(y_true, y_scores):
    """
    计算多标签数据的Micro和Macro AUPRC
    :param y_true: 真实标签，二维数组或列表，形状为[n_samples, n_labels]
    :param y_scores: 预测得分或概率，与y_true具有相同形状
    :return: Micro AUPRC和Macro AUPRC
    """
    # 宏平均（Macro-average）
    n_labels = y_true.shape[1]
    macro_precisions = []
    macro_recalls = []
    for i in range(n_labels):
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_scores[:, i])
        macro_precisions.append(precision)
        macro_recalls.append(recall)
    macro_auprc = np.mean([auc(recall, precision) for precision, recall in zip(macro_precisions, macro_recalls)])

    # 微平均（Micro-average）
    y_true_combined = y_true.ravel()
    y_scores_combined = y_scores.ravel()
    precision, recall, _ = precision_recall_curve(y_true_combined, y_scores_combined)
    micro_auprc = auc(recall, precision)

    return micro_auprc, macro_auprc


def save_macro_auprc_plot(y_true, y_scores, filename):
    """
    保存多标签数据的Macro AUPRC曲线图像
    :param y_true: 真实标签，二维数组或列表，形状为[n_samples, n_labels]
    :param y_scores: 预测得分或概率，与y_true具有相同形状
    :param filename: 保存图像的文件名
    """
    n_labels = y_true.shape[1]
    macro_precisions = []
    macro_recalls = []
    for i in range(n_labels):
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_scores[:, i])
        macro_precisions.append(precision)
        macro_recalls.append(recall)
    
    # 计算所有标签的平均精确度和召回率
    mean_recall = np.linspace(0, 1, 100)
    interpolated_precisions = []
    for i in range(n_labels):
        interpolated_precision = np.interp(mean_recall, macro_recalls[i][::-1], macro_precisions[i][::-1])
        interpolated_precisions.append(interpolated_precision)
    mean_precision = np.mean(interpolated_precisions, axis=0)

    # 计算宏平均AUPRC
    macro_auprc = auc(mean_recall, mean_precision)

    # 绘制宏平均AUPRC曲线
    plt.figure()
    plt.plot(mean_recall, mean_precision, label='Macro-average AUPRC (area = {0:0.4f})'.format(macro_auprc))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Macro-average Precision-Recall Curve')
    plt.legend(loc="lower right")
    
    # 保存图像
    plt.savefig(filename, format='png')

    return macro_auprc


def balanced_accuracy(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tpr = cm.diagonal() / cm.sum(axis=1)  # 计算每个类别的TPR
    bacc = sum(tpr) / len(tpr)
    return bacc # 计算平均值

def eval_precision(results, gt_labels):
    y_pred = np.asarray(results) > 0
    tp = gt_labels * y_pred
    fp = (1 - gt_labels) * y_pred
    under = np.sum(tp + fp, axis=0)
    under = np.where(under == 0, 1, under)
    precision = np.sum(tp, axis=0)  / under
    mAP = np.mean(precision)
    return mAP, precision

def eval_recall(results, gt_labels):
    y_pred = np.asarray(results) > 0
    tp = gt_labels * y_pred
    fn = gt_labels * (1 - y_pred)
    recall = np.sum(tp, axis=0)  / np.sum(tp + fn, axis=0)
    mAR = np.mean(recall)
    return mAR, recall


# tpr =  tp / (tp + fn)
# tnr = tn / (fp + tn )
def eval_bacc(y_pred, gt_labels):
    tp = gt_labels * y_pred
    tn = (1 - gt_labels) * (1 - y_pred)
    fp = (1 - gt_labels) * y_pred
    fn = gt_labels * (1 - y_pred)
    assert np.sum(tp + tn + fp + fn) == np.shape(gt_labels)[0]*np.shape(gt_labels)[1]
    tpr =  np.sum(tp, axis=0)  / np.sum(tp + fn, axis=0) 
    tnr = np.sum(tn, axis=0)  / np.sum(fp + tn, axis=0)
    per_class_acc = (tpr + tnr) / 2
    acc = np.mean(per_class_acc)
    return acc, per_class_acc

def eval_SE(y_pred, gt_labels):
    tp = gt_labels * y_pred
    tn = (1 - gt_labels) * (1 - y_pred)
    fp = (1 - gt_labels) * y_pred
    fn = gt_labels * (1 - y_pred)
    assert np.sum(tp + tn + fp + fn) == np.shape(gt_labels)[0]*np.shape(gt_labels)[1]
    tpr =  np.sum(tp, axis=0)  / np.sum(tp + fn, axis=0) 
    tnr = np.sum(tn, axis=0)  / np.sum(fp + tn, axis=0)
    sen = np.mean(tpr)
    spe = np.mean(tnr)
    return sen, spe


def eval_auc(results, gt_labels):
    macro_auc = roc_auc_score(gt_labels, results , average="macro")
    micro_auc = roc_auc_score(gt_labels, results,  average="micro")
    weighted_auc = roc_auc_score(gt_labels, results,  average="weighted")
    per_auc = roc_auc_score(gt_labels, results,  average=None)
    return macro_auc, micro_auc, weighted_auc, per_auc

def eval_F1(y_pred, gt_labels):
    micro_f1 = f1_score(gt_labels, y_pred, average='micro')
    macro_f1 = f1_score(gt_labels, y_pred, average='macro')
    weighted_f1 = f1_score(gt_labels, y_pred, average='weighted')
    print(f"Total macro F1-score: {macro_f1}")
    return micro_f1, macro_f1, weighted_f1

def obtaion_LT_distribution(y_true):
        # 计算每个类别的频率
    class_freq = np.bincount(y_true)

    # 确定 head, medium, tail 的阈值
    num_classes = len(class_freq)
    head_threshold = np.percentile(class_freq, 65)
    medium_threshold = np.percentile(class_freq, 20)
    # 划分类别
    head_classes = np.where(class_freq > head_threshold)[0]
    medium_classes = np.where((class_freq <= head_threshold) & (class_freq > medium_threshold))[0]
    tail_classes = np.where(class_freq <= medium_threshold)[0]
    return head_classes, medium_classes, tail_classes

def obtaion_LT_multi_label_distribution(y_true):
        # 计算每个类别的频率
    class_freq = np.sum(y_true, axis=0)

    # 确定 head, medium, tail 的阈值
    num_classes = len(class_freq)
    head_threshold = np.percentile(class_freq, 65)
    # medium_threshold = np.percentile(class_freq, 20)
    medium_threshold = 10
    # 划分类别
    head_classes = np.where(class_freq > head_threshold)[0]
    medium_classes = np.where((class_freq <= head_threshold) & (class_freq > medium_threshold))[0]
    tail_classes = np.where(class_freq <= medium_threshold)[0]
    return head_classes, medium_classes, tail_classes


def LT_eval_F1score(y_true, y_pred):
    # 假设 y_true 是测试集的标签，y_pred 是模型的预测结果
    # y_true = ...
    # y_pred = ...

    # 计算每个类别的频率
    class_freq = np.bincount(y_true)

    # 确定 head, medium, tail 的阈值
    num_classes = len(class_freq)
    head_threshold = np.percentile(class_freq, 65)
    medium_threshold = np.percentile(class_freq, 20)

    # 划分类别
    head_classes = np.where(class_freq > head_threshold)[0]
    medium_classes = np.where((class_freq <= head_threshold) & (class_freq > medium_threshold))[0]
    tail_classes = np.where(class_freq <= medium_threshold)[0]

    # 计算每个类别的 F1-score
    head_f1 = f1_score(y_true[np.isin(y_true, head_classes)], y_pred[np.isin(y_true, head_classes)], average='macro')
    medium_f1 = f1_score(y_true[np.isin(y_true, medium_classes)], y_pred[np.isin(y_true, medium_classes)], average='macro')
    tail_f1 = f1_score(y_true[np.isin(y_true, tail_classes)], y_pred[np.isin(y_true, tail_classes)], average='macro')

    # head_acc = balanced_accuracy(y_true[np.isin(y_true, head_classes)], y_pred[np.isin(y_true, head_classes)])
    # medium_acc = balanced_accuracy(y_true[np.isin(y_true, medium_classes)], y_pred[np.isin(y_true, medium_classes)])
    # tail_acc = balanced_accuracy(y_true[np.isin(y_true, tail_classes)], y_pred[np.isin(y_true, tail_classes)])


    print(f"Head F1-score: {head_f1}")
    print(f"Medium F1-score: {medium_f1}")
    print(f"Tail F1-score: {tail_f1}")

    # print(f"Head Balance Accuracy: {head_acc}")
    # print(f"Medium Balance Accuracy: {medium_acc}")
    # print(f"Tail Balance Accuracy: {tail_acc}")

def LT_eval_accscore(y_true, y_pred):
    # 假设 y_true 是测试集的标签，y_pred 是模型的预测结果
    # y_true = ...
    # y_pred = ...

    # 计算每个类别的频率
    class_freq = np.bincount(y_true)

    # 确定 head, medium, tail 的阈值
    num_classes = len(class_freq)
    head_threshold = np.percentile(class_freq, 65)
    medium_threshold = np.percentile(class_freq, 20)

    # 划分类别
    head_classes = np.where(class_freq > head_threshold)[0]
    medium_classes = np.where((class_freq <= head_threshold) & (class_freq > medium_threshold))[0]
    tail_classes = np.where(class_freq <= medium_threshold)[0]

    # 计算每个类别的 F1-score

    head_acc = accuracy_score(y_true[np.isin(y_true, head_classes)], y_pred[np.isin(y_true, head_classes)])
    medium_acc = accuracy_score(y_true[np.isin(y_true, medium_classes)], y_pred[np.isin(y_true, medium_classes)])
    tail_acc = accuracy_score(y_true[np.isin(y_true, tail_classes)], y_pred[np.isin(y_true, tail_classes)])

    print(f"Head Accuracy: {head_acc}")
    print(f"Medium Accuracy: {medium_acc}")
    print(f"Tail Accuracy: {tail_acc}")


def split_list(lst, chunk_size):
    result = []
    for i in range(0, len(lst), chunk_size):
        chunk = lst[i:i+chunk_size]
        result.append(chunk)
    return result


def count_parameters_in_million(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params / 1e6  # Convert to millions

