import torch
import CARZero
import pandas as pd 
import json
import numpy as np
from utils import *
import os
from sklearn.preprocessing import MultiLabelBinarizer



def obtain_simr(image_path, text_path):
    df = pd.read_csv(image_path)
    with open(text_path, 'r') as f:
        cls_prompts = json.load(f)

    # load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    CARZero_model = CARZero.load_CARZero(name="CARZero_vit_b_16", device=device)
  
    # process input images and class prompts 
    ## batchsize
    bs = 1024
    image_list = split_list(df['Path'].tolist(), bs)
    processed_txt = CARZero_model.process_class_prompts(cls_prompts, device)
    for i, img in enumerate(image_list):
        processed_imgs = CARZero_model.process_img(img, device)
        # zero-shot classification on 1000 images
        similarities = CARZero.dqn_shot_classification(
            CARZero_model, processed_imgs, processed_txt)
        
        if i == 0:
            similar = similarities
        else:
            similar = pd.concat([similar, similarities], axis=0)

    return similar


def tripple_openi_rusult_merge(predict_csv, label_file_path):
    pathologies = [
        # NIH
        "Atelectasis",
        "Cardiomegaly",
        "Effusion",
        "Infiltration",
        "Mass",
        "Nodule",
        "Pneumonia",
        "Pneumothorax",
        ## "Consolidation",
        "Edema",
        "Emphysema",
        "Fibrosis",
        "Pleural_Thickening",
        "Hernia",
        # ---------
        "Fracture",
        "Opacity",
        "Lesion",
        # ---------
        "Calcified Granuloma",
        "Granuloma",
        # ---------
        "No_Finding",
    ]


    mapping = dict()
    mapping["Pleural_Thickening"] = ["pleural thickening"]
    mapping["Infiltration"] = ["Infiltrate"]
    mapping["Atelectasis"] = ["Atelectases"]

    # Load data
    csv = pd.read_csv(label_file_path)
    csv = csv.replace(np.nan, "-1")

    gt = []
    for pathology in pathologies:
        mask = csv["labels_automatic"].str.contains(pathology.lower())
        if pathology in mapping:
            for syn in mapping[pathology]:
                # print("mapping", syn)
                mask |= csv["labels_automatic"].str.contains(syn.lower())
        gt.append(mask.values)

    gt = np.asarray(gt).T
    gt = gt.astype(np.float32)

    # Rename pathologies
    pathologies = np.char.replace(pathologies, "Opacity", "Lung Opacity")
    pathologies = np.char.replace(pathologies, "Lesion", "Lung Lesion")

    ## Rename by myself
    pathologies = np.char.replace(pathologies, "Pleural_Thickening", "pleural thickening")
    pathologies = np.char.replace(pathologies, "Infiltration", "Infiltrate")
    pathologies = np.char.replace(pathologies, "Atelectasis", "Atelectases")
    gt[np.where(np.sum(gt, axis=1) == 0), -1] = 1
    
    label = gt[:, :-1]

    predict = pd.read_csv(predict_csv).values

    head, medium, tail = obtaion_LT_multi_label_distribution(label)

    macro_auc, micro_auc, weighted_auc, per_auc = eval_auc(predict[:, head], label[:, head])
    print(f"Head AUC: {macro_auc}")
    macro_auc, micro_auc, weighted_auc, per_auc = eval_auc(predict[:, medium], label[:, medium])
    print(f"Medium AUC: {macro_auc}")
    macro_auc, micro_auc, weighted_auc, per_auc = eval_auc(predict[:, tail], label[:, tail])
    print(f"Tail AUC: {macro_auc}")
    macro_auc, micro_auc, weighted_auc, per_auc = eval_auc(predict, label)
    print(f"Total AUC: {macro_auc}")
    micro_prc, macro_prc = calculate_micro_macro_auprc(label, predict)
    print("Micro AUPRC: {:.4f}, Macro AUPRC: {:.4f}".format(micro_prc, macro_prc))

    for i, k in enumerate(pathologies[:-1]):
            print(f"{k}: {per_auc[i]}")


def tripple_padchest_rusult_merge(predict_csv, label_file_path):
    test_query = ['atelectasis', 'cardiomegaly', 'consolidation', 'pulmonary edema', 'pneumonia']

    with open(label_file_path, "r") as file:
         data = json.load(file) 
    label = []
    key = data.keys()
    for k in key:
        label += data[k]
    unique_label = list(set(label))

    # Sort the unique strings with stable sorting
    sorted_strings = sorted(unique_label, key=lambda x: (x, label.index(x)))
    
    index = sorted_strings.index('normal')

    labels = [ data[k] for k in key ]

    # 创建MultiLabelBinarizer对象
    mlb = MultiLabelBinarizer(classes=sorted_strings)

    # 使用fit_transform()方法进行One-Hot编码
    encoded_labels = mlb.fit_transform(labels)

    predict = pd.read_csv(predict_csv).values

    pre = np.zeros((predict.shape[0] , predict.shape[1]))
    for i in range(predict.shape[0]):
        logit = predict[i]
        ind = np.argmax(logit)
        pre[i, ind] = 1

    encoded_labels =  np.delete(encoded_labels, index, axis=1)
    # 删除normal
    sorted_strings.remove('normal')

    ## 查找test_query的index
    test_query_index = []
    for i in test_query:
        test_query_index.append(sorted_strings.index(i))

    head, medium, tail = obtaion_LT_multi_label_distribution(encoded_labels)
    count = np.sum(encoded_labels, axis=0)

    print(np.asanyarray(sorted_strings)[tail])
    macro_auc, micro_auc, weighted_auc, per_auc = eval_auc(predict[:, head], encoded_labels[:, head])
    print(f"Head AUC: {macro_auc}")
    macro_auc, micro_auc, weighted_auc, per_auc = eval_auc(predict[:, medium], encoded_labels[:, medium])
    print(f"Medium AUC: {macro_auc}")
    macro_auc, micro_auc, weighted_auc, per_auc = eval_auc(predict[:, tail], encoded_labels[:, tail])
    print(f"Tail AUC: {macro_auc}")
    macro_auc, micro_auc, weighted_auc, per_auc = eval_auc(predict, encoded_labels)
    # pd.DataFrame(per_auc, index=sorted_strings).to_csv('padchest_auc.csv')
    print(f"Total AUC: {macro_auc}")
    micro_prc, macro_prc = calculate_micro_macro_auprc(encoded_labels, predict)
    print("Micro AUPRC: {:.4f}, Macro AUPRC: {:.4f}".format(micro_prc, macro_prc))
    # 打印test_query的AUC
    for i in test_query_index:
        macro_auc, micro_auc, weighted_auc, per_auc = eval_auc(predict[:, i], encoded_labels[:, i])
        print(f"{sorted_strings[i]} AUC: {macro_auc}")
    
    n_classes = encoded_labels.shape[1]
    tail_classes = []
    auc_scores = []
    macro_precisions = []
    macro_recalls = []
    for i in range(n_classes):
        # 计算每个类别的正例数目
        positive_count = np.sum(encoded_labels[:, i])
        
        # 如果正例数目少于10，这是一个tail类别
        if positive_count <= 10:
            tail_classes.append(i)
            # 计算并存储该类别的AUC
            auc_score = roc_auc_score(encoded_labels[:, i], predict[:, i])
            auc_scores.append(auc_score)

            precision, recall, _ = precision_recall_curve(encoded_labels[:, i], predict[:, i])
            macro_precisions.append(precision)
            macro_recalls.append(recall)
    macro_auprc = np.mean([auc(recall, precision) for precision, recall in zip(macro_precisions, macro_recalls)])
    
    print("Padhcest20 AUROC: {}".format(np.mean(auc_scores)))
    print("Padhcest20 AUPRC: {}".format(macro_auprc))
    
    
    # 第一步：统计所有类别的正例数目
    class_positive_counts = np.sum(encoded_labels, axis=0)

    # 第三步：从tail类别中选出正例数目最多的top 100个类别
    top_100_tail_classes =  np.argsort(class_positive_counts)[:100]
    auc_scores = []
    macro_precisions = []
    macro_recalls = []

    # 第四步：对这些top 100个tail类别计算AUC和AUPRC
    for class_idx in top_100_tail_classes:
        auc_score = roc_auc_score(encoded_labels[:, class_idx], predict[:, class_idx])
        auc_scores.append(auc_score)

        precision, recall, _ = precision_recall_curve(encoded_labels[:, class_idx], predict[:, class_idx])
        macro_precisions.append(precision)
        macro_recalls.append(recall)

    macro_auprc = np.mean([auc(recall, precision) for precision, recall in zip(macro_precisions, macro_recalls)])

    # 输出结果
    print("Top 100 Tail Classes AUROC: {}".format(np.mean(auc_scores)))
    print("Top 100 Tail Classes AUPRC: {}".format(macro_auprc))


def triple_Chexpert14_result(predict_csv,label_file_path):
    csv_head = ['path', 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Lung Mass', 'Lung Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural Thickening', 'Hernia']
    df_test = pd.read_csv(label_file_path, sep=' ', names=csv_head)

    key = csv_head[1:]

    predict = pd.read_csv(predict_csv).values
    label = df_test[key].values
    pre = np.zeros((predict.shape[0] , predict.shape[1]))
    for i in range(predict.shape[0]):
        logit = predict[i]
        ind = np.argmax(logit)
        pre[i, ind] = 1
    
    macro_auc, micro_auc, weighted_auc, per_auc = eval_auc(predict, label)
    print(f"Total AUC: {macro_auc}")
    micro_prc, macro_prc = calculate_micro_macro_auprc(label, predict)
    print("Micro AUPRC: {:.4f}, Macro AUPRC: {:.4f}".format(micro_prc, macro_prc))
    for disease, auc in zip(key, per_auc):
        print(f"{disease}: {auc}")

    save_macro_auprc_plot(label, predict, predict_csv.replace('.csv', '.png'))
    print(f"Save {predict_csv.replace('.csv', '.png')}")


def triple_Chexpert5_result(predict_csv, label_file_path):
    key = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']

    df_test = pd.read_csv(label_file_path)

    predict = pd.read_csv(predict_csv).values
        
    label = df_test[key].values
    pre = np.zeros((predict.shape[0] , predict.shape[1]))
    for i in range(predict.shape[0]):
        logit = predict[i]
        ind = np.argmax(logit)
        pre[i, ind] = 1
    
    macro_auc, micro_auc, weighted_auc, per_auc = eval_auc(predict, label)
    print(f"Total AUC: {macro_auc}")

    micro_prc, macro_prc = calculate_micro_macro_auprc(label, predict)
    print("Micro AUPRC: {:.4f}, Macro AUPRC: {:.4f}".format(micro_prc, macro_prc))
    for disease, auc in zip(key, per_auc):
        print(f"{disease}: {auc}")


def triple_ChestXDet10_result(predict_csv, label_file_path):
    with open(label_file_path, 'r') as f:
        data = json.load(f)
    all_path = []
    all_label = [] 
    for d in data:
        all_path.append(d['file_name'])
        all_label.append(d['syms'])

    sorted_strings = ['Atelectasis', 'Calcification', 'Consolidation', 'Effusion', 'Emphysema', 'Fibrosis', 'Fracture', 'Mass', 'Nodule', 'Pneumothorax']

    # 创建MultiLabelBinarizer对象
    mlb = MultiLabelBinarizer(classes=sorted_strings)

    # 使用fit_transform()方法进行One-Hot编码
    label = mlb.fit_transform(all_label)
    label = np.asarray(label)
    # images_path = df['path'].tolist()

    predict = pd.read_csv(predict_csv).values
        
    pre = np.zeros((predict.shape[0] , predict.shape[1]))
    for i in range(predict.shape[0]):
        logit = predict[i]
        ind = np.argmax(logit)
        pre[i, ind] = 1
    
   
    micro_f1, macro_f1, weighted_f1 = eval_F1(pre, label)
    macro_auc, micro_auc, weighted_auc, per_auc = eval_auc(predict, label)
    print(f"Total AUC: {macro_auc}")
    micro_prc, macro_prc = calculate_micro_macro_auprc(label, predict)
    print("Micro AUPRC: {:.4f}, Macro AUPRC: {:.4f}".format(micro_prc, macro_prc))

    for disease, auc in zip(sorted_strings, per_auc):
        print(f"{disease}: {auc}")


def triple_chexpert5x200_result(predict_csv, label_file_path):
    df = pd.read_csv(label_file_path)
    head = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
    label = df[head].values
    predict = pd.read_csv(predict_csv).values
    predict = softmax(predict, axis=1)
    acc = accuracy_score(label.argmax(1), predict.argmax(1))
    print(acc)

    # 计算每个类别的精度
    for i, disease in enumerate(head):
        disease_label = label[:, i]
        disease_predict = predict[:, i].round()
        disease_acc = accuracy_score(disease_label, disease_predict)
        print(f"Accuracy for {disease}: {disease_acc}")




if __name__ == '__main__':    

    images = [ 
               './Dataset/OpenI/openi_multi_label_image.csv',
               './Dataset/PadChest/padchest_multi_label_image.csv',
               './Dataset/ChestXray14/chestxray14_test_image.csv',
               './Dataset/Chexpert/chexpert5_test_image.csv',
               './Dataset/ChestXDet10/chestXDet10_test_image.csv',
               './Dataset/Chexpert_5x200/chexpert_5x200_newpath.csv'
               ]
    

    texts = [ 
               './Dataset/OpenI/openi_multi_label_text.json',
               './Dataset/PadChest/padchest_multi_label_text.json',
                './Dataset/ChestXray14/chestxray14_test_text.json',
                 './Dataset/Chexpert/chexpert5_test_text.json',
                  './Dataset/ChestXDet10/chestXDet10_test_text.json',
                  './Dataset/Chexpert_5x200/chexpert_5x200_text.json'
                  ]


    result_file_name =  'test'
    

    os.makedirs('./Performance/'+ result_file_name, exist_ok=True)


    save_csvs = [    
                     './Performance/'+ result_file_name +'/Openi.csv',
                     './Performance/'+ result_file_name +'/Padchest.csv',
                     './Performance/'+ result_file_name +'/ChestXray14.csv',
                     './Performance/'+ result_file_name +'/Chexpert5.csv',
                     './Performance/'+ result_file_name +'/ChestXDet10.csv',
                    './Performance/'+ result_file_name +'/chexpert_5x200.csv'
                    ]
    
    
    for i, (img, txt, savecsv) in  enumerate(zip(images, texts, save_csvs)):
            start = time.time()
            similarities = obtain_simr(img, txt)
            similarities.to_csv(savecsv, index=False)
            print(time.time() - start)

    print('Openi')
    tripple_openi_rusult_merge(save_csvs[0], './Dataset/OpenI/custom.csv')
    print('Padchest')
    tripple_padchest_rusult_merge(save_csvs[1], "./Dataset/PadChest/manual_image.json")
    print('ChestXray14')
    triple_Chexpert14_result(save_csvs[2], './Dataset/ChestXray14/test_list.txt')
    print('Chexpert5')
    triple_Chexpert5_result(save_csvs[3], './Dataset/Chexpert/test_labels.csv')
    print('ChestXDet10')
    triple_ChestXDet10_result(save_csvs[4], './Dataset/ChestXDet10/test.json')
    print('chexpert5x200')
    triple_chexpert5x200_result(save_csvs[5], './Dataset/Chexpert_5x200/chexpert_5x200_newpath.csv')






    