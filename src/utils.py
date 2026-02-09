import torch
import numpy as np
import pandas as pd
import time
import itertools
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, accuracy_score, classification_report
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

import tensorflow as tf

"""
Set random seeds for reproducibility across PyTorch, NumPy, and TensorFlow.
"""
def setup_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    tf.random.set_seed(seed)

setup_seed(30)

"""
Compute all metrics
"""
def compute_all_metrics(y_test, predictions, dataset, time_, subdirectory, metric_results_, split='test'):
    one_hot_predictions = predictions.argmax(1)
    n_classes = len(np.unique(y_test))
    LABELS = get_class_names(dataset)
    cms = []

    print("")
    from sklearn import metrics
    precision = metrics.precision_score(y_test, one_hot_predictions, average="weighted", zero_division=0)
    accuracy = metrics.accuracy_score(y_test, one_hot_predictions)
    recall = metrics.recall_score(y_test, one_hot_predictions, average="weighted")
    f1_score = metrics.f1_score(y_test, one_hot_predictions, average="weighted", zero_division=0)
    balanced_accuracy_score = metrics.balanced_accuracy_score(y_test, one_hot_predictions) 
    kappa = cohen_kappa_score(y_test, one_hot_predictions)
    metric_results_.append(accuracy)
    metric_results_.append(balanced_accuracy_score)
    metric_results_.append(f1_score)
    metric_results_.append(f1_score)
    metric_results_.append(kappa)


    print("Precision: {:.4f}".format(precision))
    print("Recall: {:.4f}".format(recall))
    print("f1_score: {:.4f}".format(f1_score))
    print("balanced_accuracy_score: {:.4f}".format(balanced_accuracy_score))

    mAP=np.mean(np.asarray([(metrics.average_precision_score(one_hot(y_test, n_classes)[:,c], one_hot(one_hot_predictions, n_classes)[:,c], average="weighted")) for c in range(n_classes)]))
    with open(subdirectory+"/metrics_"+split+"_"+dataset+"_"+time_+".txt", 'w') as file:
        file.write("Dataset: {}".format( dataset))
        file.write("\n")
        file.write("mAP score: {:.4f}\n".format(mAP))
        file.write("precision: {:.4f}\n".format(precision))
        file.write("recall: {:.4f}\n".format(recall))
        file.write("f1_score: {:.4f}\n".format(f1_score))
        file.write("balanced_accuracy_score: {:.4f}".format(balanced_accuracy_score))
        file.write("\n")
        file.write("\n")
        
        file.write("\nF1-score (None):")
        print("F1-score (None):")
        metr = metrics.f1_score(one_hot(y_test, n_classes), one_hot(one_hot_predictions, n_classes), average=None)
        metr = np.round(metr, 4)
        print(metr)
        file.write(str(metr))
        
        file.write("\nF1-score (weighted):")
        metr = metrics.f1_score(one_hot(y_test, n_classes), one_hot(one_hot_predictions, n_classes), average="weighted")
        metr = np.round(metr, 4)
        file.write(str(metr))

        file.write("\nF1-score (macro):")
        metr = metrics.f1_score(one_hot(y_test, n_classes), one_hot(one_hot_predictions, n_classes), average="macro")
        metr = np.round(metr, 4)
        file.write(str(metr))

        file.write("\nF1-score (micro):")
        metr = metrics.f1_score(one_hot(y_test, n_classes), one_hot(one_hot_predictions, n_classes), average="micro")
        metr = np.round(metr, 4)
        file.write(str(metr))
        
        file.write("\nclassification_report:")
        report = classification_report(one_hot(y_test, n_classes), one_hot(one_hot_predictions, n_classes))
        file.write(report)

        cm = metrics.confusion_matrix(y_test, one_hot_predictions)
        metrics = {}

        sensitivities = []
        specificities = []

        for i in range(len(cm)):
            TP = cm[i, i]  # True Positives for class i
            FN = np.sum(cm[i, :]) - TP  # False Negatives for class i
            FP = np.sum(cm[:, i]) - TP  # False Positives for class i
            TN = np.sum(cm) - (TP + FP + FN)  # True Negatives for class i

            sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
            sensitivity = np.round(sensitivity, 4)
            sensitivities.append(sensitivity)

            specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
            specificity = np.round(specificity, 4)
            specificities.append(specificity)

            metrics[i] = {"TP": TP, "TN": TN, "FP": FP, "FN": FN}

        file.write("\n")    
        file.write(str(LABELS))
        file.write("\n")

        print("Métricas por clase (one-vs-all):")
        for cls, m in metrics.items():
            print(f"Clase {cls}: TP={m['TP']}, FP={m['FP']}, FN={m['FN']}, TN={m['TN']}")
        file.write("\n")    
            
        file.write(f"\nSensitivity per class:, {sensitivities}")
        file.write(f"\nSpecificity per class:, {specificities}")
        print()
        print(f"Sensitivity per class:, {sensitivities}")
        print(f"Specificity per class:, {specificities}")
    
    confusion_matrix = cm
    cms.append(confusion_matrix)

    normalised_confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

    width = 6
    height = 6  
    axis_size = 10
    ticks_size = 8
    fontsize = 8
    title = "Confusion matrix of {} data. Balanced accuracy: {:.2f}%".format(split, balanced_accuracy_score*100)

    plt.figure(figsize=(width, height))
    plt.imshow(normalised_confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)  # You can switch to grey if needed

    thresh = confusion_matrix.max() * 0.5

    for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
        plt.text(j, i, format(confusion_matrix[i, j]),
                    horizontalalignment="center",
                    color="white" if i == j else "black",
                    fontsize=fontsize)

    tick_marks = np.arange(confusion_matrix.shape[0])
    plt.xticks(tick_marks, LABELS, rotation=45, fontsize=ticks_size)
    plt.yticks(tick_marks, LABELS, fontsize=ticks_size)
    plt.tight_layout()
    plt.ylabel('True label', fontsize=axis_size)
    plt.xlabel('Predicted label', fontsize=axis_size)
    plt.title(title, fontsize=axis_size)
    plt.subplots_adjust(left=0.15, bottom=0.12, right=0.95, top=0.88)

    
    plt.savefig(subdirectory + '/CM_' + split + '_' + dataset+'_'+time_+'.png')
    

    path_fig = subdirectory + '/CMn_' + split + '_' + dataset+'_'+time_+'.png'
    normalised_confusion_matrix = normalised_confusion_matrix * 100

    plt.figure(figsize=(width, height))
    plt.imshow(normalised_confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)  # You can switch to grey if needed

    thresh = normalised_confusion_matrix.max() * 0.5

    for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
        plt.text(j, i, f"{normalised_confusion_matrix[i, j]:.2f}",
                    horizontalalignment="center",
                    color="white" if i == j else "black",
                    fontsize=fontsize)

    tick_marks = np.arange(confusion_matrix.shape[0])
    plt.xticks(tick_marks, LABELS, rotation=45, fontsize=ticks_size)
    plt.yticks(tick_marks, LABELS, fontsize=ticks_size)
    plt.tight_layout()
    plt.ylabel('True label', fontsize=axis_size)
    plt.xlabel('Predicted label', fontsize=axis_size)
    plt.title(title, fontsize=axis_size)
    plt.subplots_adjust(left=0.15, bottom=0.12, right=0.95, top=0.88)

    plt.savefig(path_fig)

    
    print("ROC curve")

    y_preds = one_hot_predictions
    y_test_np = y_test #.numpy()
    y_preds_np = y_preds #.numpy()


    n_classes = 3
    y_test_bin = label_binarize(y_test_np, classes=[0, 1, 2])
    y_preds_bin = label_binarize(y_preds_np, classes=[0, 1, 2])

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_preds_bin[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    colors = ['blue', 'red', 'green']
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
                    label='ROC curve (area = {0:0.2f}) for class {1}'.format(roc_auc[i], i+1))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve of proposed model')
    plt.legend(loc="lower right")
    plt.subplots_adjust(left=0.25, bottom=0.15, right=0.9, top=0.85)

    path_fig = subdirectory + '/roc_' + dataset+'_'+time_+'.png'
    plt.savefig(path_fig)


    from sklearn.metrics import precision_recall_curve, average_precision_score


"""
Overlap function
"""
def overlap_class( new_data_class, samples, overlap_shift, window_size):
    init = 0
    n = 0
    while(init <= (samples.shape[0]-window_size)):
        new_data_class.append(samples[init:init+window_size])
        init = init + int(window_size - (window_size * overlap_shift))
        n = n + 1
    return new_data_class, n

"""
Overlap intra-class and intra-user
"""
def overlap_data(data, labels, users_data, shift=0.5): #(408620, 100, 6) (408620,)
    
    classes = np.unique(labels)
    indexes = {}
    indexes_users = {}
    users_unique = np.unique(users_data)
    users = np.array(users_data)

    for u in users_unique:
        indexes_users[u] = np.where(users == u )[0]
    for c in classes:
        indexes[c] = np.where(labels == c )[0]          

    new_data_class = []
    new_labels = []
    new_users = []
    for u in indexes_users.keys():
        for c in indexes.keys():
            index = np.intersect1d(indexes_users[u], indexes[c])
            samples = data[index]
            samples = samples.reshape(samples.shape[0]*samples.shape[1], samples.shape[2])  # (N, 9)
            new_data_class, n = overlap_class(new_data_class, samples, shift, window_size=data.shape[1])
            new_labels = new_labels + list(np.repeat(c, n))
            new_users = new_users + list(np.repeat(u, n))

    new_labels = np.array(new_labels)
    new_users = np.array(new_users)
    new_data_class = np.array(new_data_class)

    return new_data_class, new_labels, new_users

"""
Normalize the training, validation, and test data using StandardScaler fitted on training data.
"""
def normalise_data(dict_arrays):
    train_data = dict_arrays['x_train']
    val_data = dict_arrays['x_val']
    test_data = dict_arrays['x_test']

    scaler = StandardScaler()

    train_data_flat = train_data.reshape(-1, 9)  # Flatten to (num_samples*timesteps, 9)
    scaler.fit(train_data_flat)

    dict_arrays['x_train'] = scaler.transform(train_data_flat).reshape(train_data.shape)
    dict_arrays['x_val'] = scaler.transform(val_data.reshape(-1, 9)).reshape(val_data.shape)
    dict_arrays['x_test'] = scaler.transform(test_data.reshape(-1, 9)).reshape(test_data.shape)

    return dict_arrays

"""
Apply z-score normalization per axis group (acc_x, acc_y, ..., mag_z)
on a wide dataframe with repeated sensor axis order.
"""
def zscore_per_axis(df):

    n_axes = 9  # acc_x ... mag_z
    n_repeats = df.shape[1] // n_axes
    
    normalized = df.copy()
    axis_names = ["acc_x","acc_y","acc_z","gyr_x","gyr_y","gyr_z","mag_x","mag_y","mag_z"]
    
    for i, axis in enumerate(axis_names):
        cols = [j for j in range(i, df.shape[1], n_axes)]
        
        vals = df.iloc[:, cols]
        
        mean = vals.values.mean()
        std  = vals.values.std()
        normalized.iloc[:, cols] = (vals - mean) / std
    
    return normalized

"""
Read the full raw data from a CSV file and return it as a DataFrame.
"""
def get_processed_fold(df, dataset, modeltype, subdirectory, sensors, fold, seg5, normalize, overlap, overlap_shift, feat=9):
    file_users_split = subdirectory+'/'+dataset+'_'+str(fold)+'.txt'

    if ( (dataset.startswith('de')) & (normalize == True)):
        print("Normalizing ...")
        df_info = df.iloc[:,:5]
        df_data = df.iloc[:,5:]
        df_data = zscore_per_axis(df_data)

        df = pd.concat([df_info, df_data], axis=1)


    x_train, y_train, x_val, y_val, x_test, y_test, y_train_all, y_val_all, y_test_all, data_train, data_val, data_test, uuid_train, uuid_val, uuid_test, splits_txt = split_data_val(df, 
                                                                                                                                                  dataset, 
                                                                                                                                                  file_users_split, 
                                                                                                                                                  test_size=0.3)
    if (sensors == 2):
        dict_arrays = get_data_arrays(x_train, y_train, x_val, y_val, x_test, y_test, dataset, False)
    elif (sensors == 3):
        dict_arrays, users_train, users_val, users_test = get_data_arrays_3sns(x_train, y_train, y_train_all, data_train.iloc[:,0], 
                                                                x_val, y_val, y_val_all, data_val.iloc[:,0], 
                                                                x_test, y_test, y_test_all, data_test.iloc[:,0], 
                                                                dataset, seg5, feat=feat, shuffle_data = False, axis=True)
        
    else:
        print("No sensors:")
        exit()


    train_X = dict_arrays['x_train']
    train_Y = dict_arrays['y_train']
    val_X = dict_arrays['x_val']
    val_Y = dict_arrays['y_val']
    test_X = dict_arrays['x_test']
    test_Y = dict_arrays['y_test']
    y_test_all = dict_arrays['y_test_all']
    y_train_all = dict_arrays['y_train_all']
    y_val_all = dict_arrays['y_val_all']

    train_X = np.nan_to_num(train_X, nan=0.0, posinf=0.0, neginf=0.0)
    train_Y = np.nan_to_num(train_Y, nan=0.0, posinf=0.0, neginf=0.0)
    val_X = np.nan_to_num(val_X, nan=0.0, posinf=0.0, neginf=0.0)
    val_Y = np.nan_to_num(val_Y, nan=0.0, posinf=0.0, neginf=0.0)
    test_X = np.nan_to_num(test_X, nan=0.0, posinf=0.0, neginf=0.0)
    test_Y = np.nan_to_num(test_Y, nan=0.0, posinf=0.0, neginf=0.0)

    y_test_all = np.nan_to_num(y_test_all, nan=0.0, posinf=0.0, neginf=0.0)
    y_val_all = np.nan_to_num(y_val_all, nan=0.0, posinf=0.0, neginf=0.0)
    y_train_all = np.nan_to_num(y_train_all, nan=0.0, posinf=0.0, neginf=0.0)

    print("Shapes:")
    print(train_X.shape, train_Y.shape)
    print(val_X.shape, val_Y.shape)
    print(test_X.shape, test_Y.shape)

    if ( overlap ):
        if (dataset.startswith('vivabem')):
            train_X, train_Y, train_users = overlap_data(train_X, train_Y, users_train, overlap_shift)
            val_X, val_Y, val_users = overlap_data(val_X, val_Y, users_val, overlap_shift)
            test_X, test_Y, test_users = overlap_data(test_X, test_Y, users_test, overlap_shift)
        
        elif (dataset.startswith('eat')):
            train_X, y_train_all, train_users = overlap_data(train_X, y_train_all, users_train, overlap_shift)
            val_X, y_val_all, val_users = overlap_data(val_X, y_val_all, users_val, overlap_shift)
            test_X, y_test_all, test_users = overlap_data(test_X, y_test_all, users_test, overlap_shift)

            test_Y = y_test_all.copy()
            val_Y = y_val_all.copy()
            train_Y = y_train_all.copy()

            test_Y = np.where(test_Y == 2, 0, np.where(test_Y == 4, 1, 2))
            val_Y = np.where(val_Y == 2, 0, np.where(val_Y == 4, 1, 2))
            train_Y = np.where(train_Y == 2, 0, np.where(train_Y == 4, 1, 2))
        
        print('Fin overlap 1')

    dict_arrays['x_train'] = train_X
    dict_arrays['y_train'] = train_Y
    dict_arrays['x_val'] = val_X
    dict_arrays['y_val'] = val_Y
    dict_arrays['x_test'] = test_X
    dict_arrays['y_test'] = test_Y
    dict_arrays['uuid_train'] = uuid_train
    dict_arrays['uuid_val'] = uuid_val
    dict_arrays['uuid_test'] = uuid_test

    
    return dict_arrays

"""
Get processed fold data for unsupervised learning, returning only the input vector for the model.
"""
def get_processed_fold_unsup(df, dataset, modeltype, subdirectory, sensors, fold, seg5, normalize, overlap, overlap_shift, feat=9):
    
    file_users_split = subdirectory+'/'+dataset+'_'+str(fold)+'.txt'
    df_data = df

    if ( (dataset.startswith('de')) & (normalize == True)):
        print("Normalizing ...")
        df_data = df.iloc[:,5:]
        df_data = zscore_per_axis(df_data)

        df_data = df_data.reset_index(drop=True)
        df_data = np.array(df_data).astype(np.float32)


    df_data =  df_data.astype(np.float32)
    df_data = df_data.reshape((df_data.shape[0], df_data.shape[1], 1))


    df_data = np.nan_to_num(df_data, nan=0.0, posinf=0.0, neginf=0.0)

    feat = 9
    d2 = df_data.shape[-2] // feat
    df_data = np.reshape(df_data, (-1, d2, feat))
    print(df_data.shape)
    
    return df_data


"""
Get an average confusion matrix
"""
def normalize_cm(cm, LABELS, 
                    title=None,
                    cmap=plt.cm.Blues):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(10,10))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=LABELS, yticklabels=LABELS,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()    
    
    return plt

"""
Get processed fixed data for unsupervised learning, returning only the input vector for the model.
"""
def get_processed_fixed(df, dataset, subdirectory, sensors, fold, seg5, overlap_shift, fixed_splits=False):
    file_users_split = subdirectory+'/'+dataset+'_'+str(fold)+'.txt'
    dict_arrays = {}

    basepath = "/home/elian.riveros/dl-13-elian/notebooks/workspaces/lstm-har/gatednet_ax2Sns.worktrees/4cf329b/"
    train_X = np.load(f'{basepath}dataset/eatdrinkanother_all/train_eatdrinkother_norm.npy', allow_pickle=True)
    val_X = np.load(f'{basepath}dataset/eatdrinkanother_all/val_eatdrinkother_norm.npy', allow_pickle=True)
    test_X = np.load(f'{basepath}dataset/eatdrinkanother_all/test_eatdrinkother_norm.npy', allow_pickle=True)

    train_Y = np.load(f'{basepath}dataset/eatdrinkanother_all/eatdrinkother_train_y.npy', allow_pickle=True)
    val_Y = np.load(f'{basepath}dataset/eatdrinkanother_all/eatdrinkother_val_y.npy', allow_pickle=True)
    test_Y = np.load(f'{basepath}dataset/eatdrinkanother_all/eatdrinkother_test_y.npy', allow_pickle=True)

    train_X = np.reshape(train_X, (-1, 500, 9))
    val_X = np.reshape(val_X, (-1, 500, 9))
    test_X = np.reshape(test_X, (-1, 500, 9))
    
    users_train = np.load('/home/elian.riveros/dl-13-elian/notebooks/workspaces/eatdrinkanother_data/users_train.npy', allow_pickle=True)
    users_val = np.load('/home/elian.riveros/dl-13-elian/notebooks/workspaces/eatdrinkanother_data/users_val.npy', allow_pickle=True)
    users_test = np.load('/home/elian.riveros/dl-13-elian/notebooks/workspaces/eatdrinkanother_data/users_test.npy', allow_pickle=True)

    uuid_train = np.unique(users_train)
    uuid_val = np.unique(users_val)
    uuid_test = np.unique(users_test)

    train_X = train_X.astype(np.float32)
    train_Y = train_Y.astype(np.float32)
    val_X = val_X.astype(np.float32)
    val_Y = val_Y.astype(np.float32)
    test_X = test_X.astype(np.float32)
    test_Y = test_Y.astype(np.float32)

    train_X = np.nan_to_num(train_X, nan=0.0, posinf=0.0, neginf=0.0)
    train_Y = np.nan_to_num(train_Y, nan=0.0, posinf=0.0, neginf=0.0)
    val_X = np.nan_to_num(val_X, nan=0.0, posinf=0.0, neginf=0.0)
    val_Y = np.nan_to_num(val_Y, nan=0.0, posinf=0.0, neginf=0.0)
    test_X = np.nan_to_num(test_X, nan=0.0, posinf=0.0, neginf=0.0)
    test_Y = np.nan_to_num(test_Y, nan=0.0, posinf=0.0, neginf=0.0)

    train_X, train_Y = overlap_data(train_X, train_Y, users_train, overlap_shift)
    val_X, val_Y = overlap_data(val_X, val_Y, users_val, overlap_shift)
    test_X, test_Y = overlap_data(test_X, test_Y, users_test,overlap_shift)
    
    if ( sensors == 2):

        train_Y = torch.FloatTensor(train_Y.squeeze())

        val_Y = torch.FloatTensor(val_Y.squeeze())

        test_Y = torch.FloatTensor(test_Y.squeeze())
        
        train_accX = train_X[:,:,0:1]
        train_accY = train_X[:,:,1:2]
        train_accZ = train_X[:,:,2:3]
        train_gyrX = train_X[:,:,3:4]
        train_gyrY = train_X[:,:,4:5]
        train_gyrZ = train_X[:,:,5:6]

        val_accX = val_X[:,:,0:1]
        val_accY = val_X[:,:,1:2]
        val_accZ = val_X[:,:,2:3]
        val_gyrX = val_X[:,:,3:4]
        val_gyrY = val_X[:,:,4:5]
        val_gyrZ = val_X[:,:,5:6]

        test_accX = test_X[:,:,0:1]
        test_accY = test_X[:,:,1:2]
        test_accZ = test_X[:,:,2:3]
        test_gyrX = test_X[:,:,3:4]
        test_gyrY = test_X[:,:,4:5]
        test_gyrZ = test_X[:,:,5:6]

        train_X = np.concatenate((train_accX, train_accY, train_accZ, train_gyrX, train_gyrY, train_gyrZ), axis=2)
        val_X = np.concatenate((val_accX, val_accY, val_accZ, val_gyrX, val_gyrY, val_gyrZ), axis=2)
        test_X = np.concatenate((test_accX, test_accY, test_accZ, test_gyrX, test_gyrY, test_gyrZ), axis=2)


        print(" 2 sensors")
        print("Shapes:")
        print(train_X.shape, train_Y.shape)
        print(val_X.shape, val_Y.shape)
        print(test_X.shape, test_Y.shape)
        exit()

    elif ( sensors == 1):

        train_Y = torch.FloatTensor(train_Y.squeeze())

        val_Y = torch.FloatTensor(val_Y.squeeze())

        test_Y = torch.FloatTensor(test_Y.squeeze())
        
        train_accX = train_X[:,:,0:1]
        train_accY = train_X[:,:,1:2]
        train_accZ = train_X[:,:,2:3]

        val_accX = val_X[:,:,0:1]
        val_accY = val_X[:,:,1:2]
        val_accZ = val_X[:,:,2:3]

        test_accX = test_X[:,:,0:1]
        test_accY = test_X[:,:,1:2]
        test_accZ = test_X[:,:,2:3]

        train_X = np.concatenate((train_accX, train_accY, train_accZ), axis=2)
        val_X = np.concatenate((val_accX, val_accY, val_accZ), axis=2)
        test_X = np.concatenate((test_accX, test_accY, test_accZ), axis=2)
    
    elif ( sensors == 0): # Gyroscope sensor

        train_Y = torch.FloatTensor(train_Y.squeeze())

        val_Y = torch.FloatTensor(val_Y.squeeze())

        test_Y = torch.FloatTensor(test_Y.squeeze())
        
        train_gyrX = train_X[:,:,3:4]
        train_gyrY = train_X[:,:,4:5]
        train_gyrZ = train_X[:,:,5:6]

        val_gyrX = val_X[:,:,3:4]
        val_gyrY = val_X[:,:,4:5]
        val_gyrZ = val_X[:,:,5:6]

        test_gyrX = test_X[:,:,3:4]
        test_gyrY = test_X[:,:,4:5]
        test_gyrZ = test_X[:,:,5:6]

        train_X = np.concatenate((train_gyrX, train_gyrY, train_gyrZ), axis=2)
        val_X = np.concatenate((val_gyrX, val_gyrY, val_gyrZ), axis=2)
        test_X = np.concatenate((test_gyrX, test_gyrY, test_gyrZ), axis=2)


        print(" 1 sensor gyr")
        print("Shapes:")
        print(train_X.shape, train_Y.shape)
        print(val_X.shape, val_Y.shape)
        print(test_X.shape, test_Y.shape)
        exit()
    
    elif ( sensors == -1): # Accelerometer sensor

        train_Y = torch.FloatTensor(train_Y.squeeze())

        val_Y = torch.FloatTensor(val_Y.squeeze())

        test_Y = torch.FloatTensor(test_Y.squeeze())
        
        train_accX = train_X[:,:,0:1]
        train_accY = train_X[:,:,1:2]
        train_accZ = train_X[:,:,2:3]

        val_accX = val_X[:,:,0:1]
        val_accY = val_X[:,:,1:2]
        val_accZ = val_X[:,:,2:3]

        test_accX = test_X[:,:,0:1]
        test_accY = test_X[:,:,1:2]
        test_accZ = test_X[:,:,2:3]

        train_X = np.concatenate((train_accX, train_accY, train_accZ), axis=2)
        val_X = np.concatenate((val_accX, val_accY, val_accZ), axis=2)
        test_X = np.concatenate((test_accX, test_accY, test_accZ), axis=2)


        print(" 1 sensor acc")
        print("Shapes:")
        print(train_X.shape, train_Y.shape)
        print(val_X.shape, val_Y.shape)
        print(test_X.shape, test_Y.shape)  
        exit()
    
    elif ( sensors == 3):

        train_Y = torch.FloatTensor(train_Y.squeeze())

        val_Y = torch.FloatTensor(val_Y.squeeze())

        test_Y = torch.FloatTensor(test_Y.squeeze())


        print(" 3 sensors")
        print("Shapes:")
        print(train_X.shape, train_Y.shape)
        print(val_X.shape, val_Y.shape)
        print(test_X.shape, test_Y.shape)

    print("Shapes:")
    print(train_X.shape, train_Y.shape)
    print(val_X.shape, val_Y.shape)
    print(test_X.shape, test_Y.shape)   
    
    dict_arrays['x_train'] = train_X
    dict_arrays['y_train'] = train_Y
    dict_arrays['x_val'] = val_X
    dict_arrays['y_val'] = val_Y
    dict_arrays['x_test'] = test_X
    dict_arrays['y_test'] = test_Y
    dict_arrays['uuid_train'] = uuid_train
    dict_arrays['uuid_val'] = uuid_val
    dict_arrays['uuid_test'] = uuid_test
    
    return dict_arrays

"""
Split the data into training, validation, and test sets based on user IDs, ensuring that the same users do not appear in different sets.
"""
def split_data_val(df,  dataset, file_users_split, test_size=0.2): # per users

    if (dataset=='cola2-agm'):
        df = df[ (df[0]!='S1003') & (df[0]!='S1011') ]
    elif  (dataset=='vivabem12_lnd_ma'):
        df = df[ (df[0]!='S1047') & (df[0]!='S1056') ]
    elif (dataset=='vivabem12_lnd_mb'):
        df = df[ (df[0]!='S1002') & (df[0]!='S1003') & (df[0]!='S1056') ]
    elif (dataset=='vivabem12_lnd_ma_mb'):
        df = df[ (df[0]!='S1002') & (df[0]!='S1003') & (df[0]!='S1056') ]
    
    num_activities = np.unique(df.iloc[:,2])
    dict2 = {}
    for i in range(len(num_activities)):
        dict2[num_activities[i]] = i
    df = df.replace({2:dict2})
    
    uuids = np.unique(df.iloc[:,0])

    uuid_train, uuid_test = train_test_split(uuids,
                                        test_size = test_size)

    data_train = df[df.iloc[:,0].isin(uuid_train)]
    data_test = df[df.iloc[:,0].isin(uuid_test)]

    uuids_train = np.unique(data_train.iloc[:,0])

    uuid_train, uuid_val = train_test_split(uuids_train,
                                        train_size = 0.86,
                                        test_size = 0.14)

        
    splits_txt = str(uuid_train) + "\n" + str(uuid_val) + "\n" + str(uuid_test) + "\n"

    data_train = df[df.iloc[:,0].isin(uuid_train)]
    data_val = df[df.iloc[:,0].isin(uuid_val)]

    with open(file_users_split, 'w') as f:
        f.write("uuid_train: " + str(uuid_train))
        f.write("activities: " + str(np.unique(data_train.iloc[:,2])) + "\n")

        f.write("uuid_val: " + str(uuid_val))
        f.write("activities: " + str(np.unique(data_train.iloc[:,2])) + "\n")

        f.write("uuid_test: " + str(uuid_test))
        f.write("activities: " + str(np.unique(data_train.iloc[:,2])) + "\n")


    data_train = data_train.reset_index(drop=True)
    data_val = data_val.reset_index(drop=True)
    data_test = data_test.reset_index(drop=True)

    
    

    x_train = data_train.iloc[:,5:]  #uuid(0), timestamp(1), act id(2), met id(3), met name(4), data(5), ...
    y_train = data_train.iloc[:,2]
    y_train_all_str = data_train.iloc[:,4]

    x_val = data_val.iloc[:, 5:]
    y_val = data_val.iloc[:,2]
    y_val_all_str = data_val.iloc[:,4]

    x_test = data_test.iloc[:, 5:]
    y_test = data_test.iloc[:,2]
    y_test_all_str = data_test.iloc[:,4]

    y_train.reset_index(drop = True, inplace = True)
    y_val.reset_index(drop = True, inplace = True)
    y_test.reset_index(drop = True, inplace = True)

    x_train = np.array(x_train).astype(np.float32)
    x_val = np.array(x_val).astype(np.float32)
    x_test = np.array(x_test).astype(np.float32)

    y_train =  np.array(y_train).astype(np.float32)
    y_val  = np.array(y_val).astype(np.float32)
    y_test  = np.array(y_test).astype(np.float32)


    y_train = y_train.squeeze()
    y_val = y_val.squeeze()
    y_test = y_test.squeeze()

    string_to_id = {string: idx for idx, string in enumerate(np.unique(y_test_all_str))}
    y_test_all = [string_to_id[string] for string in y_test_all_str]

    string_to_id = {string: idx for idx, string in enumerate(np.unique(y_val_all_str))}
    y_val_all = [string_to_id[string] for string in y_val_all_str]

    string_to_id = {string: idx for idx, string in enumerate(np.unique(y_train_all_str))}
    y_train_all = [string_to_id[string] for string in y_train_all_str]

    y_train_all =  np.array(y_train_all).astype(np.float32)
    y_val_all  = np.array(y_val_all).astype(np.float32)
    y_test_all  = np.array(y_test_all).astype(np.float32)


    return x_train, y_train, x_val, y_val, x_test, y_test, y_train_all, y_val_all, y_test_all, data_train, data_val, data_test, uuid_train, uuid_val, uuid_test, splits_txt


"""
Resize dataset to 3-dimensional vectors for model input.
"""
def resize_data(dict_arrays):
    x_train = dict_arrays['x_train']
    y_train = dict_arrays['y_train']
    x_val = dict_arrays['x_val']
    y_val = dict_arrays['y_val']
    x_test = dict_arrays['x_test']
    y_test = dict_arrays['y_test']

    x_train = np.reshape(x_train, (-1, 100, 6))
    x_val = np.reshape(x_val, (-1, 100, 6))
    x_test = np.reshape(x_test, (-1, 100, 6))
    print(x_train.shape)
    print(x_val.shape)
    print(x_test.shape)

    y_train = resize_y(y_train)
    y_val = resize_y(y_val)
    y_test = resize_y(y_test)

    dict_arrays = {'x_train': x_train, 'y_train': y_train, 
                'x_val': x_val, 'y_val': y_val, 
                'x_test': x_test, 'y_test':y_test}
                
    return dict_arrays


"""
Resize dataset for 3 sensors to 3-dimensional vectors.
"""
def resize_data_3sns(dict_arrays):
    x_train = dict_arrays['x_train']
    y_train = dict_arrays['y_train']
    x_val = dict_arrays['x_val']
    y_val = dict_arrays['y_val']
    x_test = dict_arrays['x_test']
    y_test = dict_arrays['y_test']

    x_train = np.reshape(x_train, (-1, 100, 9))
    x_val = np.reshape(x_val, (-1, 100, 9))
    x_test = np.reshape(x_test, (-1, 100, 9))
    print(x_train.shape)
    print(x_val.shape)
    print(x_test.shape)

    y_train = resize_y(y_train)
    y_val = resize_y(y_val)
    y_test = resize_y(y_test)

    dict_arrays = {'x_train': x_train, 'y_train': y_train, 
                'x_val': x_val, 'y_val': y_val, 
                'x_test': x_test, 'y_test':y_test}
                
    return dict_arrays


"""
Resize dataset for 3 sensors to 3-dimensional vectors.
"""
def resize_y(y_):
    c = np.ones([y_.shape[0], 5])
    c[:,0] = c[:,0] * y_

    c[:,1] = c[:,1] * y_
    c[:,2] = c[:,2] * y_
    c[:,3] = c[:,3] * y_
    c[:,4] = c[:,4] * y_

    c = c.reshape([y_.shape[0]*5, 1])
    c = c.squeeze() #//c.squeeze()
    return c
    
"""
Read full raw data from a CSV file.
"""
def read_full_raws(saved_file, dataset_name=None): 
    
    print("\nReading full_raws from file: ", saved_file)
    full_raws = pd.read_csv(saved_file, header=0)
    
    if dataset_name.startswith('de_fake_padts'):
        full_raws.columns = range(full_raws.shape[1])
        full_raws = full_raws.iloc[1:].reset_index(drop=True)

    print("full_raws shape:", full_raws.shape)

    return full_raws

"""
Save split data to CSV files.
"""
def save_csv_splitdata(data_train, data_val, data_test, file_data_train, file_data_test, file_data_val):
	print()
	print("Saving split data...")
	data_train.to_csv(file_data_train, header=None, index=False)
	data_val.to_csv(file_data_val, header=None, index=False)
	data_test.to_csv(file_data_test, header=None, index=False)
	print("shape data_train: ",data_train.shape)
	print("shape data_val: ",data_val.shape)
	print("shape data_test: ",data_test.shape)
	print()
	print("Split data saved in: ")
	print(file_data_train)
	print(file_data_val)
	print(file_data_test)

"""
Resize dataset for 3 sensors to 3-dimensional vectors.
"""
def resize_data_axis3sns(dict_arrays):
    x_train = dict_arrays['x_train']
    y_train = dict_arrays['y_train']
    y_train_all = dict_arrays['y_train_all']
    x_val = dict_arrays['x_val']
    y_val = dict_arrays['y_val']
    y_val_all = dict_arrays['y_val_all']
    x_test = dict_arrays['x_test']
    y_test = dict_arrays['y_test']
    y_test_all = dict_arrays['y_test_all']    

    x_train = np.reshape(x_train, (-1, 100, 9))
    x_val = np.reshape(x_val, (-1, 100, 9))
    x_test = np.reshape(x_test, (-1, 100, 9))

    print(x_train.shape)
    print(x_val.shape)
    print(x_test.shape)

    y_train = resize_y(y_train)
    y_val = resize_y(y_val)
    y_test = resize_y(y_test)

    y_train_all = resize_y(y_train_all)
    y_val_all = resize_y(y_val_all)
    y_test_all = resize_y(y_test_all)

    dict_arrays = {'x_train': x_train, 'y_train': y_train, 'y_train_all': y_train_all,
                'x_val': x_val, 'y_val': y_val, 'y_val_all': y_val_all,
                'x_test': x_test, 'y_test':y_test, 'y_test_all': y_test_all}
                
    return dict_arrays

"""
Resize dataset for 3 sensors to 3-dimensional vectors for windows of 5sec.
"""
def resize_data_axis3sns_5sec(dict_arrays, feat=9):
    x_train = dict_arrays['x_train']
    x_val = dict_arrays['x_val']
    x_test = dict_arrays['x_test']
    
    d2 = x_train.shape[-2] // feat
    if ( feat==18):
        d2 = 500
    x_train = np.reshape(x_train, (-1, d2, feat))
    x_val = np.reshape(x_val, (-1, d2, feat))
    x_test = np.reshape(x_test, (-1, d2, feat))

    dict_arrays['x_train'] = x_train
    dict_arrays['x_val'] = x_val
    dict_arrays['x_test'] = x_test
                
    return dict_arrays

        
"""
Divide the 'y' vector to 5 times the size. For windows of 5sec.
"""

def resize_users_lst(u_):
       
    c = np.empty([u_.shape[0], 5], dtype='object')
    c[:,0] = u_

    c[:,1] = u_
    c[:,2] = u_
    c[:,3] = u_
    c[:,4] = u_

    c = c.reshape([u_.shape[0]*5, 1])
    c = c.squeeze()
    return c

'''
Get data arrays and reshape to 1sec with 3 sensors
'''
def get_data_arrays_3sns(x_train, y_train, y_train_all, users_train, 
                        x_valid, y_valid, y_val_all, users_val, 
                        x_test, y_test, y_test_all, users_test, 
                        dataset, seg5, feat=9, shuffle_data=False, axis=False):
    
    x_train = x_train.astype(np.float32)
    x_valid = x_valid.astype(np.float32)
    x_test = x_test.astype(np.float32)

    y_train =  y_train.astype(np.float32)
    y_valid  = y_valid.astype(np.float32)
    y_test =  y_test.astype(np.float32)

    y_test_all =  y_test_all.astype(np.float32)
    y_val_all =  y_val_all.astype(np.float32)
    y_train_all =  y_train_all.astype(np.float32)
    
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_valid = x_valid.reshape((x_valid.shape[0], x_valid.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    users_train = np.array(users_train)
    users_val = np.array(users_val)
    users_test = np.array(users_test)

    if shuffle_data == True:
        x_train , y_train = shuffle(x_train, y_train)
        x_valid , y_valid = shuffle(x_valid, y_valid)
        x_test , y_test = shuffle(x_test, y_test)

    dict_arrays = {'x_train': x_train, 'y_train': y_train, 'y_train_all': y_train_all,
                'x_val': x_valid, 'y_val': y_valid, 'y_val_all': y_val_all,
                'x_test': x_test, 'y_test':y_test, 'y_test_all': y_test_all,}
    
    if seg5 == False:
        print("Resize to 1sec  ",dataset)
        if axis == True:
            dict_arrays = resize_data_axis3sns(dict_arrays)
        else:
            dict_arrays = resize_data_3sns(dict_arrays)

        users_train = resize_users_lst(users_train)
        users_val = resize_users_lst(users_val)
        users_test = resize_users_lst(users_test)
    
    else:
        dict_arrays = resize_data_axis3sns_5sec(dict_arrays, feat)

    return dict_arrays, users_train, users_val, users_test


"""
Get data arrays and reshape to 1sec with 3 sensors
"""
def get_data_arrays(x_train, y_train, x_valid, y_valid, x_test, y_test, dataset, magni):

    print("Resize to 1sec  ",dataset)
    x_train = x_train.astype(np.float32)
    x_valid = x_valid.astype(np.float32)
    x_test = x_test.astype(np.float32)

    y_train =  y_train.astype(np.float32)
    y_valid  = y_valid.astype(np.float32)
    y_test =  y_test.astype(np.float32)

    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_valid = x_valid.reshape((x_valid.shape[0], x_valid.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
    print(x_train.shape)
    print(x_valid.shape)
    print(x_test.shape)

    x_train , y_train = shuffle(x_train, y_train)
    x_valid , y_valid = shuffle(x_valid, y_valid)
    x_test , y_test = shuffle(x_test, y_test)

    n_classes = len(np.unique(y_train))
    print("No of classes:", n_classes)

    dict_arrays = {'x_train': x_train, 'y_train': y_train, 
                'x_val': x_valid, 'y_val': y_valid, 
                'x_test': x_test, 'y_test':y_test}
    
    dict_arrays = resize_data(dict_arrays)

    return dict_arrays


"""
Build metrics for training, validation, and test sets.
"""
def build_metrics(_model, dict_arrays):
    
    model = _model

    full_raws_train_fea_np = dict_arrays['x_train']
    full_raws_y_train = dict_arrays['y_train']
    full_raws_val_fea_np = dict_arrays['x_val']
    full_raws_y_val = dict_arrays['y_val']
    full_raws_test_fea_np = dict_arrays['x_test']
    full_raws_y_test = dict_arrays['y_test']

    preds_t = model.predict(full_raws_train_fea_np)
    preds_t_flat = np.argmax(preds_t, axis=1).reshape(-1)
    accuracy_train = float(np.sum(preds_t_flat==full_raws_y_train))/full_raws_y_train.shape[0]
    balanced_accuracy_train = balanced_accuracy_score(full_raws_y_train, preds_t_flat)
    f1_score_train_we = f1_score(full_raws_y_train,  preds_t_flat, average = 'weighted') #macro weighted
    f1_score_train_mic = f1_score(full_raws_y_train,  preds_t_flat, average = 'micro')
    kappa_train = cohen_kappa_score(full_raws_y_train,  preds_t_flat)

    preds_t_val = model.predict(full_raws_val_fea_np)
    preds_t_flat_val = np.argmax(preds_t_val, axis=1).reshape(-1)
    accuracy_val = float(np.sum(preds_t_flat_val==full_raws_y_val))/full_raws_y_val.shape[0]
    balanced_accuracy_val = balanced_accuracy_score(full_raws_y_val, preds_t_flat_val)
    f1_score_val_we = f1_score(full_raws_y_val,  preds_t_flat_val, average = 'weighted') #macro weighted
    f1_score_val_mic = f1_score(full_raws_y_val,  preds_t_flat_val, average = 'micro')
    kappa_val = cohen_kappa_score(full_raws_y_val,  preds_t_flat_val)

    preds_t_test = model.predict(full_raws_test_fea_np)
    preds_t_flat_test = np.argmax(preds_t_test, axis=1).reshape(-1)
    accuracy_test = float(np.sum(preds_t_flat_test==full_raws_y_test))/full_raws_y_test.shape[0]
    balanced_accuracy_test = balanced_accuracy_score(full_raws_y_test, preds_t_flat_test)
    f1_score_test_we = f1_score(full_raws_y_test,  preds_t_flat_test, average = 'weighted') #macro weighted
    f1_score_test_mic = f1_score(full_raws_y_test,  preds_t_flat_test, average = 'micro')
    kappa_test = cohen_kappa_score(full_raws_y_test,  preds_t_flat_test)

    return [[accuracy_train, balanced_accuracy_train, f1_score_train_we, f1_score_train_mic, kappa_train], 
            [accuracy_val, balanced_accuracy_val, f1_score_val_we, f1_score_val_mic, kappa_val],
             [accuracy_test, balanced_accuracy_test, f1_score_test_we, f1_score_test_mic, kappa_test]]

""" 
Plot normalized confusion matrix
"""
def plot_confusion_matrix( y_true, y_pred, LABELS,
                          normalize=True,
                          title=None,
                          path_save='cm.png',
                          cmap=plt.cm.Blues):

    print("\nCreating the confusion matrix ...")
    cm = metrics.confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(15,15))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=LABELS, yticklabels=LABELS,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()    
    plt.savefig(path_save)
    plt.close()

"""
Convert class labels to one-hot encoding.
"""
def one_hot(y_, n_classes=7):

    y_ = y_.reshape(len(y_))
    return np.eye(n_classes)[np.array(y_, dtype=np.int32)]  # Returns FLOATS

"""
Build a metrics table and save it as a CSV file.
"""
def build_metrics_table( dict_arrays, metric_results, table, time_, modeltype, dataset, last_epoch_early_stopping, learning_rate, dropout_rate, n_batch, LSTM_layers, lstm_hidden_units, lstm_reg, clf_reg, clipvalue, obs, subdirectory):
    new_row = { 'time' : time_,
                'modeltype' : modeltype,
                'dataset': dataset,
                'uuid_val': dict_arrays['uuid_val'],
                'uuid_test': dict_arrays['uuid_test'],
                'n_epochs': last_epoch_early_stopping, 
                'lr' : str(learning_rate),
                'do' : str(dropout_rate),
                'ov' : "",
                'batch' : str(n_batch),
                'layers' : str(LSTM_layers),
                'h_units' : str(lstm_hidden_units),
                'lstm_reg' : str(lstm_reg),
                'clf_reg' : str(clf_reg),
                'clipvalue' : str(clipvalue),
                'train': round(metric_results[0][0], 4),
                'train_bal' : round(metric_results[0][1], 4),
                'f1_score_train_we': round(metric_results[0][2], 4),
                'kappa_train': round(metric_results[0][4], 4),
                'val': round(metric_results[1][0], 4),
                'val_bal': round(metric_results[1][1], 4),
                'f1_score_val_we': round(metric_results[1][2], 4),
                'kappa_val': round(metric_results[1][4], 4),
                'test': round(metric_results[2][0], 4),
                'test_bal': round(metric_results[2][1], 4),
                'f1_score_test_we': round(metric_results[2][2], 4),
                'kappa_test': round(metric_results[2][4], 4),
                'obs' : obs,
                'path': subdirectory
            }

	
    print(new_row)
    table = table.append(new_row, ignore_index=True)
    table_name = subdirectory+"/table_"+dataset+"_"+time_+".csv"
    print("Saving table in...", table_name)
    table.to_csv(table_name, sep=',', encoding='utf-8', index=False)

"""
Compute various metrics such as precision, recall, F1-score, and mAP, and save them to a text file. Also, create and save confusion matrices.
"""
def compute_metrics(predictions, y_test, n_classes, LABELS, subdirectory, dataset, time_):
    local_time_ = time.strftime("%Y%m%d-%H%M%S")
    one_hot_predictions = predictions.argmax(1)
    precision = metrics.precision_score(y_test, one_hot_predictions, average="weighted", zero_division=0)
    recall = metrics.recall_score(y_test, one_hot_predictions, average="weighted")
    f1_score = metrics.f1_score(y_test, one_hot_predictions, average="weighted", zero_division=0)
    mAP=np.mean(np.asarray([(metrics.average_precision_score(one_hot(y_test, n_classes)[:,c], one_hot(one_hot_predictions, n_classes)[:,c], average="weighted")) for c in range(n_classes)]))
    with open(subdirectory+"/metrics_"+dataset+"_"+time_+"_"+local_time_+".txt", 'w') as file:
        file.write("Dataset: {}".format( dataset))
        file.write("\n")
        file.write("mAP score: {:.4f}\n".format(mAP))
        file.write("precision: {:.4f}\n".format(precision))
        file.write("recall: {:.4f}\n".format(recall))
        file.write("f1_score: {:.4f}".format(f1_score))
        file.write("\n")
        for c in range(n_classes):
            metr = metrics.average_precision_score(one_hot(y_test, n_classes)[:,c], one_hot(one_hot_predictions, n_classes)[:,c], average="weighted")
            metr = round(metr, 4)
            file.write("\n")
            file.write(str(LABELS[c])+"\t" + str(metr))
    
    confusion_matrix = metrics.confusion_matrix(y_test, one_hot_predictions)
    

    normalised_confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]


    width = 20
    height = 20

    plt.figure(figsize=(width, height))
    plt.imshow(
        normalised_confusion_matrix,
        interpolation='nearest',
        cmap=plt.cm.Blues # Blues or grey
    )
    thresh = confusion_matrix.max() *.5
    for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
        plt.text(j, i, format(confusion_matrix[i, j]),
                    horizontalalignment="center",
                    color="white" if (confusion_matrix[i, j] > thresh) and (confusion_matrix[i, j] != 0) else "black") # > if Blues, < if grey
    plt.title("Confusion matrix {} (F1={:.4f} - mAP={:.4f}) \n(normalised to % of total test data)".format(dataset, metrics.f1_score(y_test, one_hot_predictions, average="weighted"), mAP))
    plt.colorbar()
    tick_marks = np.arange(n_classes)
    plt.xticks(tick_marks, LABELS, rotation=45)
    plt.yticks(tick_marks, LABELS)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(subdirectory + '/CM_' + dataset+'_'+time_+'_'+local_time_+'.png')

    path_fig = subdirectory + '/CMn_' + dataset+'_'+time_+'_'+local_time_+'.png'
    plot_confusion_matrix( y_test, one_hot_predictions, LABELS,
                        normalize=True,
                        title="Normalized Confusion Matrix",
                        path_save = path_fig,
                        cmap=plt.cm.Blues)

    return confusion_matrix      


"""
Get class names for the dataset.
"""
def get_class_names(dataset):
    if  dataset == 'data_5':
        classes_name = np.array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
        classes_name = np.array(['CHAIR', 'DANCE', 'DRINK', 'DRY', 'EAT', 'PAPERS', 'STAIRS', 'STAND', 'SWEEP', 'TYPING', 'WALK'])

    return classes_name

    
"""
Get file paths for raw datasets based on the dataset name and magnification factor.
"""
def get_raw_datasets(dataset, magni):
    basename = "../../files/"

    if dataset=='data_5':
        file_full_raws = basename+"fullraws_vivabem12_5_nolytv_nowspr_noaer_acc_gyr_mag__coord__5sec_100hz_11act.csv"
        file_data_train = basename+".csv"
        file_data_val = basename+".csv"
        file_data_test = basename+".csv"
                
    return file_data_train, file_data_val, file_data_test, file_full_raws


"""
Plot training and validation loss and accuracy over epochs, and save the plot as a PNG file.
"""
def plot_loss_acc(epoch, epoch_accuracy, epoch_loss, epoch_validation, 
          epoch_lossval, directory, dataset, time_, plots_loss_dir):
    plt.figure()  
    plt.ylim(0, 2.0)
    plt.plot(epoch_accuracy.values(), 'r--')
    plt.plot(epoch_validation.values(), 'b--')
    plt.title(dataset + '- loss and accuracy')
    # summarize history for loss
    plt.plot(epoch_loss.values(), 'm-')
    plt.plot(epoch_lossval.values(), 'c-')
    plt.ylabel('loss & accuracy (--)')
    plt.xlabel('epoch')
    plt.legend(['train accuracy', 'val accuracy', 'train loss', 'val loss'], loc='upper right')
    plt.savefig(plots_loss_dir+"/"+str(epoch)+"_loss-acc_"+dataset+"-{}".format(time_)+'.png')
    plt.close()