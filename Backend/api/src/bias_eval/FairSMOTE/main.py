
from sklearn.preprocessing import MinMaxScaler
import time
import optuna
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import category_encoders as ce
import math
import time
import pandas as pd
import random,time,csv
import numpy as np
import math,copy,os
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import sklearn.metrics as metrics


import sys
sys.path.append(os.path.abspath('..'))

# ! pip install table_evaluator

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import category_encoders as ce
import math
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, auc, roc_auc_score, confusion_matrix
import os
from statistics import mean
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from datetime import datetime
import pprint
import json


from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, auc, roc_auc_score, confusion_matrix
import os
from statistics import mean
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer
import seaborn as sns
import matplotlib.pyplot as plt
from ctgan import CTGAN
import pdb
import unittest
import random
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors as NN
from xhtml2pdf import pisa



from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import StratifiedGroupKFold
from datetime import datetime, timedelta
import pprint
import json
import pandas as pd
from google.cloud import storage
# import datetime

def save_dataset_to_csv(dataset, file_path):
    """
    Save the dataset to a local CSV file.
    Args:
        dataset (pd.DataFrame): The dataset to save.
        file_path (str): The local path to save the CSV file.
    """
    dataset.to_csv(file_path, index=False)
    print(f"Dataset saved locally to {file_path}")


def upload_to_gcs(user_provided_url, local_file_path):
    """
    Upload a file to GCS and generate a signed URL for download.

    Args:
        user_provided_url (str): The GCS location provided by the user (e.g., gs://bucket_name/folder/filename.csv).
        local_file_path (str): The local path to the file to upload.

    Returns:
        str: A signed URL for the uploaded file, allowing the user to download it.
    """
    # Extract bucket and destination blob name from the GCS URL
    if user_provided_url.startswith("gs://"):
        gcs_parts = user_provided_url[5:].split("/", 1)
        bucket_name = gcs_parts[0]
        destination_blob_name = gcs_parts[1]
    else:
        raise ValueError("Invalid GCS URL. Must start with 'gs://'")

    # Initialize GCP storage client
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    # Upload the local file to GCS
    blob.upload_from_filename(local_file_path)
    print(f"File uploaded to {user_provided_url}")

    # Generate a signed URL valid for 24 hours
    download_url = f"The file was uploaded to this url: {user_provided_url}"
    return download_url


# def get_user_input_from_json(json_data):
#     try:
#         dataset_info = json_data.get('dataset', {})
#         columns_info = json_data.get('columns', {})
#         path = dataset_info.get('path', '')
#         PRIV_FEATURE = columns_info.get('PRIV_FEATURE', '')
#         PRIV_VALUE_FEATURE = columns_info.get('PRIV_VALUE_FEATURE', '')
#         target_column = columns_info.get('target', '')
#         orig_data = upload_csv_and_load_df(path)
#
#         unique_values = orig_data[target_column].unique()
#         # if not all(value in [0, 1] for value in unique_values):
#         #     raise ValueError(f"Target column '{target_column}' contains values other than 0 or 1: {unique_values}")
#
#         return target_column, orig_data, PRIV_FEATURE,PRIV_VALUE_FEATURE
#
#         target_column, orig_data, PRIV_FEATURE,PRIV_VALUE_FEATURE = get_user_input_from_json(json_input)
#
#     except KeyError as e:
#         raise KeyError(f"Missing key in input JSON: {e}")
#     except Exception as e:
#         raise Exception(f"Error in get_user_input_from_json: {e}")
#


def upload_csv_and_load_df(file_path):
    try:
        # Attempt to load the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        print(f'DataFrame loaded successfully with shape: {df.shape}')
        return df
    except Exception as e:
        print(f'Error loading the CSV file into a DataFrame: {str(e)}')
        return None


def print_metrics(y_gt, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_gt, y_pred).ravel()
    print('conf matrix:\n ', confusion_matrix(y_gt, y_pred) ,
          '\naccuracy: ', (tp+tn)/(tp+tn+fn+fp),
          'precision: ', tp/(tp+fp),
          'recall: ', tp/(tp+fn),
          'fpr: ', fp/(fp+tn),
          'tpr: ', tp /(tp + fn),
          'fn+fp', fn+fp)
    #returm fpr, tpr, accuracy
    return fp/(fp+tn), tp /(tp + fn), (tp+tn)/(tp+tn+fn+fp)

# bias metric - equal opportunity difference
def opportunity_diff_tpr(tpr1, tpr2):
    return abs(tpr1-tpr2)

# bias metric - equal opportunity difference
def opportunity_diff_fpr(fpr1, fpr2):
    return abs(fpr1-fpr2)

# bias metric - avarage absolute odds difference
def odds_diff(tpr1, tpr2, fpr1, fpr2):
    return 0.5*(abs(tpr2-tpr1) + abs(fpr2-fpr1))

# bias metric - statistical parity difference
def parity_diff(y_pred_0, y_pred_1):
    return abs(len(y_pred_0)/(len(y_pred_0)+len(y_pred_1)) - len(y_pred_1)/(len(y_pred_1)+len(y_pred_0)))


    # bias metric - Disparate impact
def disparate_impact(y_pred_0, y_pred_1):
    return 1
    #return min((len(y_pred_0)/len(y_pred_0+y_pred_1)) / (len(y_pred_1)/len(y_pred_1+len(y_pred_0)))
    #,(len(y_pred_1)/len(y_pred_0+y_pred_1)) / (len(y_pred_0)/len(y_pred_1+len(y_pred_0))))

def fairness_metrics_roc_auc_std(data , y):
    clf = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=1234)

    cv = StratifiedKFold(n_splits=5, random_state=1234, shuffle=True)
    results = pd.DataFrame(columns=['training_score', 'test_score'])

    fprs, tprs, scores, op_list ,op_f_list, od_list ,par_dif_list, acc_list = [], [], [], [],[], [], [],[]

    start_time = datetime.now()
    for (train, test), i in zip(cv.split(data, y), range(5)):
        clf.fit(data.iloc[train], y.iloc[train])
        _, _, auc_score_train = compute_roc_auc(train,clf, data, y)
        fpr, tpr, auc_score = compute_roc_auc(test,clf, data, y)
        scores.append((auc_score_train, auc_score))
        fprs.append(fpr)
        tprs.append(tpr)
        y_pred = clf.predict(data.iloc[test])
        o_p, o_p_fpr, o_d ,par_dif, acc_ = calc_bias(data.iloc[test],  y.iloc[test], PRIV_FEATURE,y_pred, 'subexperiment - baseline Ensemble'+' model:'+str(i))
        acc_list.append(acc_)
        op_list.append(o_p)
        op_f_list.append(o_p_fpr)
        od_list.append(o_d)
        par_dif_list.append(par_dif)

    std_op = np.std(op_list)
    op_str = f"{mean(op_list):.3f} ± {std_op:.3f}"

    std_op_f = np.std(op_f_list)
    op_f_str = f"{mean(op_f_list):.3f} ± {std_op_f:.3f}"

    std_od = np.std(od_list)
    od_str = f"{mean(od_list):.3f} ± {std_od:.3f}"

    std_par_dif = np.std(par_dif_list)
    par_dif_str = f"{mean(par_dif_list):.3f} ± {std_par_dif:.3f}"


    std_acc = np.std(acc_list)
    acc_str = f"{mean(acc_list):.3f} ± {std_acc:.3f}"
    time_elapsed_baseline = datetime.now() - start_time
    plot_roc_curve(fprs, tprs);
    pd.DataFrame(scores, columns=['AUC Train', 'AUC Test'])
    print('Accaracy list : ', acc_list)
    print('equal opportunity for tpr : ', op_list)
    print('equal opportunity for fpr : ',op_f_list)
    print('odds difference list : ' , od_list)
    print('The mean accuracy is ', acc_str)
    print('The mean equal opportunity for tpr: (close to 0)', op_str)
    print('The mean equal opportunity for fpr: (close to 0)', op_f_str)
    print('The mean odds difference: (close to 0)',od_str)
    print('The mean parity difference: (close to 0)', par_dif_str)

    return acc_str, op_str, op_f_str ,od_str, par_dif_str


def calc_bias(data,y, priv_feature,y_pred, experiment_text):
    data_full =  pd.concat([data, y], axis=1)
    data_full['y_pred'] = y_pred

    label = target_column
    print('performance for whole ds:'+ experiment_text)
    fpr, tpr, acc = print_metrics(data_full[target_column], data_full['y_pred'] )

    feature_cat_list = list(data[priv_feature].unique())
    #print(feature_cat_list)
    valid_cat0 = data_full[data_full[priv_feature]==feature_cat_list[0]]
    valid_cat1 = data_full[data_full[priv_feature]==feature_cat_list[1]]
    print('performance for 0 sub-group:')
    fpr_0, tpr_0 , acc_0 = print_metrics(valid_cat0[target_column], valid_cat0['y_pred'])
    print('performance for 1 sub-group:')

    fpr_1, tpr_1, acc_1 = print_metrics(valid_cat1[target_column], valid_cat1['y_pred'])
    op_diff = opportunity_diff_tpr(tpr_0, tpr_1)
    op_diff_fpr = opportunity_diff_fpr(fpr_0, fpr_1)
    od_diff = odds_diff(tpr_0, tpr_1, fpr_0, fpr_1)
    par_diff = parity_diff(valid_cat0[valid_cat0['y_pred']==1], valid_cat1[valid_cat1['y_pred']==1])
    # dis_impact = disparate_impact(valid_cat0[valid_cat0['y_pred']==1], valid_cat1[valid_cat1['y_pred']==1])
    print('bias metrics:')
    print('equal opportunity for tpr: (close to 0)', op_diff)
    print('equal opportunity for fpr: (close to 0)', op_diff_fpr)
    print('avarage absolute odds difference: (close to 0)',od_diff)
    print('statistical parity difference: (close to 0) ',par_diff)
    return (op_diff,op_diff_fpr, od_diff,par_diff,acc)


def fairness_metrics_roc_auc(data , y):
    clf = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=1234)

    cv = StratifiedKFold(n_splits=5, random_state=1234, shuffle=True)
    results = pd.DataFrame(columns=['training_score', 'test_score'])

    fprs, tprs, scores, op_list ,op_f_list, od_list ,par_dif_list, acc_list = [], [], [], [],[], [], [],[]

    start_time = datetime.now()
    for (train, test), i in zip(cv.split(data, y), range(5)):
        clf.fit(data.iloc[train], y.iloc[train])
        _, _, auc_score_train = compute_roc_auc(train,clf, data, y)
        fpr, tpr, auc_score = compute_roc_auc(test,clf, data, y)
        scores.append((auc_score_train, auc_score))
        fprs.append(fpr)
        tprs.append(tpr)
        y_pred = clf.predict(data.iloc[test])
        o_p, o_p_fpr, o_d ,par_dif, acc_ = calc_bias(data.iloc[test],  y.iloc[test], PRIV_FEATURE,y_pred, 'subexperiment - baseline Ensemble'+' model:'+str(i))
        acc_list.append(acc_)
        op_list.append(o_p)
        op_f_list.append(o_p_fpr)
        od_list.append(o_d)
        par_dif_list.append(par_dif)

    std_op = np.std(op_list)
    op_str = f"{mean(op_list):.3f} ± {std_op:.3f}"

    std_op_f = np.std(op_f_list)
    op_f_str = f"{mean(op_f_list):.3f} ± {std_op_f:.3f}"

    std_od = np.std(od_list)
    od_str = f"{mean(od_list):.3f} ± {std_od:.3f}"

    std_par_dif = np.std(par_dif_list)
    par_dif_str = f"{mean(par_dif_list):.3f} ± {std_par_dif:.3f}"


    std_acc = np.std(acc_list)
    acc_str = f"{mean(acc_list):.3f} ± {std_acc:.3f}"
    time_elapsed_baseline = datetime.now() - start_time
    plot_roc_curve(fprs, tprs);
    pd.DataFrame(scores, columns=['AUC Train', 'AUC Test'])
    print('Accaracy list : ', acc_list)
    print('equal opportunity for tpr : ', op_list)
    print('equal opportunity for fpr : ',op_f_list)
    print('odds difference list : ' , od_list)
    print('The mean accuracy is ', acc_str)
    print('The mean equal opportunity for tpr: (close to 0)', op_str)
    print('The mean equal opportunity for fpr: (close to 0)', op_f_str)
    print('The mean odds difference: (close to 0)',od_str)
    print('The mean parity difference: (close to 0)', par_dif_str)

    return acc_list, op_list, op_f_list ,od_list,par_dif_list

def fairness_metrics_roc_auc_no_print(data , y):
    clf = RandomForestClassifier(n_estimators=10, max_depth=5,random_state=1234)

    cv = StratifiedKFold(n_splits=5, random_state=1234, shuffle=True)
    results = pd.DataFrame(columns=['training_score', 'test_score'])

    fprs, tprs, scores, op_list ,op_f_list, od_list ,par_dif_list, acc_list = [], [], [], [],[], [], [],[]

    start_time = datetime.now()
    for (train, test), i in zip(cv.split(data, y), range(5)):
        clf.fit(data.iloc[train], y.iloc[train])
        _, _, auc_score_train = compute_roc_auc(train,clf, data, y)
        fpr, tpr, auc_score = compute_roc_auc(test,clf, data, y)
        scores.append((auc_score_train, auc_score))
        fprs.append(fpr)
        tprs.append(tpr)
        y_pred = clf.predict(data.iloc[test])
        o_p, o_p_fpr, o_d ,par_dif, acc_ = calc_bias(data.iloc[test],  y.iloc[test], PRIV_FEATURE,y_pred, 'subexperiment - baseline Ensemble'+' model:'+str(i))
        acc_list.append(acc_)
        op_list.append(o_p)
        op_f_list.append(o_p_fpr)
        od_list.append(o_d)
        par_dif_list.append(par_dif)

    return acc_list, op_list, op_f_list ,od_list,par_dif_list


def plot_roc_curve(fprs, tprs):
    """Plot the Receiver Operating Characteristic from a list
    of true positive rates and false positive rates."""

    # Initialize useful lists + the plot axes.
    tprs_interp = []
    aucs, acc = [], []
    mean_fpr = np.linspace(0, 1, 100)
    f, ax = plt.subplots(figsize=(14,10))

    # Plot ROC for each K-Fold + compute AUC scores.
    for i, (fpr, tpr) in enumerate(zip(fprs, tprs)):
        tprs_interp.append(np.interp(mean_fpr, fpr, tpr))
        tprs_interp[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        ax.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    # Plot the luck line.
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Luck', alpha=.8)

    # Plot the mean ROC.
    mean_tpr = np.mean(tprs_interp, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    # Plot the standard deviation around the mean ROC.
    std_tpr = np.std(tprs_interp, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic')
    ax.legend(loc="lower right")
    return (f, ax)

def compute_roc_auc(index,clf ,data, y):
    y_predict = clf.predict_proba(data.iloc[index])[:,1]
    fpr, tpr, thresholds = roc_curve(y.iloc[index], y_predict)
    auc_score = auc(fpr, tpr)
    return fpr, tpr, auc_score


def get_numeric_features(dataframe):
    numeric_features = []
    for column in dataframe.columns:
        if dataframe[column].dtype in ['int64', 'float64','int','uint8','int32']:  # Check if the datatype is numeric
            numeric_features.append(column)
    return numeric_features


def fairness_metrics_roc_auc_no_print(data , y):
    clf = RandomForestClassifier(n_estimators=10, max_depth=5,random_state=1234)

    cv = StratifiedKFold(n_splits=5, random_state=1234, shuffle=True)
    results = pd.DataFrame(columns=['training_score', 'test_score'])

    fprs, tprs, scores, op_list ,op_f_list, od_list ,par_dif_list, acc_list = [], [], [], [],[], [], [],[]

    start_time = datetime.now()
    for (train, test), i in zip(cv.split(data, y), range(5)):
        clf.fit(data.iloc[train], y.iloc[train])
        _, _, auc_score_train = compute_roc_auc(train,clf, data, y)
        fpr, tpr, auc_score = compute_roc_auc(test,clf, data, y)
        scores.append((auc_score_train, auc_score))
        fprs.append(fpr)
        tprs.append(tpr)
        y_pred = clf.predict(data.iloc[test])
        o_p, o_p_fpr, o_d ,par_dif, acc_ = calc_bias(data.iloc[test],  y.iloc[test], PRIV_FEATURE,y_pred, 'subexperiment - baseline Ensemble'+' model:'+str(i))
        acc_list.append(acc_)
        op_list.append(o_p)
        op_f_list.append(o_p_fpr)
        od_list.append(o_d)
        par_dif_list.append(par_dif)

    return acc_list, op_list, op_f_list ,od_list,par_dif_list


def get_ngbr(data_no_label, knn):
      rand_sample_idx = random.randint(0, len(data_no_label) - 1)
      rand_sample = data_no_label[rand_sample_idx]
      distance, ngbr = knn.kneighbors(rand_sample.reshape(1, -1))
      # while True:
      rand_ngbr_idx = random.randint(0, len(ngbr))
      #   if distance[rand_ngbr_idx] == 0:
      #     continue  # avoid the sample itself, where distance ==0
      #   else:
      return data_no_label[rand_ngbr_idx], rand_sample


def mitigate(dataframe, target_name, priv_feature, priv_value,download_url):
    def get_ngbr(df, knn):
        rand_sample_idx = random.randint(0, df.shape[0] - 1)
        parent_candidate = df.iloc[rand_sample_idx]
        ngbr = knn.kneighbors(parent_candidate.values.reshape(1, -1), 3, return_distance=False)
        candidate_1 = df.iloc[ngbr[0][0]]
        candidate_2 = df.iloc[ngbr[0][1]]
        candidate_3 = df.iloc[ngbr[0][2]]
        return parent_candidate, candidate_2, candidate_3

    def generate_samples(no_of_samples, df, feature_columns):
        total_data = df.values.tolist()
        knn = NN(n_neighbors=5, algorithm='auto').fit(df)

        for _ in range(no_of_samples):
            cr = 0.8
            f = 0.8
            parent_candidate, child_candidate_1, child_candidate_2 = get_ngbr(df, knn)
            new_candidate = []
            for key, value in parent_candidate.items():
                if isinstance(parent_candidate[key], bool):
                    new_candidate.append(
                        parent_candidate[key] if cr < random.random() else not parent_candidate[key])
                elif isinstance(parent_candidate[key], str):
                    new_candidate.append(
                        random.choice([parent_candidate[key], child_candidate_1[key], child_candidate_2[key]]))
                elif isinstance(parent_candidate[key], list):
                    temp_lst = []
                    for i, each in enumerate(parent_candidate[key]):
                        temp_lst.append(parent_candidate[key][i] if cr < random.random() else
                                        int(parent_candidate[key][i] +
                                            f * (child_candidate_1[key][i] - child_candidate_2[key][i])))
                    new_candidate.append(temp_lst)
                else:
                    new_candidate.append(
                        abs(parent_candidate[key] + f * (child_candidate_1[key] - child_candidate_2[key])))
            total_data.append(new_candidate)

        final_df = pd.DataFrame(total_data)

        # Rename columns based on feature_columns list
        if len(feature_columns) == len(final_df.columns):
            final_df = final_df.rename(columns=dict(zip(final_df.columns, feature_columns)), errors="raise")
        else:
            print("Error: The length of feature_columns does not match the number of columns in the DataFrame.")

        return final_df

    # json_input = {
    #     'dataset': {
    #         'path': r'C:\Users\groze\Documents\אוניברסיטה\תואר שני\מחברות למאמר\datasets\german credit\german_credit_data.csv'
    #     },
    #     'columns': {
    #         'target': 'Risk',
    #         'PRIV_FEATURE': 'Sex',
    #         'PRIV_VALUE_FEATURE': 1
    #     }
    # }
    global PRIV_FEATURE, PRIV_VALUE_FEATURE, target_column , orig_data
    orig_data = dataframe
    PRIV_FEATURE = priv_feature
    PRIV_VALUE_FEATURE = priv_value
    target_column = target_name
    try:

        # target_column, orig_data, PRIV_FEATURE,PRIV_VALUE_FEATURE = get_user_input_from_json(json_input)
        orig_data[PRIV_FEATURE] = orig_data[PRIV_FEATURE].apply(lambda x: 1 if x == PRIV_VALUE_FEATURE else 0)
        PRIV_VALUE_FEATURE = 1
        orig_data[PRIV_FEATURE] = orig_data[PRIV_FEATURE].astype(int)

        categorical_columns = orig_data.select_dtypes(include=['object', 'category']).columns.tolist()
        # rempove target column from categorical columns
        if target_column in categorical_columns:
            categorical_columns.remove(target_column)
        orig_data[target_column], _ = pd.factorize(orig_data[target_column])
        orig_data[target_column] = orig_data[target_column].astype(int)
        feature_columns = list(orig_data.columns)

        y_orig = orig_data[target_column]
        orig_data = orig_data.drop([target_column],axis=1)
        orig_data = pd.get_dummies(orig_data, columns=categorical_columns)
        print(orig_data.info())
        print(orig_data.columns)

        if categorical_columns == ['']:
            categorical_columns = list()
        y_orig.dropna(inplace=True)

        # Assuming df is your dataframe
        numeric_feature_columns = get_numeric_features(orig_data)
        print("Numeric Features:", numeric_feature_columns)

        orig_data = orig_data.loc[:, numeric_feature_columns]
        from sklearn.preprocessing import MinMaxScaler

        # dataset for training
        orig_data.dropna(thresh=len(numeric_feature_columns) / 2, inplace=True)
        orig_data.reset_index(drop=True, inplace=True)
        # y_orig = orig_data[TARGET_COL].to_frame()

        scaler = MinMaxScaler()
        orig_data = pd.DataFrame(scaler.fit_transform(orig_data), columns=orig_data.columns)
        orig_dataset = pd.concat([orig_data, y_orig], axis=1)
        # orig_dataset[target_column], _ = pd.factorize(orig_data[target_column])

        # Based on class
        orig_data_one, orig_data_zero = [x for _, x in orig_dataset.groupby(orig_dataset[target_column] == 0)]

        # Based on priv feature
        orig_data_one_priv, orig_data_one_non_priv = [x for _, x in orig_data_one.groupby(
            orig_data_one[PRIV_FEATURE] != PRIV_VALUE_FEATURE)]
        orig_data_zero_priv, orig_data_zero_non_priv = [x for _, x in orig_data_zero.groupby(
            orig_data_zero[PRIV_FEATURE] != PRIV_VALUE_FEATURE)]

        print(orig_data_one_priv.shape, orig_data_one_non_priv.shape, orig_data_zero_priv.shape,
              orig_data_zero_non_priv.shape)

        if (orig_data_one_priv.shape[0] / orig_data_one_non_priv.shape[0] ) >= (orig_data_zero_priv.shape[0] / orig_data_zero_non_priv.shape[0]) :
            PRIV_CLASS = 1
        else:
            PRIV_CLASS = 0

        maximum = max(orig_data_zero_priv.shape, orig_data_zero_non_priv.shape, orig_data_one_priv.shape,
                      orig_data_one_non_priv.shape)
        if maximum == orig_data_zero_priv.shape:
            print("orig_data_zero_priv is maximum")
        if maximum == orig_data_zero_non_priv.shape:
            print("orig_data_zero_non_priv is maximum")
        if maximum == orig_data_one_priv.shape:
            print("orig_data_one_priv is maximum")
        if maximum == orig_data_one_non_priv.shape:
            print("orig_data_one_non_priv is maximum")

        maximum = maximum[0]

        print(orig_data_one_priv.shape, orig_data_one_non_priv.shape, orig_data_zero_priv.shape,
              orig_data_zero_non_priv.shape)

        # Plot for orig_data_one_priv.shape[0] and orig_data_one_non_priv.shape[0]
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # Bar colors
        colors = ['blue', 'orange']

        # Plot 1 - orig_data_one_priv and orig_data_one_non_priv
        axs[0].bar(['priv', 'non_priv'], [orig_data_one_priv.shape[0], orig_data_one_non_priv.shape[0]], color=colors)
        axs[0].set_xlabel('Data Type')
        axs[0].set_ylabel('Count')
        axs[0].set_title('Label 1')

        # Plot 2 - orig_data_zero_priv and orig_data_zero_non_priv
        axs[1].bar(['priv', 'non_priv'], [orig_data_zero_priv.shape[0], orig_data_zero_non_priv.shape[0]], color=colors)
        axs[1].set_xlabel('Data Type')
        axs[1].set_ylabel('Count')
        axs[1].set_title('Label 0')

        # Adjust spacing between subplots
        plt.subplots_adjust(wspace=0.4)

        plt.savefig('plot_original_ratios.jpg')

        orig_data.reset_index(drop=True, inplace=True)
        orig_dataset.reset_index(drop=True, inplace=True)

        print(orig_data.columns)
        print(orig_dataset.columns)
        orig_acc, orig_op_diff, orig_op_diff_fpr, orig_od_diff, orig_par_diff = fairness_metrics_roc_auc(orig_data,
                                                                                                         y_orig)
        orig_acc_std, orig_op_diff_std, orig_op_diff_fpr_std, orig_od_diff_std, orig_par_diff_std = fairness_metrics_roc_auc_std(orig_data,y_orig)

        # Based on class
        orig_data_one, orig_data_zero = [x for _, x in orig_dataset.groupby(orig_dataset[target_column] == 0)]

        # Based on priv 1
        one_one_df, one_zero_df = [x for _, x in orig_data_one.groupby(orig_data_one[PRIV_FEATURE] == 1)]
        zero_one_df, zero_zero_df = [x for _, x in orig_data_zero.groupby(orig_data_zero[PRIV_FEATURE] == 1)]

        one_one = one_one_df.shape[0]
        one_zero = one_zero_df.shape[0]
        zero_one = zero_one_df.shape[0]
        zero_zero = zero_zero_df.shape[0]

        print(zero_zero, zero_one, one_zero, one_one)

        maximum = max(zero_zero, zero_one, one_zero, one_one)
        if maximum == zero_zero:
            print("zero_zero is maximum")
        if maximum == zero_one:
            print("zero_one is maximum")
        if maximum == one_zero:
            print("one_zero is maximum")
        if maximum == one_one:
            print("one_one is maximum")

        zero_zero_to_be_incresed = maximum - zero_zero  ## where both are 0
        one_zero_to_be_incresed = maximum - one_zero  ## where class is 1 attribute is 0
        zero_one_to_be_incresed = maximum - zero_one  ## where class is 1 attribute is 1
        one_one_to_be_incresed = maximum - one_one  ## where class is 1 attribute is 1

        print(zero_zero_to_be_incresed, one_zero_to_be_incresed, zero_one_to_be_incresed, one_one_to_be_incresed)

        zero_zero_df = generate_samples(zero_zero_to_be_incresed, zero_zero_df, zero_zero_df.columns)
        one_zero_df = generate_samples(one_zero_to_be_incresed, one_zero_df, one_zero_df.columns)
        zero_one_df = generate_samples(zero_one_to_be_incresed, zero_one_df, zero_one_df.columns)
        one_one_df = generate_samples(one_one_to_be_incresed, one_one_df, one_one_df.columns)

        print ('the len of one one df is: ',len(one_one_df))
        print ('the len of one one df is: ',len(one_zero_df))
        print ('the len of one one df is: ',len(zero_one_df))
        print ('the len of one one df is: ',len(one_one_df))

        Upsampled_Dataset = pd.concat([zero_zero_df, zero_one_df, one_zero_df, one_one_df], ignore_index=True)
        Upsampled_Dataset = Upsampled_Dataset.reset_index()
        print(Upsampled_Dataset.columns)
        # Upsampled_Dataset.to_csv('Upsampled_dataset.csv')
        csv_file_path = 'unsampled_dataset.csv'
        save_dataset_to_csv(Upsampled_Dataset, csv_file_path)
        if download_url is not None:
            save_download = upload_to_gcs(download_url, csv_file_path)
        else:
            save_download = None
        os.remove(csv_file_path)

        # Save the upsampled dataset to a JSON file
        # unsampled_json = Upsampled_Dataset.to_json( orient='records')
        unsampled_json_dict = json.loads(Upsampled_Dataset.to_json(orient='records'))

        Upsampled_Dataset[target_column] = Upsampled_Dataset[target_column].astype(int)
        y_upsampled = Upsampled_Dataset[target_column]
        Upsampled_df_features = Upsampled_Dataset.drop([target_column,'index'],axis=1)
        print(Upsampled_df_features.columns)


        upsampled_acc, upsampled_op_diff, upsampled_op_diff_fpr, upsampled_od_diff, upsampled_par_diff = fairness_metrics_roc_auc(
            Upsampled_df_features, y_upsampled)

        upsampled_acc_std, upsampled_op_diff_std, upsampled_op_diff_fpr_std, upsampled_od_diff_std, upsampled_par_diff_std = fairness_metrics_roc_auc_std(
            Upsampled_df_features, y_upsampled)

        # # Initialize wandb
        # wandb.init(project="couple-comparisons")

        # Create a bar plot for each couple comparison
        fig, ax = plt.subplots(2, 3, figsize=(20, 10), dpi=100)
        plt.subplots_adjust(hspace=0.5)
        plt.suptitle('Couple Comparisons')

        # Set background color
        fig.patch.set_facecolor('#f9f9f9')

        # Acc vs. Pois_acc plot
        ax[0, 0].bar(['orig_acc', 'debias_acc'], [mean(orig_acc), mean(upsampled_acc)], color=['#4e79a7', '#f28e2c'])
        ax[0, 0].set_title('Acc vs. debias_acc', fontsize=12, fontweight='bold')
        ax[0, 0].set_xlabel('Accuracy', fontsize=10)
        ax[0, 0].set_ylabel('Value', fontsize=10)
        ax[0, 0].tick_params(axis='both', which='major', labelsize=8)
        ax[0, 0].spines['top'].set_visible(False)
        ax[0, 0].spines['right'].set_visible(False)

        # Op vs. Pois_op plot
        ax[0, 1].bar(['orig_op', 'debias_op'], [mean(orig_op_diff), mean(upsampled_op_diff)], color=['#4e79a7', '#f28e2c'])
        ax[0, 1].set_title('Op vs. debias_op', fontsize=12, fontweight='bold')
        ax[0, 1].set_xlabel('Equal opportunity for TPR', fontsize=10)
        ax[0, 1].set_ylabel('Value', fontsize=10)
        ax[0, 1].tick_params(axis='both', which='major', labelsize=8)
        ax[0, 1].spines['top'].set_visible(False)
        ax[0, 1].spines['right'].set_visible(False)

        # Op_f vs. Pois_op_f plotdebias
        ax[1, 0].bar(['orig_op_diff_fpr', 'debias_op_diff_fpr'], [mean(upsampled_op_diff_fpr), mean(upsampled_op_diff_fpr)],
                     color=['#4e79a7', '#f28e2c'])
        ax[1, 0].set_title('Op_f vs. debias_op_f', fontsize=12, fontweight='bold')
        ax[1, 0].set_xlabel('Equal opportunity for FPR', fontsize=10)
        ax[1, 0].set_ylabel('Value', fontsize=10)
        ax[1, 0].tick_params(axis='both', which='major', labelsize=8)
        ax[1, 0].spines['top'].set_visible(False)
        ax[1, 0].spines['right'].set_visible(False)

        # Od vs. Pois_od plot
        ax[1, 1].bar(['orig_od_diff', 'debias_od_diff'], [mean(orig_od_diff), mean(upsampled_od_diff)],
                     color=['#4e79a7', '#f28e2c'])
        ax[1, 1].set_title('Od vs. debias_od', fontsize=12, fontweight='bold')
        ax[1, 1].set_xlabel('Avarage absolute odds difference', fontsize=10)
        ax[1, 1].set_ylabel('Value', fontsize=10)
        ax[1, 1].tick_params(axis='both', which='major', labelsize=8)
        ax[1, 1].spines['top'].set_visible(False)
        ax[1, 1].spines

        # Od vs. Pois_od plot
        ax[1, 2].bar(['orig_par_diff', 'debias_par_diff'], [mean(orig_par_diff), mean(upsampled_par_diff)],
                     color=['#4e79a7', '#f28e2c'])
        ax[1, 2].set_title('orig_par_diff vs. debias_par_diff', fontsize=12, fontweight='bold')
        ax[1, 2].set_xlabel('parity diff', fontsize=10)
        ax[1, 2].set_ylabel('Value', fontsize=10)
        ax[1, 2].tick_params(axis='both', which='major', labelsize=8)
        ax[1, 2].spines['top'].set_visible(False)
        ax[1, 2].spines

        fig.delaxes(ax[0, 2])
        plt.savefig('couple_comparisons.jpg')


        # Define paths to the images
        plot_original_ratios_path = '/content/plot_original_ratios.jpg'
        plot_upsampled_ratios_path = '/content/plot_upsampled_ratios.jpg'
        couple_comparisons_path = '/content/couple_comparisons.jpg'

        import base64

        # Load the image as binary data
        with open('plot_original_ratios.jpg', 'rb') as original_ratios_file:
            plot_original_ratios = original_ratios_file.read()
        os.remove('plot_original_ratios.jpg')

        # Load the image as binary data
        # with open('plot_upsampled_ratios.jpg', 'rb') as upsampled_ratios_file:
        #     plot_upsampled_ratios = upsampled_ratios_file.read()
        # os.remove('plot_upsampled_ratios.jpg')

        # Load the image as binary data
        with open('couple_comparisons.jpg', 'rb') as couple_comparisons_file:
            couple_comparisons = couple_comparisons_file.read()
        os.remove('couple_comparisons.jpg')

        # Convert the binary data to Base64
        plot_original_ratios_base64 = base64.b64encode(plot_original_ratios).decode('utf-8')
        # Convert the binary data to Base64
        # plot_upsampled_ratios_base64 = base64.b64encode(plot_upsampled_ratios).decode('utf-8')
        # Convert the binary data to Base64
        couple_comparisons_base64 = base64.b64encode(couple_comparisons).decode('utf-8')

        ORIG_SIZE = len(orig_data)
        UPSAMPLED_SIZE = len(Upsampled_Dataset)


        # Define the HTML content
        html_content = f'''
                <!DOCTYPE html>
                <html>
                <head>
                    <style>
                        body {{
                            font-family: Arial, sans-serif;
                            margin: 20px;
                        }}
                        h1, h2, h3 {{
                            color: #333;
                            border-bottom: 2px solid #333;
                            padding-bottom: 5px;
                        }}
                        table {{
                            width: 100%;
                            border-collapse: collapse;
                            margin-top: 10px;
                            border: 1px solid #ccc;
                        }}
                        th, td {{
                            padding: 10px;
                            text-align: left;
                            border-bottom: 1px solid #ccc;
                        }}
                        th {{
                            width: 30%;  /* Set the width for table headers */
                        }}
                        td {{
                            width: 70%;  /* Set the width for table cells */
                        }}
                        img {{
                            max-width: 100%;
                            height: auto;
                            border: 1px solid #ccc;
                            margin-top: 10px;
                        }}
                    </style>
                </head>
                <body>
                    <h1>Dataset Analysis Report Of Method - 'FairSMOTE'</h1>
                    <h3>Background Information on Metrics</h3>
                    <p>This report presents key metrics to evaluate the fairness and accuracy of the original dataset:</p>
                    <ul>
                        <li><strong>Accuracy:</strong> The proportion of correct predictions among the total predictions made. A higher accuracy indicates better overall performance.</li>
                        <li><strong>Equal Opportunity TPR Difference:</strong> This metric measures the difference in True Positive Rates (TPR) between the privileged and unprivileged groups. A value closer to 0 indicates better fairness, as it signifies that both groups have similar rates of positive predictions.</li>
                        <li><strong>Equal Opportunity FPR Difference:</strong> This metric evaluates the difference in False Positive Rates (FPR) between groups. A value closer to 0 is desirable, indicating that both groups experience similar rates of false positive predictions.</li>
                        <li><strong>Odds Difference:</strong> This metric reflects the difference in odds of being predicted as positive between groups. A smaller value indicates better fairness, suggesting that the model treats both groups similarly.</li>
                        <li><strong>Parity Difference:</strong> This metric represents the difference in the positive prediction rates between the privileged and unprivileged groups. A value closer to 0 indicates better fairness.</li>
                    </ul>

                    <h2>Original Dataset: ({ORIG_SIZE} rows)</h2>
                    <table>
                        <tr>
                            <th>Target Column</th>
                            <td>{target_column}</td>
                        </tr>
                        <tr>
                            <th>Privileged Feature</th>
                            <td>{PRIV_FEATURE}</td>
                        </tr>
                        <tr>
                            <th>Privileged Value</th>
                            <td>{priv_value}</td>
                        </tr>
                    </table>
                    <h3>Original Dataset Metrics:</h3>
                    <table>
                        <tr>
                            <th>Accuracy</th>
                            <td>{orig_acc_std}</td>
                        </tr>
                        <tr>
                            <th>Equal Opportunity TPR Difference</th>
                            <td>{orig_op_diff_std}</td>
                        </tr>
                        <tr>
                            <th>Equal Opportunity FPR Difference</th>
                            <td>{orig_op_diff_fpr_std}</td>
                        </tr>
                        <tr>
                            <th>Odds Difference</th>
                            <td>{orig_od_diff_std}</td>
                        </tr>
                        <tr>
                            <th>Parity Difference</th>
                            <td>{orig_par_diff_std}</td>
                        </tr>
                    </table>

                    <h2>Original Dataset Ratios</h2>
                    <img src="data:image/png;base64,{plot_original_ratios_base64}" alt="Original Dataset Ratios">

                    <h2>Upsampled Dataset: ({UPSAMPLED_SIZE} rows)</h2>
                    <table>
                        <tr>
                            <th>Accuracy</th>
                            <td>{upsampled_acc_std}</td>
                        </tr>
                        <tr>
                            <th>Equal Opportunity TPR Difference</th>
                            <td>{upsampled_op_diff_std}</td>
                        </tr>
                        <tr>
                            <th>Equal Opportunity FPR Difference</th>
                            <td>{upsampled_op_diff_fpr_std}</td>
                        </tr>
                        <tr>
                            <th>Odds Difference</th>
                            <td>{upsampled_od_diff_std}</td>
                        </tr>
                        <tr>
                            <th>Parity Difference</th>
                            <td>{upsampled_par_diff_std}</td>
                        </tr>
                    </table>

               
                    <h2>Comparison Between Original and Upsampled Datasets</h2>
                    <img src="data:image/png;base64,{couple_comparisons_base64}" alt="Comparison Between Original and Upsampled Datasets">

                    <h3>Choosing the Preferred Method</h3>
                    <p>When selecting the preferred method for bias mitigation, it is essential to balance fairness and accuracy. Here are some tips to consider:</p>
                    <ul>
                        <li><strong>If fairness is a priority:</strong> Focus on metrics like Equal Opportunity TPR Difference, Equal Opportunity FPR Difference, and Parity Difference. Choose a method that minimizes these differences, even if it means sacrificing some accuracy.</li>
                        <li><strong>If accuracy is a priority:</strong> Review the Accuracy metric and select a method that maintains high accuracy while ensuring acceptable fairness levels.</li>
                        <li><strong>Overall assessment:</strong> Consider both fairness metrics and accuracy collectively. A method that offers a reasonable compromise between the two may be the most appropriate choice for your application.</li>
                    </ul>
                </body>
                </html>
        '''

        statistics = {
            "Original": {
                "accuracy": orig_acc_std,
                "opportunity_difference": orig_op_diff_std,
                "opportunity_difference_fpr": orig_op_diff_fpr_std,
                "odds_difference": orig_od_diff_std,
                "parity_difference": orig_par_diff_std
            },
            "Upsampled": {
                "accuracy": upsampled_acc_std,
                "opportunity_difference": upsampled_op_diff_std,
                "opportunity_difference_fpr": upsampled_op_diff_fpr_std,
                "odds_difference": upsampled_od_diff_std,
                "parity_difference": upsampled_par_diff_std,
                "data" : save_download

            }
        }
        from io import BytesIO
        import base64

        pdf_bytes = BytesIO()

        # Create the PDF
        pisa_status = pisa.CreatePDF(html_content, dest=pdf_bytes)

        if pisa_status.err:
            print("An error occurred while converting HTML to PDF")
            return None, None
        else:
            pdf_data = pdf_bytes.getvalue()

            # Write the PDF to a file
            with open("report.pdf", "wb") as result_file:
                result_file.write(pdf_data)

            # Encode the PDF data to a base64 string
            base64_encoded_string = base64.b64encode(pdf_data).decode('utf-8')
            os.remove('report.pdf')
        # # Write the HTML content to a file
        # with open('report.html', 'w', encoding='utf-8') as f:
        #     f.write(html_content)
        #     print("Report saved as 'report.html'")
        #
        # with open('results.json', 'w') as f:
        #     json.dump(statistics, f, ensure_ascii=False)
        #     print("Report saved as 'report.json'")
        return statistics, base64_encoded_string
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None



