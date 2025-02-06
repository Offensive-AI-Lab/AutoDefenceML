# %%
import pickle
from importlib import import_module
import importlib.util
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import confusion_matrix

# %%
min_values_for_categorical = 10


# max to categorical 20
# number of bins 5-10
# %%


# %%
def get_categorical_columns(df, threshold):
    """
    Returns a list of categorical columns in the dataframe
    Args:
        df (pd.DataFrame): The dataframe to get the categorical columns from
        threshold (int): The threshold for the number of unique values in a column to be considered categorical if column dtype is not object
    return:
        list: A list of categorical columns
    """

    return [col for col in df.columns if df[col].dtype == 'object']


# %%
def change_columns_to_datetime(df):
    """
    change all datetime  columns in the dataframe from object  to datetime
    if the column is not datetime it will stay object
    Args:
        df (pd.DataFrame): The dataframe to change the columns to datetime
    return:
        pd.DataFrame: The dataframe with the columns changed to datetime
    """
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_datetime(df[col], dayfirst=True)
            except:
                pass
    return df


def create_one_hot_dataset(df, target, cat_threshold=min_values_for_categorical):
    """
    Creates a one hot encoded dataset from the given dataframe and categorical columns
    and a dict of the one hot encoded columns and their original columns
    Args:
        df (pd.DataFrame): The dataframe to create the one hot encoded dataset from
        target (str): The target column name
        cat_threshold (int): The threshold for the number of unique values in a column to be considered categorical if column dtype is not object
    return:
        pd.DataFrame: The one hot encoded dataset
        list: A list of the one hot encoded columns
        dict: one hot encoded column name:[original column, original value]

    """
    df = change_columns_to_datetime(df)

    categorical_columns = get_categorical_columns(df.drop(target, axis=1), cat_threshold)
    one_to_original = {}
    one_hot_encoded_columns = pd.get_dummies(df[categorical_columns], columns=categorical_columns)
    # remove the original categorical columns

    for col in categorical_columns:
        for value in df[col].unique():
            one_to_original[col + '_' + str(value)] = [col, value]
    df = df.drop(categorical_columns, axis=1)
    df = pd.concat([df, one_hot_encoded_columns], axis=1)
    # save the cat columns with pickle
    # with open(folder_path + '\\' + df_name + '_cat.pkl', 'wb') as f:
    #     pickle.dump(categorical_columns, f)

    return df, one_hot_encoded_columns.columns, one_to_original,categorical_columns


def calc_tpr_fpr(y_true, y_pred):
    """
    Calculates the true positive rate and false positive rate for the given true and predicted values using the sklearn.metrics.confusion_matrix
    Args:
        y_true (list): The true values
        y_pred (list): The predicted values
    return:
        float: The true positive rate
        float: The false positive rate
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp / (tp + fn), fp / (fp + tn)


def equalized_odds(tpr_1, tpr_0, fpr_1, fpr_0):
    """
    Calculates the equalized odds for the given true positive rates and false positive rates
    Args:
        tpr_1 (float): The true positive rate for the positive class
        tpr_0 (float): The true positive rate for the negative class
        fpr_1 (float): The false positive rate for the positive class
        fpr_0 (float): The false positive rate for the negative class
    return:
        float: The equalized odds
    """
    return 0.5 * (abs(tpr_1 - tpr_0) + abs(fpr_1 - fpr_0))


def calc_e_o_feature(df, col, y_true, y_pred):
    """
    Calculates the equalized odds for the given column
    Args:
        df (pd.DataFrame): The dataframe to get the categorical columns from
        col (str): The column to calculate the equalized odds for
        y_true (list): The true values
        y_pred (list): The predicted values
    return:
        float: The equalized odds
    """
    tpr_1, fpr_1 = calc_tpr_fpr(y_true[df[col] == 1], y_pred[df[col] == 1])
    tpr_0, fpr_0 = calc_tpr_fpr(y_true[df[col] == 0], y_pred[df[col] == 0])
    return equalized_odds(tpr_1, tpr_0, fpr_1, fpr_0)








def create_df_prior(target, dataframe, cat_threshold,coulums_to_check=None):
    """
    Calculates the equalized odds and mjoritt for each feature the given dataset
    model - catboost
    Args:
        df (pd.DataFrame): The dataframe to get the categorical columns from
        target (str): The target column name
    return:
        df (pd.DataFrame) : The dataframe with the equalized odds for each column
        the df has the columns: 'feature','value','equalized_odds','majority_1','majority_0'
    """
    df = dataframe
    one_hot_df, one_hot_cols, one_hot_dict,cat_columns = create_one_hot_dataset(df, target, cat_threshold)
    model = CatBoostClassifier(iterations=1000, learning_rate=0.1, depth=6, loss_function='Logloss', verbose=False)
    model.fit(one_hot_df.drop(target, axis=1), one_hot_df[target], cat_features=list(one_hot_cols))

    y_pred = model.predict(one_hot_df.drop(target, axis=1))
    y_true = one_hot_df[target]
    # prior_df = pd.DataFrame(
    #     columns=[ 'feature', 'value', 'equalized_odds', 'tpr 1', 'fpr 1', 'tpr 0', 'fpr 0'])
    report = {
        "features": [],
        "metrics" : []
    }
    report["metrics"].append({"name": "equalized_odds", "columns": []})
    report["metrics"].append({"name": "tpr_1", "columns": []})
    report["metrics"].append({"name": "fpr_1", "columns": []})
    report["metrics"].append({"name": "tpr_0", "columns": []})
    report["metrics"].append({"name": "fpr_0", "columns": []})


    for col in one_hot_cols:
        # print(col)
        col_feature, col_value = one_hot_dict[col]
        if coulums_to_check!=None and col_feature not in coulums_to_check:
            continue
        # if we dint have preds or if is only one class continue
        if len(np.unique(y_pred[one_hot_df[col] == 1])) == 1 or len(np.unique(y_pred[one_hot_df[col] == 0])) == 1:
            continue
        tpr1, fpr1 = calc_tpr_fpr(y_true[one_hot_df[col] == 1], y_pred[one_hot_df[col] == 1])
        tpr0, fpr0 = calc_tpr_fpr(y_true[one_hot_df[col] == 0], y_pred[one_hot_df[col] == 0])
        # adding the shap value of col
        # new_row = {'feature': col_feature, 'value': col_value,
        #            'equalized_odds': calc_e_o_feature(one_hot_df, col, y_true, y_pred), 'tpr 1': tpr1, 'fpr 1': fpr1,
        #            'tpr 0': tpr0, 'fpr 0': fpr0}
        equalized_odds = calc_e_o_feature(one_hot_df, col, y_true, y_pred)

        # Update the JSON dictionary
        feature = f"{col_feature}_{col_value}"
        if feature not in report["features"]:
            report["features"].append(feature)
        report["metrics"][0]["columns"].append({"name" : feature,"value": equalized_odds})
        report["metrics"][1]["columns"].append({"name" : feature,"value": tpr1})
        report["metrics"][2]["columns"].append({"name" : feature,"value": fpr1})
        report["metrics"][3]["columns"].append({"name" : feature,"value": tpr0})
        report["metrics"][4]["columns"].append({"name" : feature,"value": fpr0})

    report["categorical_columns"] = cat_columns

        # prior_df = pd.concat([prior_df, pd.DataFrame([new_row])], ignore_index=True)
    # #sort the df by  feautre and equalized odds
    # prior_df = prior_df.sort_values(by=['feature', 'equalized_odds'], ascending=[True, False])
    # # Convert DataFrame to HTML table string
    # table_html = prior_df.to_html()
    #
    # # Create the full HTML content
    # html_content = f"""
    # <!DOCTYPE html>
    # <html>
    # <head>
    #     <title>DATASET: {df_name} bias detection</title>
    # </head>
    # <body>
    #     <h1>DATASET: {df_name} bias detection</h1>
    #     {table_html}
    # </body>
    # </html>
    # """
    #
    # # Specify the path where you want to save the HTML file
    # html_file_path = f'{folder_path}\prior_{df_name}.html'
    #
    # # Write the HTML content to a file
    # with open(html_file_path, 'w') as file:
    #     file.write(html_content)
    #
    # print(f"HTML file has been created and saved to: {html_file_path}")
    # prior_df.to_csv(f'{folder_path}\prior_{df_name}.csv')
    return report

def detection(dataframe, target_name):
    # bank_path = "C:\\Users\\user\PycharmProjects\\bias_eval\FairUS\\bank\\"
    report_prior = create_df_prior(target_name, dataframe, min_values_for_categorical)
    print(report_prior)
    return report_prior

# %%
#
# law = pd.read_csv('C:\master\datasets\law\law.csv')
# lea_target='pass_bar'
# law_report = detection(law, lea_target)

# # law ,columns, one_to_original = create_df_prior(law,'pass_bar', 'law', 'c:\master\datasets\\law')
# law_prior, law_model = create_df_prior(law, 'pass_bar', 'law', 'c:\master\datasets\\law')
# # %%
# law_prior
# # %%
# compass = pd.read_csv('C:\master\datasets\compas\compas.csv')
# compas_prior, compas_model = create_df_prior(compass, 'two_year_recid', 'compas', 'c:\master\datasets\\compas')
#
# # %%
# compas_prior
# # %%
# nurit = pd.read_csv('C:\master\datasets\\nurit\\nurit.csv')
# nurit_prior, nurit_model = create_df_prior(nurit, 'DiagPeriodL90D', 'nurit', 'c:\master\datasets\\nurit')
# # %%
# credit = pd.read_csv('C:\master\datasets\\credit\\credit.csv')
# credit_prior, credit_model = create_df_prior(credit, 'default payment', 'credit', 'c:\master\datasets\\credit')
# %%
