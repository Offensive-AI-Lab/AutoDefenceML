import category_encoders as ce
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, precision_score
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
import json
import re

from xhtml2pdf import pisa


def get_user_input_from_json(json_data):
    try:
        dataset_info = json_data.get('dataset', {})
        path = dataset_info.get('path', '')
        name = dataset_info.get('name', '')
        columns_info = json_data.get('columns', {})
        categorical_columns = columns_info.get('categorical', [])
        target_column = columns_info.get('target', '')

        orig_data = upload_csv_and_load_df(path)
        unique_values = orig_data[target_column].unique()

        if not all(value in [0, 1] for value in unique_values):
            raise ValueError(f"Target column '{target_column}' contains values other than 0 or 1: {unique_values}")

        return name, categorical_columns, target_column, orig_data
    except KeyError as e:
        raise KeyError(f"Missing key in input JSON: {e}")
    except Exception as e:
        raise Exception(f"Error in get_user_input_from_json: {e}")

def upload_csv_and_load_df(file_path):
    try:
        # Attempt to load the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        print(f'DataFrame loaded successfully with shape: {df.shape}')
        return df
    except Exception as e:
        print(f'Error loading the CSV file into a DataFrame: {str(e)}')
        return None


def get_user_input():
    DATASET_NAME = input("The dataset name is: ")
    categorical_columns = input("Enter categorical columns (comma-separated): ").split(',')
    target_column = input("Enter target column: ")


    return DATASET_NAME, categorical_columns, target_column

def calculate_metrics(data, target_column, feature_columns, categorical_cols):
    skf = StratifiedKFold(n_splits=5, random_state=1234, shuffle=True)
    accuracy_scores = []
    recall_scores = []
    precision_scores = []

    for train_index, test_index in skf.split(data, data[target_column]):
        train_data = data.iloc[train_index]
        test_data = data.iloc[test_index]

        train_X = train_data[feature_columns]
        train_y = train_data[target_column]
        test_X = test_data[feature_columns]
        test_y = test_data[target_column]
        model = xgb.XGBClassifier()
        model.fit(train_X, train_y)
        predictions = model.predict(test_X)

        accuracy_scores.append(accuracy_score(test_y, predictions))
        recall_scores.append(recall_score(test_y, predictions))
        precision_scores.append(precision_score(test_y, predictions))

    mean_accuracy = np.mean(accuracy_scores)
    mean_recall = np.mean(recall_scores)
    mean_precision = np.mean(precision_scores)
    std_accuracy = np.std(accuracy_scores)
    std_recall = np.std(recall_scores)
    std_precision = np.std(precision_scores)

    return mean_accuracy, mean_recall, mean_precision , std_accuracy, std_recall ,std_precision

def ensemble_poison_detector(orig_data, y_orig):
    orig_data = orig_data.copy()
    orig_data_np = orig_data.to_numpy()
    y_orig_np = y_orig.to_numpy()
    np.random.seed(20)

    models = []
    weights = []  # New list to store the weights of the models

    clf_size_data = int(len(orig_data_np) * 0.1)
    model_types = ['GBM','GBM','XGBoost','XGBoost', 'RandomForest','RandomForest','LogisticRegression','LogisticRegression','KNN','KNN']


    num_models = len(model_types)


    for model_type in model_types:
        permuted_indices = np.random.permutation(len(orig_data_np))
        train_indices, test_indices = permuted_indices[clf_size_data:], permuted_indices[:clf_size_data]

        if model_type == 'XGBoost':
            model = XGBClassifier(n_estimators=5, max_depth=3)
            weight = 1.0  # Increased weight for XGBoost
        elif model_type == 'RandomForest':
            model = RandomForestClassifier(n_estimators=5, max_depth=5)
            weight = 1.0  # Increased weight for Random Forest
        elif model_type == 'SVM':
            model = SVC(kernel='linear')
            weight = 1.0  # Default weight for other models
        elif model_type == 'NeuralNetwork':
            model = MLPClassifier(hidden_layer_sizes=(100,))
            weight = 1.0
        elif model_type == 'LogisticRegression':
            model = LogisticRegression(max_iter=5000)
            weight = 1.0
        elif model_type == 'NaiveBayes':
            model = GaussianNB()
            weight = 1.0
        elif model_type == 'KNN':
            model = KNeighborsClassifier()
            weight = 1.0
        elif model_type == 'GBM':
            model = GradientBoostingClassifier(n_estimators=100, max_depth=3)  # You can adjust hyperparameters
            weight = 1.0

        model.fit(orig_data_np[train_indices, :-1], y_orig_np[train_indices])
        models.append(model)
        weights.append(weight)  # Append the weight to the list

    agreement_percentages = []
    for i in range(len(orig_data_np)):
        predictions = np.array([model.predict([orig_data_np[i, :-1]]) for model in models])
        weighted_predictions = predictions * np.array(weights)[:, np.newaxis]  # Apply weights to predictions
        agreement = np.count_nonzero(weighted_predictions == y_orig_np[i])
        agreement_percent = agreement / ( np.sum(weights))  # Consider weights in the calculation
        agreement_percentages.append(agreement_percent)

    orig_data['Agreement_per'] = agreement_percentages
    return orig_data


# Press the green button in the gutter to run the script.
def run_ensemble_depoisoning(dataframe, target_name):
    # json_input = {
    #     'dataset': {
    #         'path': r'C:\Users\groze\Documents\אוניברסיטה\תואר שני\מחברות למאמר\datasets\musk2\m2.csv',
    #         'name': 'musk2'
    #     },
    #     'columns': {
    #         'categorical': [],
    #         'target': 'Class'
    #     }
    # }
    try:
        # DATASET_NAME, categorical_columns, target_column, orig_data = get_user_input_from_json(json_input)
        global target_column, orig_data, categorical_columns
        orig_data= dataframe
        target_column = target_name
        feature_columns = list(orig_data.columns)
        feature_columns.remove(target_column)
        categorical_columns = orig_data.select_dtypes(include=['object', 'category']).columns.tolist()

        if categorical_columns == ['']:
            categorical_columns = list()
        if target_column in categorical_columns:
            categorical_columns.remove(target_column)
        orig_data[target_column], _ = pd.factorize(orig_data[target_column])
        # dataset for training
        orig_data.dropna(thresh=len(feature_columns) / 2, inplace=True)
        orig_data.reset_index(drop=True, inplace=True)
        y_orig = orig_data[target_column]
        y_orig.dropna(inplace=True)
        orig_data = orig_data.loc[:, feature_columns]
        encoder = ce.TargetEncoder(cols=categorical_columns)

        encoder.fit(orig_data, y_orig)
        orig_data = encoder.transform(orig_data)
        orig_dataset = pd.concat([orig_data, y_orig], axis=1)
        mean_accuracy, mean_recall, mean_precision, std_accuracy, std_recall, std_precision = calculate_metrics(
            orig_dataset, target_column, feature_columns, categorical_columns)

        # Plot results
        metrics = ['Accuracy', 'Recall', 'Precision']
        means = [mean_accuracy, mean_recall, mean_precision]
        stds = [std_accuracy, std_recall, std_precision]

        plt.figure(figsize=(6, 3))
        plt.bar(metrics, means, yerr=stds, capsize=10)
        plt.ylabel('Metrics')
        plt.title('Mean Metrics with Standard Deviation')
        plt.savefig('plot_metrics_before_depoisoning.jpg')

        mean_acc_std = str(round(mean_accuracy, 4)) + " ± " + str(round(std_accuracy, 4))
        mean_recall_std = str(round(mean_recall, 4)) + " ± " + str(round(std_recall, 4))
        mean_precision_std = str(round(mean_precision, 4)) + " ± " + str(round(std_precision, 4))


        print("Mean Orig Accuracy:", mean_acc_std)
        print("Mean Orig Recall:", mean_recall_std)
        print("Mean Orig Precision:", mean_precision_std)

        updated_data = ensemble_poison_detector(orig_data, y_orig)
        Suspicious_poison_indices = updated_data[updated_data['Agreement_per'] <= 0.1].index

        # Remove rows with specified indexes
        depoisoned_dataset = orig_dataset.drop(Suspicious_poison_indices)
        depoisoned_mean_accuracy, depoisoned_mean_recall, depoisoned_mean_precision, depoisoned_std_accuracy, depoisoned_std_recall, depoisoned_std_precision = calculate_metrics(
            depoisoned_dataset, target_column, feature_columns, categorical_columns)

        # Plot results
        metrics = ['Accuracy', 'Recall', 'Precision']
        means = [mean_accuracy, mean_recall, mean_precision]
        stds = [std_accuracy, std_recall, std_precision]

        plt.figure(figsize=(6, 3))
        plt.bar(metrics, means, yerr=stds, capsize=10)
        plt.ylabel('Metrics')
        plt.title('Mean Metrics with Standard Deviation')
        plt.savefig('plot_metrics_after_depoisoning.jpg')

        depoisoned_mean_acc_std = str(round(depoisoned_mean_accuracy, 4)) + " ± " + str(round(depoisoned_std_accuracy, 4))
        depoisoned_mean_recall_std = str(round(depoisoned_mean_recall, 4)) + " ± " + str(round(depoisoned_std_recall, 4))
        depoisoned_mean_precision_std = str(round(depoisoned_mean_precision, 4)) + " ± " + str(round(depoisoned_std_precision, 4))

        print("Mean Depoisoned Accuracy:", depoisoned_mean_acc_std)
        print("Mean Depoisoned Recall:", depoisoned_mean_recall_std)
        print("Mean Depoisoned Precision:", depoisoned_mean_precision_std)

        Suspicious_poison_indices = list(Suspicious_poison_indices)

        ORIG_SIZE = orig_dataset.shape[0]
        DEPOISONED_SIZE = depoisoned_dataset.shape[0]
        NUM_POISON_SAMPLES = orig_dataset.shape[0] - depoisoned_dataset.shape[0]

        # Define paths to the images
        plot_metrics_before_depoisoning = '/content/plot_metrics_before_depoisoning.jpg'
        plot_metrics_after_depoisoning = '/content/plot_metrics_after_depoisoning.jpg'

        import base64

        # Load the image as binary data
        with open('plot_metrics_before_depoisoning.jpg', 'rb') as metrics_before_depoisoning_file:
            plot_before_depoisoning = metrics_before_depoisoning_file.read()

        # Load the image as binary data
        with open('plot_metrics_after_depoisoning.jpg', 'rb') as plot_metrics_after_depoisoning_file:
            plot_after_depoisoning = plot_metrics_after_depoisoning_file.read()

        # Convert the binary data to Base64
        plot_before_depoisoning_base64 = base64.b64encode(plot_before_depoisoning).decode('utf-8')
        # Convert the binary data to Base64
        plot_after_depoisoning_base64 = base64.b64encode(plot_after_depoisoning).decode('utf-8')

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
    <h1>Dataset Depoisoning Report Ensemble</h1>

    <h2>Original Dataset: ({ORIG_SIZE} rows)</h2>

    <h3>Original Dataset Metrics:</h3>
    <table>
        <tr>
            <th>Accuracy</th>
            <td>{mean_acc_std}</td>
        </tr>
        <tr>
            <th>Recall</th>
            <td>{mean_recall_std}</td>
        </tr>
        <tr>
            <th>Precision</th>
            <td>{mean_precision_std}</td>
        </tr>
    </table>

    <h3>Original Dataset Metrics: plot</h3>
    <img src="data:image/png;base64,{plot_before_depoisoning_base64}" alt="Depoisoned Dataset Metrics">

    <h2>Depoisoned Dataset:  ({DEPOISONED_SIZE} rows)</h2>
    <table>
        <tr>
            <th>Accuracy</th>
            <td>{depoisoned_mean_acc_std}</td>
        </tr>
        <tr>
            <th>Recall</th>
            <td>{depoisoned_mean_recall_std}</td>
        </tr>
        <tr>
            <th>Precision</th>
            <td>{depoisoned_mean_precision_std}</td>
        </tr>
    </table>

    <h3>Depoisoned Dataset Metrics: plot</h3>
    <img src="data:image/png;base64,{plot_after_depoisoning_base64}" alt="Depoisoned Dataset Metrics">
    <h2> There are {NUM_POISON_SAMPLES} instances that suspected to be poisoned, their indexes are {Suspicious_poison_indices}: </h2>

</body>
</html>

        '''

        statistics = {
            "Original": {
                "accuracy": mean_acc_std,
                "recall": mean_recall_std,
                "precision": mean_precision_std
            },
            "Upsampled": {
                "accuracy": depoisoned_mean_acc_std,
                "recall": depoisoned_mean_recall_std,
                "precision": depoisoned_mean_precision_std
            },
            "The suspicious poison indices":{
                "indices": Suspicious_poison_indices
            }
        }
        # Write the HTML content to a file
    #     with open('report_depoisoning.html', 'w', encoding='utf-8') as f:
    #         f.write(html_content)
    #         print("Report saved as 'report_depoisoning.html'")
    #
    #     with open('results.json', 'w') as f:
    #         json.dump(statistics, f, ensure_ascii=False)
    #         print("Report saved as 'report.json'")
    #
    # except Exception as e:
    #     print(f"An error occurred: {e}")
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
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None
    return statistics, base64_encoded_string