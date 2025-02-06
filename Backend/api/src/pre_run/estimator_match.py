import traceback
import logging
from ..user_files.helpers import get_files_package_root


class EstimatorHandler:
    """
    A utility class for matching machine learning models with appropriate ART estimators,
    and wrapping the models with the selected estimators.

    Attributes:
    ----------
    file_loader: A FileLoader object for loading metadata.
    ml_type (str): The type of machine learning task ('classification' or 'regression').
    implementation (str): The framework used for the machine learning model (e.g., 'pytorch', 'tensorflow', 'sklearn').
    algorithm (str): The specific ML algorithm used in the model (relevant for sklearn implementations).
    input_shape (tuple/list): The shape of the input data.
    input_val_range (tuple/list): The range of values for the input data (can be a tuple of tuples for columns).
    num_of_classes (int): The number of classes in a classification problem.
    estimator: Estimator type (from the ART library) obtained after wrapping the model.
    map (dict): A mapping of ML types, implementations, and algorithms to corresponding ART estimator types.

    Methods:
    --------
    _estimator_match(): Match an estimator to an ML model based on parameters.

    _estimator_params(): Extract relevant parameters for the selected estimator.

    wrap(): Match, wrap, and save the model using the appropriate estimator.

    """

    def __init__(self, input, fileloader_obj):
        """
                Initialize an EstimatorHandler instance.

                Parameters:
                -----------
                input (dict): Input parameters, including ML type, implementation, algorithm, etc.
                metadata (dict): Metadata related to the machine learning model.

                Raises:
                -------
                TypeError: If metadata is not of type dict.

                """
        self.__file_loader = fileloader_obj
        self.metadata = fileloader_obj.metadata
        try:

            self.__ml_type = input["ML_type"]
            self.__implementation = input["implementation"]
            # self.__algorithm = input['algorithm']
            self.__input_shape = self.metadata['ml_model']['dim']['input']
            self.__input_val_range = self.metadata['ml_model']['dim']["clip_values"]
            self.__num_of_classes = self.metadata['ml_model']['dim']["num_classes"]
            self.estimator = None  # get a value after self._wrap_model will run
            self.map = {"implementation": {
                "pytorch": {
                    "ML_type": {
                        "classification": "PyTorchClassifier",
                        "regression": "PyTorchRegressor"
                    }
                },
                "tensorflow": {
                    "ML_type": {
                        "classification": "TensorFlowV2Classifier"


                    }
                },
                "keras": {
                    "ML_type": {
                        "classification": "KerasClassifier"

                    }
                },
                "sklearn": {
                    "ML_type": {
                        "classification": "SklearnClassifier",
                        "regression": "ScikitlearnRegressor"
                    }
                },
                "xgboost": {
                    "ML_type": {
                        "classification": "XGBoostClassifier",

                    }
                },
                "catboost": {
                    "ML_type": {
                        "classification": "CatBoostARTClassifier",

                    }
                },
            }
            }
        except KeyError as err:
            raise Exception("Can't wrap model!Missing input parameters").with_traceback(err.__traceback__)

    def _estimator_match(self):
        """
                Match an estimator to an ML model based on parameters.

                Returns:
                --------
                str: Estimator type.

                """
        try:
            estimator = self.map['implementation'][self.__implementation]['ML_type'][self.__ml_type]
            return estimator
        except KeyError as err:
            raise Exception("Can't find estimator type!\nWrong input parameters").with_traceback(err.__traceback__)

    def _estimator_params(self):
        """
                Extract relevant parameters for the selected estimator.

                Returns:
                --------
                dict: A dictionary of estimator parameters.

                """
        param_dict = {'loss': False, 'optimizer': False, 'clip_values': None, 'nb_classes': None, 'input_shape': None}
        if self.__implementation == 'sklearn':
            if self.__input_val_range:
                param_dict['clip_values'] = self.__input_val_range
        elif self.__implementation == 'xgboost':
            if self.__input_val_range:
                param_dict['clip_values'] = self.__input_val_range
            if self.__ml_type == "regression":
                pass
            elif self.__ml_type == "classification":
                param_dict['nb_classes'] = self.__num_of_classes
                if self.__input_shape[0]:
                    param_dict['nb_features'] = self.__input_shape[0]

        elif self.__implementation == 'catboost':
            if self.__input_val_range:
                param_dict['clip_values'] = self.__input_val_range
            if self.__input_shape[0]:
                param_dict['nb_features'] = self.__input_shape[0]
            if self.__ml_type == "regression":
                pass
            elif self.__ml_type == "classification":
                param_dict['nb_features'] = self.__input_shape[0]
        elif self.__implementation == 'tensorflow' or self.__implementation == 'keras':
            if self.__ml_type == "regression":
                pass
            elif self.__ml_type == "classification":
                if self.__input_val_range:
                    param_dict['clip_values'] = self.__input_val_range
                if self.__implementation != 'keras':
                    if self.metadata['ml_model']['optimizer']['type']:
                        param_dict['optimizer'] = True
                    param_dict['nb_classes'] = self.__num_of_classes
                    param_dict['input_shape'] = self.__input_shape
                    if self.metadata['ml_model']['loss']['type'] :
                        param_dict['loss'] = True
            else:
                raise Exception("Can't wrap model!\nML type must be classification or regression")
        elif self.__implementation == 'pytorch':
            param_dict['clip_values'] = self.__input_val_range
            if self.metadata['ml_model']['optimizer']['type']:
                param_dict['optimizer'] = True
            param_dict['loss'] = True
            if self.__ml_type == "regression":
                param_dict['input_shape'] = self.__input_shape

            elif self.__ml_type == "classification":
                param_dict['input_shape'] = self.__input_shape
                param_dict['nb_classes'] = self.__num_of_classes
                if self.__input_val_range:
                    param_dict['clip_values'] = self.__input_val_range

            else:
                raise Exception("Can't wrap model!\nML type must be classification or regression")

        else:
            raise Exception("Can't wrap model!\nImplementation type must be sklearn,pytorch or tensorflow")
        # extract the params that needed - meaning not None
        param_dict = {k: v for k, v in param_dict.items() if v}
        return param_dict

    def wrap(self):
        """
               Match, wrap, and save the model using the appropriate estimator.

               """
        try:
            logging.info("wrapping model with estimator")
            estimator_obj = self._estimator_match()
            params = self._estimator_params()
            estimator_dict = {"object": estimator_obj, "params": params}
            logging.info(f"chosen estimator is: {estimator_dict}")
            dest_path = get_files_package_root() + f"/Estimator_params.json"
            logging.info("saving estimator params...")
            self.__file_loader.save_file(obj=estimator_dict, path=dest_path, as_json=True)
            logging.info("params saved successfully")
        except Exception as err:
            logging.error(f"Error occurred while wrapping estimator.\n Error: {err}")
            logging.error(traceback.format_exc())