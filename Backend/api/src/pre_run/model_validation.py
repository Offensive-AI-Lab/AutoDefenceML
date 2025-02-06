
import numpy as np
import traceback
import logging


class ModelDatasetValidator:
    """
        A utility class for validating the compatibility of machine learning models and data loaders.

        Attributes:
        ----------
        metadata (dict): Metadata related to the machine learning model and data loader.
        model: The machine learning model to validate.
        dataloader: The data loader to validate.
        framework (str): The framework used for the machine learning model (e.g., 'pytorch', 'tensorflow', 'sklearn').

        Methods:
        --------
        can_data_run_through_tf_model():
            Validate if data can run through a TensorFlow model.

        can_data_run_through_sklearn_model():
            Validate if data can run through a scikit-learn model.

        can_data_run_through_pytorch_model():
            Validate if data can run through a PyTorch model.

        validate_dummy():
            A placeholder method for custom validation logic.

        validate():
            Perform compatibility validation based on the framework and return the result.

        """
    def __init__(self,fileloader_obj):
        """
                Initialize a ModelDatasetValidator instance.

                Parameters:
                -----------
                metadata (dict): Metadata related to the machine learning model and data loader.

                Raises:
                -------
                ValueError: If the provided model or data loader does not match the expected framework.

                """
        self.metadata = fileloader_obj.metadata
        self.model = fileloader_obj.get_model()
        self.dataloader = fileloader_obj.get_dataloader()
        self.framework = self.metadata['ml_model']['meta']['framework']

    def can_data_run_through_tf_model(self):
        """
                Validate if data can run through a TensorFlow model.

                Returns:
                --------
                tuple or bool: A tuple (num_samples, num_predictions) if successful, False otherwise.

                Raises:
                -------
                ValueError: If the model or data loader does not match the expected framework.

                """
        # Check if the model is a TensorFlow model
        import tensorflow as tf
        from keras.models import Model
        if not isinstance(self.model, tf.keras.models.Model) or not isinstance(self.model,Model ):
            raise ValueError("The provided model should be a TensorFlow model.")

        dataset = self.dataloader.get_dataset()

        # Check if the dataloader is a TensorFlow dataloader
        if not isinstance(dataset, tf.data.Dataset):
            raise ValueError("The provided dataloader should be a TensorFlow dataloader.")

        # Get the input shape of the model
        x,y = self.dataloader.get_next_batch()

        try:
            predictions = self.model.predict(x)
            return len(np.array(y)), len(np.array(predictions))
        except Exception as e:
            return False
    def can_data_run_through_sklearn_model(self):
        """
                Validate if data can run through a scikit-learn model.

                Returns:
                --------
                bool: True if the data can run through the model, False otherwise.

                Raises:
                -------
                ValueError: If the model or data loader does not match the expected framework.

                """
        from sklearn.base import BaseEstimator
        import pandas as pd
        # Check if the model is a scikit-learn estimator
        if not isinstance(self.model, BaseEstimator):
            raise ValueError("The provided model should be a scikit-learn estimator.")

        # Check if the dataloader is an iterable (list or numpy array)
        if not isinstance(self.dataloader.get_dataset(), (list, np.ndarray, pd.DataFrame)):
            raise ValueError("The provided dataloader should be an iterable (list or numpy array).")

        # Get the input shape of the model
        x, y =  self.dataloader.get_next_batch()


        # Check if the dataloader produces data with the same shape as the model's input
        try:
            y_pred = self.model.predict(x)
            return np.array(y).shape == np.array(y_pred).shape
        except Exception as err:
            return False
    def can_data_run_through_pytorch_model(self):
        """
               Validate if data can run through a PyTorch model.

               Returns:
               --------
               bool: True if the data can run through the model, False otherwise.

               Raises:
               -------
               ValueError: If the model or data loader does not match the expected framework.

               """
        import torch
        from torch.utils.data import DataLoader
        # Check if the model is a PyTorch model
        if not isinstance(self.model, torch.nn.Module):
            raise ValueError("The provided model should be a PyTorch model.")

        # Check if the dataloader is a PyTorch dataloader
        if not isinstance(self.dataloader, DataLoader):
            raise ValueError("The provided dataloader should be a PyTorch dataloader.")

        batch = next(iter(self.dataloader))

        try:
            sample_to_try = torch.tensor(batch[0][0].unsqueeze(0)) # take 1 sample as a batch
            self.model.forward(sample_to_try)  # Pass input from the sample batch
            return True
        except Exception as err:
            logging.error(f"Something went wrong while trying to pass data through model, Error:{traceback.format_exc()},{batch[0][0]}, Data: {sample_to_try}")

            return False

    def can_data_run_through_catboost_model(self):
        """
                Validate if data can run through a CatBoost model.

                Returns:
                --------
                bool: True if the data can run through the model, False otherwise.

                Raises:
                -------
                ValueError: If the model or data loader does not match the expected framework.

                """
        from catboost.core import _CatBoostBase
        import pandas as pd
        if not isinstance(self.model, _CatBoostBase):
            raise ValueError("The provided model should be a CatBoost model.")

        # Check if the dataloader is a scikit-learn dataloader

        # Check if the dataloader is an iterable (list or numpy array)
        if not isinstance(self.dataloader.get_dataset(), (list, np.ndarray, pd.DataFrame)):
            raise ValueError("The provided dataloader should be an iterable (list or numpy array).")

        # Get the input shape of the model
        x, y = self.dataloader.get_next_batch()

        # Check if the dataloader produces data with the same shape as the model's input
        try:
            y_pred = self.model.predict(x).flatten()
            return np.array(y).shape == np.array(y_pred).shape
        except Exception as err:
            return False

    def can_data_run_through_xgboost_model(self):
        from xgboost.sklearn import XGBModel
        import pandas as pd
        if not isinstance(self.model, XGBModel):
            raise ValueError("The provided model should be a xgboost estimator.")

        # Check if the dataloader is a scikit-learn dataloader

        # Check if the dataloader is an iterable (list or numpy array)
        if not isinstance(self.dataloader.get_dataset(), (list, np.ndarray, pd.DataFrame)):
            raise ValueError("The provided dataloader should be an iterable (list or numpy array).")

        # Get the input shape of the model
        x, y = self.dataloader.get_next_batch()

        # Check if the dataloader produces data with the same shape as the model's input
        try:
            y_pred = self.model.predict(x)
            return np.array(y).shape == np.array(y_pred).shape
        except Exception as err:
            return False
    def validate_dummy(self):
        return True
    def validate(self):
        """
                Perform compatibility validation based on the framework and return the result.

                Returns:
                --------
                bool: True if the data is compatible with the model, False otherwise.

                Raises:
                -------
                Exception: If the provided framework is not one of the expected values.

                """
        if self.framework == 'pytorch':
            return self.can_data_run_through_pytorch_model()
        elif self.framework == 'tensorflow' or self.framework == 'keras':
            return self.can_data_run_through_tf_model()

        elif self.framework == 'sklearn':
            return self.can_data_run_through_sklearn_model()

        elif self.framework == 'xgboost':
            return self.can_data_run_through_xgboost_model()
        elif self.framework == 'catboost':
            return self.can_data_run_through_catboost_model()


        else:
            raise ValueError(f"Expected framework to be either pytorch, tensorflow or sklearn, but got {self.framework}")

