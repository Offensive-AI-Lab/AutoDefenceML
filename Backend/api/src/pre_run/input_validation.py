
import traceback
import logging

class InputValidator:
    """
        A utility class for validating input parameters, including machine learning models, data loaders,
        loss functions, optimizers, and model-specific parameters.

        Attributes:
        ----------
        metadata (dict or str): Metadata related to the input parameters.
        __file_loader (FileLoader): An instance of the FileLoader class for loading files and objects.

        Methods:
        --------
        validate_model(model):
            Validate the machine learning model.

        validate_dataloader(dataloader):
            Validate the data loader.

        validate_loss_func():
            Validate the loss function.

        validate_optimizer():
            Validate the optimizer.

        validate_input_shape():
            Validate the input shape parameter.

        validate_num_of_classes():
            Validate the number of classes parameter.

        validate_range_of_vals():
            Validate the range of values parameter.

        validatae_model_param():
            Validate model-specific parameters.

        validate_dummy():
            A placeholder method for custom validation logic.

        validate(print_res=True):
            Perform all validation checks and print the results.

        get_input():
            Get the validated input parameters.

        """
    def __init__(self, fileloader_obj) -> None:
        """
                Initialize an InputValidator instance.

                Parameters:
                -----------
                metadata (dict or str): Metadata related to the input parameters.

                Raises:
                -------
                TypeError: If metadata is not of type dict or str.

                """
        self.__file_loader = fileloader_obj
        self.metadata = fileloader_obj.metadata
        self.__input = {"ML_type": self.metadata['ml_model']['meta']['ml_type']
            , "implementation": self.metadata['ml_model']['meta']['framework'],
                        "Loss": None, "Optimizer": None}



    def validate_model(self, model):
        """
               Validate the machine learning model.

               Parameters:
               -----------
               model: The machine learning model to validate.

               Returns:
               --------
               bool: True if the model is valid, False otherwise.

               """
        try:
            # e.g pytorch, tensorflow, sklearn
            model_implementation_type = self.metadata['ml_model']['meta']['framework']
            if model_implementation_type == "sklearn":
                from sklearn.base import BaseEstimator
                return isinstance(model, BaseEstimator)
            elif model_implementation_type == "xgboost":
                from xgboost.sklearn import XGBModel
                return isinstance(model, XGBModel)
            elif model_implementation_type == "catboost":
                from catboost.core import _CatBoostBase
                return isinstance(model, _CatBoostBase)
            elif model_implementation_type == "tensorflow":
                import tensorflow as tf
                return isinstance(model, tf.keras.models.Model)
            elif model_implementation_type == "keras":
                from keras.models import Model
                import tensorflow as tf
                return isinstance(model, Model) or isinstance(model, tf.keras.models.Model)
            elif model_implementation_type == "pytorch":
                import torch.nn as nn
                return isinstance(model, nn.Module)
        except Exception as err:
            logging.error(f'Did not manage to validate ML model, Error occurred\nError:\n{traceback.format_exc()}')
            return False

    def validate_dataloader(self,dataloader):
        """
                Validate the data loader.

                Parameters:
                -----------
                dataloader: The data loader to validate.

                Returns:
                --------
                bool: True if the data loader is valid, False otherwise.

                """
        try:
            # if the framework is sklearn then the dataloader must be custom
            # if self.metadata['dataloader']['definition']['class_name'] == 'CustomLoader':
            #     output = dataloader.get_next_batch() # Check if the dataloader return two elements
            #     return True
            framework = self.metadata['ml_model']['meta']['framework']
            if framework == 'pytorch':
                from torch.utils.data import DataLoader
                return isinstance(dataloader, DataLoader)
            elif framework == 'tensorflow' or framework == 'keras' :
                import tensorflow as tf
                dataset=dataloader.get_dataset()
                return isinstance(dataset, tf.data.Dataset)
            elif framework == 'sklearn' or framework == 'xgboost' or framework == 'catboost':
                x, y = dataloader.get_next_batch()
                return True

        except Exception as err:
            logging.error(f'Did not manage to validate dataloader, Error occurred\nError:\n{err}')

            return False

    def validate_loss_func(self):
        """
               Validate the loss function.

               Returns:
               --------
               bool: True if the loss function is valid, False otherwise.

               """
        try:
            model_implementation_type = self.metadata['ml_model']['meta']['framework']
            valid_loss_func_type = None
            if model_implementation_type == "pytorch":
                from torch.nn.modules.loss import _Loss, L1Loss, KLDivLoss, MSELoss, \
                    HuberLoss, SmoothL1Loss,MarginRankingLoss, MultiMarginLoss, \
                    MultiLabelSoftMarginLoss,NLLLoss, GaussianNLLLoss, \
                    PoissonNLLLoss, CrossEntropyLoss, BCELoss,BCEWithLogitsLoss,CosineEmbeddingLoss, \
                    CTCLoss, HingeEmbeddingLoss, SoftMarginLoss,TripletMarginLoss, TripletMarginWithDistanceLoss

                valid_loss_func_type = _Loss()
                loss_func = self.__file_loader.get_loss()
                if loss_func == None:
                    logging.error("Pytorch must have a loss function provided")
                    return False
            elif model_implementation_type == "tensorflow" or model_implementation_type == "keras":
                from keras.losses import Loss
                valid_loss_func_type = Loss()
                loss_func = self.__file_loader.get_loss()
            else:
                logging.info("No loss function provided")
                return True
            self.__input['Loss'] = valid_loss_func_type
            if loss_func:
                return isinstance(loss_func, type(valid_loss_func_type))
            else:
                return True

        except Exception as err:

            logging.error(f'Did not manage to validate loss function, Error occurred\nError:\n{traceback.format_exc()}')
            return False


    def validate_optimizer(self):
        """
                Validate the optimizer.

                Returns:
                --------
                bool: True if the optimizer is valid, False otherwise.

                """
        try:
            optimizer_type = self.metadata['ml_model']['optimizer']['type']
            lr = self.metadata['ml_model']['optimizer']['learning_rate']
            model = self.__file_loader.get_model()
            if self.metadata['ml_model']['meta']['framework'] == "pytorch":
                model_params= model.parameters()
                if optimizer_type:
                    import torch.optim as optim
                    optimizer_class = getattr(optim, optimizer_type)
                    optimizer = optimizer_class(model_params, lr=lr) #missing params argument
                    valid_optimizer_type = optim
                    self.__input['Optimizer'] = optimizer
                    return isinstance(optimizer, optim.Optimizer.__base__)
                else:
                    logging.info("No optimizer provided")
                    return True

            elif  self.metadata['ml_model']['meta']['framework'] == "keras" or self.metadata['ml_model']['meta']['framework'] == "tensorflow":
                if optimizer_type:
                    import keras
                    import keras.optimizers
                    optimizer = None
                    optimizer_class = getattr(keras.optimizers, optimizer_type)
                    optimizer = optimizer_class(learning_rate=lr)
                    from keras.optimizers import Optimizer
                    valid_optimizer_type = Optimizer
                    self.__input['Optimizer'] = optimizer
                    return isinstance(optimizer, valid_optimizer_type)
                else:
                    logging.info("No Optimizer provided")
                    return True


            else:
                logging.info("No Optimizer provided")
                return True



        except Exception as err:
            logging.error(f'Did not manage to validate optimizer, Error occurred\nError:\n{traceback.format_exc()}')
            return False

    def validate_input_shape(self):
        """
                Validate the input shape parameter.

                Returns:
                --------
                bool: True if the input shape is valid, False otherwise.

                """
        try:
            shape = self.metadata['ml_model']['dim']["input"]
            is_shape = all([shape is not None,
                            (isinstance(shape, tuple) or isinstance(shape, list)),
                            all([isinstance(dim, int) for dim in shape])])
            return is_shape
        except Exception as err:
            logging.error(f'Did not manage to validate parameter, Error occurred\nError:\n{traceback.format_exc()}')
            return False

    def validate_num_of_classes(self):
        """
                Validate the number of classes parameter.

                Returns:
                --------
                bool: True if the number of classes is valid, False otherwise.

                """
        # this is optional value
        try:
            num_classes = self.metadata['ml_model']['dim']["num_classes"]
            if num_classes:
                return isinstance(num_classes, int)
            else:
                # if self.metadata['ml_model']['meta']['framework'] == "pytorch" or self.metadata['ml_model']['meta']['framework'] == "tensorflow":
                logging.error("Number of classes not provided")
                return False
            return True
        except Exception as err:
            logging.error(f'Did not manage to validate parameter, Error occurred\nError:\n{traceback.format_exc()}')
            return False


    def validate_range_of_vals(self):
        """
               Validate the range of values parameter.

               Returns:
               --------
               bool: True if the range of values is valid, False otherwise.

               """
        # this is optional value
        data_range = self.metadata['ml_model']['dim']["clip_values"]
        if data_range:
            is_singel_range = all([
                                   isinstance(data_range, list) or isinstance(data_range, tuple),
                                   len(data_range) == 2, isinstance(data_range[0], (float,int)) and
                                   isinstance(data_range[1], (float,int)) and data_range[0] < data_range[1]])
            try:
                is_var_of_ranges = [isinstance(r, tuple) and len(r) == 2 and
                                    isinstance(r[0], (float,int)) and isinstance(r[1], (float,int))
                                    and r[0] < r[1] for r in data_range]
            except:
                is_var_of_ranges = [False]
            return is_singel_range or all(is_var_of_ranges)
        else:
            return True

    def validate_model_param(self):
        """
                Validate model-specific parameters.

                Returns:
                --------
                bool: True if the model-specific parameters are valid, False otherwise.

                """
        try:
            model_implementation_type = self.metadata['ml_model']['meta']['framework']
            ml_type = self.metadata['ml_model']['meta']['ml_type']
            if model_implementation_type == "sklearn" or model_implementation_type == "xgboost" or model_implementation_type == "catboost":
                # needed prarams are: clip_values --> range of values
                return self.validate_num_of_classes() and self.validate_input_shape() and self.validate_range_of_vals() # optional clip values
            elif model_implementation_type == "tensorflow" or model_implementation_type == "keras" or model_implementation_type == "pytorch":
                return self.validate_num_of_classes() and self.validate_input_shape() and self.validate_range_of_vals()

            else:
                logging.info(f"No such implementation: {model_implementation_type}")
        except Exception as err:
            logging.error(f'Did not manage to validate params, Error occurred\nError:\n{traceback.format_exc()}')
            return False

    def validate(self, print_res=True):
        """
                Perform all validation checks and print the results.

                Parameters:
                -----------
                print_res (bool): Whether to print the validation results.

                Returns:
                --------
                bool: True if all checks pass, False otherwise.

                """
        print(f"Validation results:")
        model = self.__file_loader.get_model()
        print(f"  - Model: {model}")
        dataloader = self.__file_loader.get_dataloader()
        print(f"  - dataloader: {dataloader}")
        model_validity = self.validate_model(model)
        print(f"  - model_validity: {model_validity}")
        loss_func_validity = self.validate_loss_func()
        print(f"  - Loss function: {loss_func_validity}")
        optimizer_validity = self.validate_optimizer()
        print(f"  - Optimizer: {optimizer_validity}")
        dataloader_validity = self.validate_dataloader(dataloader)
        print(f"  - Dataloader: {dataloader_validity}")
        model_params_validity = self.validate_model_param()
        print(f"  - Model params: {model_params_validity}")

        
        if print_res:
            print(f"""Validation results:\nmodel: {model_validity}\nloss function: {loss_func_validity}\noptimzer: {optimizer_validity}\ndataloader: {dataloader_validity}\nparams: {model_params_validity} """)
        return all([model_validity, loss_func_validity, optimizer_validity, dataloader_validity, model_params_validity])

    def get_input(self):
        """
                Get the validated input parameters.

                Returns:
                --------
                dict: A dictionary containing validated input parameters.
                """
        if self.validate(print_res=False):
            return self.__input
        else:
            logging.info("Input is not valid!")
            return None

