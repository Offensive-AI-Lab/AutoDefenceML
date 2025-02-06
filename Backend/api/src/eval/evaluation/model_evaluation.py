import copy
import logging
import traceback
import numpy as np
from skopt import gp_minimize
from skopt.space import Integer, Categorical
from sklearn.metrics import *
from sklearn.preprocessing import label_binarize
from art.defences.postprocessor import *
from art.defences.preprocessor import *
from art.utils import compute_success
import os
import inspect
import random
from ..user_files_eval.helpers import get_run_files
from art_attacks_plugin import *
import pandas as pd

class Attacker:
    """
    This class is responsible for the attack process.
    Attributes:
    ----------
    estimator: The estimator to attack.
    dataloader: The dataloader to use for the attack.
    ml_type: The type of the estimator (classification/regression).
    framework: The framework used for the estimator (e.g., 'pytorch', 'tensorflow', 'sklearn').
    opt_params_attack: A boolean indicating whether to optimize the hyperparameters of the attack.
    epsilon: The epsilon value for the attack.

    Methods:
    --------
    attack_eval(): Evaluate the success of an attack.
    convert_to_numpy(): Convert a tensor to a numpy array.
    check_zoo(): Check if the Zoo attack is compatible with the estimator.
    optimize_evasion_attack(): Optimize the hyperparameters of an attack.
    attack_estimator(): Attack the estimator.

    """

    def __init__(self, dataloader, ml_type, framework, opt_params_attack, epsilon):
        """
        Initialize an Attacker instance.
        """
        self.dataloader = dataloader
        self.ml_type = ml_type
        self.framework = framework
        self.opt_params_attack = opt_params_attack
        self.epsilon = epsilon

    @staticmethod
    def attack_eval(estimator, x_test, y_test, adv_x, ml_type):
        """
        Evaluate the success of an attack.
        parameters:
        estimator: The estimator to attack.
        x_test: The clean data.
        y_test: The labels of the clean data.
        adv_x: The adversarial examples.
        ml_type: The type of the estimator (classification/regression).
        return: The success rate of the attack.
        """
        if ml_type == 'classification':
            return compute_success(classifier=estimator, x_clean=x_test, labels=y_test, x_adv=adv_x)
        elif ml_type == 'regression':
            preds = estimator.predict(x_test)
            return mean_squared_error(y_test, preds)

    @staticmethod
    def convert_to_numpy(framework, tensor):
        """
        Convert a tensor to a numpy array.
        parameters:
        framework: The framework used for the tensor (e.g., 'pytorch', 'tensorflow', 'sklearn').
        tensor: The tensor to convert.
        return: The tensor as a numpy array.
        """
        if framework == "pytorch":
            import torch
            return torch.Tensor.numpy(tensor)
        elif framework == "tensorflow":
            # return tf.make_ndarray(tensor)
            return np.array(tensor)
        else:
            return np.array(tensor)

    def check_zoo(self, attack, HP, estimator):
        """
        Check if the Zoo attack is compatible with the estimator and the dataloader on different values for 'nb_parallel' parameter.
        parameters:
        attack: The Zoo attack.
        HP: The hyperparameters of the attack.
        estimator: The estimator to attack.
        """
        for param in [100, 10, 1]:
            try:
                HP['nb_parallel'] = param
                sample_data_x, sample_data_y = self.dataloader.get_next_batch()
                attack_init = attack(estimator, **HP)
                attack_init.generate(np.array(sample_data_x))
                break
            except Exception as e:
                pass

    def optimize_evasion_attack(self, attack , estimator, optimize):
        """
        This function only responsible for optimize the hyper parameters a post processor type defense.
        :param attack:
        :return: the optimized defense instance.
        """
        # attack, estimator = kwargs.values()
        attacks_parameters = {'CarliniL2Method': ['confidence'],
                              'BasicIterativeMethod': ['eps_step'],
                              'NewtonFool': ['eta'],
                              'UniversalPerturbation': ['attacker', 'delta'],
                              'ZooAttack': ['confidence', 'learning_rate', 'binary_search_steps',
                                            'variable_h'],
                              'ElasticNet': ['confidence', 'learning_rate'],
                              'BoundaryAttack': ['delta'], 'SimBA': ['freq_dim', 'stride'],
                              'AutoProjectedGradientDescent': ['eps_step'],
                              'BrendelBethgeAttack': ['lr', 'lr_decay', 'momentum'],
                              'DecisionTreeAttack': ['offset'],
                              'DeepFool': ['nb_grads'],
                              'FrameSaliencyAttack': ['method'],
                              'GeoDA': ['bin_search_tol', 'lambda_param', 'sigma'],
                              'HighConfidenceLowUncertainty': ['conf', 'unc_increase'],
                              'HopSkipJump': ['max_eval', 'init_eval', 'init_size'],
                              'LowProFool': ['threshold', 'eta', 'eta_decay'],
                              'MomentumIterativeMethod': ['eps_step'],
                              'PixelThreshold': ['es'],
                              'SaliencyMapMethod': ['gamma', 'theta'],
                              'ShadowAttack': ['sigma', 'nb_steps', 'learning_rate', 'lambda_c', 'lambda_s'],
                              'SignOPTAttack': ['k', 'alpha', 'beta', 'num_trial'],
                              'SpatialTransformation': ['max_translation', 'max_rotation']
                              }

        attack_parameter_range = {'CarliniL2Method': {'confidence': (0, 0.5)},
                                  'CarliniInfMethod': {'confidence': (0, 0.5)},
                                  'BasicIterativeMethod': {'eps_step': (0.1, 1)},
                                  'NewtonFool': {'eta': (0.001, 0.1)},
                                  'UniversalPerturbation': {
                                      'attacker': ('deepfool', 'fgsm', 'bim'),
                                      'delta': (0.01, 0.5)},
                                  'ZooAttack': {'confidence': (0.0, 0.5), 'learning_rate': (0.001, 0.1),
                                                'binary_search_steps': (5, 10),
                                                'variable_h': (0.1, 0.4)},
                                  'ElasticNet': {'confidence': (0.0, 0.5), 'learning_rate': (0.001, 0.1)},
                                  'BoundaryAttack': {'delta': (0.01, 0.5)},
                                  'SimBA': {'freq_dim': (1, 8), 'stride': (1, 8)},
                                  'AutoProjectedGradientDescent': {'eps_step': (0.1, 1)},
                                  'BrendelBethgeAttack': {'lr': (0.001, 0.1), 'lr_decay': (0.1, 1.0),
                                                          'momentum': (0.1, 1.0)},
                                  'DecisionTreeAttack': {'offset': (0.0001, 0.01)},
                                  'DeepFool': {'nb_grads': (1, 10)},
                                  'FrameSaliencyAttack': {'method': (
                                      'iterative_saliency', 'iterative_saliency_refresh',
                                      'one_shot')},
                                  'GeoDA': {'bin_search_tol': (0.001, 0.1),
                                            'lambda_param': (0.01, 1), 'sigma': (0.0001, 0.01)},
                                  'HighConfidenceLowUncertainty': {'conf': (0.7, 1),
                                                                   'unc_increase': (1, 200)},
                                  'HopSkipJump': {'max_eval': (1000, 20000),
                                                  'init_eval': (1, 1000),
                                                  'init_size': (1, 1000)},
                                  'LowProFool': {'threshold': (0.1, 1), 'eta': (0.1, 1),
                                                 'eta_decay': (0.5, 1)},
                                  'MomentumIterativeMethod': {'eps_step': (0.1, 1)},

                                  'PixelThreshold': {'es': (0, 1)},
                                  'SaliencyMapMethod': {
                                      'gamma': (0.001, 1), 'theta': (-1, 1)},
                                  'ShadowAttack': {'sigma': (0.001, 1), 'nb_steps': (1, 1000),
                                                   'learning_rate': (0.001, 0.1),
                                                   'lambda_c': (0.001, 1),
                                                   'lambda_s': (0.001, 1)},
                                  'SignOPTAttack': {'k': (10, 1000), 'alpha': (0.1, 1),
                                                    'beta': (0.0001, 0.01),
                                                    'num_trial': (1, 1000)},
                                  'SpatialTransformation': {'max_translation': (0, 100),
                                                            'max_rotation': (0, 180)}

                                  }

        def _optimize(attack, hyperparams, estimator):
            """
            This function is responsible for the optimization process of the attack hyperparameters.
            :param attack: The attack to optimize.
            :param hyperparams: The hyperparameters to optimize.
            :param estimator: The estimator to attack.
            return: The success rate of the attack.
            """
            HP = {k: int(v) if isinstance(v, np.int64) else v for k, v in
                  zip(attacks_parameters[attack.__name__], hyperparams)}

            if attack.__name__ == 'ZooAttack':
                self.check_zoo(attack, HP, estimator)

            attack = attack(estimator, **HP)
            attack_success_rate = 0
            num_batches = 0
            dataframe = self.dataloader.get_dataset()
            column_names = dataframe.columns
            for data, label in self.dataloader:
                data = Attacker.convert_to_numpy(tensor=data, framework=self.framework)
                generate_params = {}
                if self.framework == 'xgboost' or self.framework == 'sklearn' or self.framework == 'catboost':
                    # attach togeter x and y to one dataset

                    generate_params = {'dataset': dataframe, 'mask': None, 'columns_names': column_names}
                x_test_adv = attack.generate(data, **generate_params)
                attack_success_rate += Attacker.attack_eval(estimator=estimator, x_test=data, y_test=label,
                                                            adv_x=x_test_adv, ml_type=self.ml_type)

                num_batches += 1
            if num_batches == 0:
                return 0
            return -attack_success_rate / num_batches

        if optimize and attack.__name__ in attack_parameter_range:
            np.int = int
            search_space = [v for k, v in attack_parameter_range[attack.__name__].items()]
            func = lambda params: _optimize(attack, params, estimator=estimator)
            result = gp_minimize(func, search_space, n_calls=10)
            final_HP = {k: int(v) if isinstance(v, np.int64) else v for k, v in
                        zip(attacks_parameters[attack.__name__], result.x)}
            if attack.__name__ == 'ZooAttack':
                self.check_zoo(attack, final_HP, estimator)
            if "eps" in attack.attack_params:
                final_HP["eps"] = self.epsilon
            elif "epsilon" in attack.attack_params:
                final_HP["epsilon"] = self.epsilon

            optimized_attack = attack(estimator, **final_HP)
            return optimized_attack
        else:
            final_HP = {}
            if "eps" in attack.attack_params:
                final_HP["eps"] = self.epsilon
            elif "epsilon" in attack.attack_params:
                final_HP["epsilon"] = self.epsilon
            if attack.__name__ == 'ZooAttack':
                self.check_zoo(attack, final_HP, estimator)
            attack = attack(estimator, **final_HP)
            return attack

    def attack_estimator(self, optim_attack, dataloader):
        """
        This function is responsible for the attack process and creates adversarial examples.
        :param optim_attack: The attack to use.
        :param dataloader: The dataloader to use for the attack.
        return: The path to the adversarial examples.
        """
        attack_name = optim_attack.__str__().split('(')[0].split('.')[-1]
        adv_path = f"{get_run_files()}/adv_{attack_name}.npy"

        # Initialize an empty list to hold all adversarial examples

        all_adv_x = []

        for data, label in dataloader:
            data = Attacker.convert_to_numpy(framework=self.framework, tensor=data)
            # data = data.reshape()
            generate_params = {}
            if self.framework in ['xgboost', 'sklearn', 'catboost']:
                dataframe = dataloader.get_dataset()
                column_names = dataframe.columns
                # attach togeter x and y to one dataset
                generate_params = {'dataset': dataframe, 'mask': None, 'columns_names': column_names}
            adv_x = optim_attack.generate(data , **generate_params)
            # if estimator.preprocessing_defences and len(estimator.preprocessing_defences) > 0:
            #     for defense in estimator.preprocessing_defences:
            #         adv_x = defense(adv_x, np.array(label))
            # check if adv_x is a tuple:
            if isinstance(adv_x, tuple):
                adv_x = adv_x[0]
            try:
                adv_x = np.stack(adv_x)  # Stack elements along a new axis if adv_x is a list of arrays
            except Exception as e:
                print("Inconsistent shapes in adv_x. Check individual shapes or reshape as needed.")
                raise e
                # Handle reshaping or padding here if necessary

                # Append the batch to the all_adv_x list

            all_adv_x.append(adv_x)

                # Concatenate all batches into one array

        all_adv_x = np.concatenate(all_adv_x, axis=0)

        # Save the full concatenated array to disk

        np.save(adv_path, all_adv_x)

        print(f"Adversarial examples saved to {adv_path}")
        return adv_path


class Defender:
    """
    This class is responsible for the defense process.
    Attributes:
    ----------
    estimator: The estimator to defend.
    dataloader: The dataloader to use for the defense.
    defenses: The defenses to use.
    ml_type: The type of the estimator (classification/regression).

    Methods:
    --------
    optimize_post_processor_optimization(): Optimize the hyperparameters of a post processor defense.
    optimize_pre_processor_optimization(): Optimize the hyperparameters of a pre processor defense.
    optimize_defense_hyperparameters(): Optimize the hyperparameters of a defense.

    """

    def __init__(self, dataloader, defenses, ml_type):
        """
        Initialize a Defender instance.
        """
        self.dataloader = dataloader
        self.defenses = defenses
        self.ml_type = ml_type

    def optimize_post_processor_optimization(self,estimator,  defense, adv_examples_path):
        """
        This function only responsible for optimize the hyper parameters a post processor type defense.
        :param defense: the defense we wish to optimize its hyper paraemeters.
        :param adv_examples_path: the adversarial examples we want to defend from.
        :return: the optimized defense instance.
        """

        def _optimize(defense_to_optimize, hyperparams):
            if isinstance(hyperparams[0], np.int64):  # Only for the Rounded hyperparamer - it doesnt work with
                # np.int64.
                hyperparams[0] = int(hyperparams[0])
            HP = {k: v for k, v in zip(self.defenses_hyperparameters[defense], hyperparams)}
            defense_to_optimize = defense_to_optimize(**HP)

            estimator.postprocessing_defences = old_defs + [
                defense_to_optimize]  # each time we will add the next defense on to

            iter_dataloader = iter(self.dataloader)
            batches = 0
            with open(adv_examples_path, 'rb') as f:
                size_max = os.fstat(f.fileno()).st_size
                while f.tell() < size_max:
                    adv_examples = np.load(f)
                    x_clean, y_test = next(iter_dataloader,None)
                    attack_success_rate = Attacker.attack_eval(estimator, x_clean, y_test, adv_examples,
                                                               self.ml_type)
                    batches += 1
            if batches == 0:
                return 0
            return -attack_success_rate / batches

        search_space = [v for k, v in self.defenses_hyperparameters_ranges[defense].items()]
        old_defs = estimator.postprocessing_defences
        if old_defs is None:
            old_defs = []

        func = lambda params: _optimize(defense, params)
        result = gp_minimize(func, search_space, n_calls=20, acq_optimizer='sampling')

        estimator.postprocessing_defences = old_defs
        final_HP = {k: v for k, v in zip(self.defenses_hyperparameters[defense], result.x)}
        optimized_defense = defense(**final_HP)
        # self.optim__defense_param[optimized_defense.__name__] = final_HP
        return optimized_defense, final_HP

    def optimize_pre_processor_optimization(self,estimator, defense, adv_examples_path):
        """
        This function only responsible for optimize the hyper parameters a post processor type defense.
        :param defense: the defense we wish to optimize its hyper paraemeters.
        :param adv_examples_path: the path to the adversarial examples we want to defend from.
        :return: the optimized defense instance.
        """

        def _optimize(estimator, defense_to_optimize, hyperparams):
            """
            Optimize the defense hyperparameters by comparing respective batches from adversarial examples
            and the dataloader.
            :param defense_to_optimize: The defense to optimize.
            :param hyperparams: The hyperparameters to optimize.
            """
            try:
                # Handle np.int64 conversion for specific hyperparameters
                if isinstance(hyperparams[0], np.int64):  # Only for Rounded hyperparameter
                    hyperparams[0] = int(hyperparams[0])

                # Prepare the hyperparameters for the defense
                HP = {k: v for k, v in zip(self.defenses_hyperparameters[defense], hyperparams)}
                # if 'length' in defense_to_optimize.params:
                #     HP['length'] = int(estimator.input_shape[1] * .2)
                if 'estimator' in defense_to_optimize.params:
                    HP['estimator'] = estimator
                if 'num_classes' in defense_to_optimize.params:
                    HP['num_classes'] = estimator.nb_classes
                if 'clip_values' in defense_to_optimize.params:
                    HP['clip_values'] = estimator.clip_values

                # Apply the defense with the optimized hyperparameters
                defense_instance = defense_to_optimize(**HP)
                estimator.preprocessing_defences = [defense_instance]

                # Initialize variables for attack success rate computation
                iter_dataloader = iter(self.dataloader)
                batches = 0
                attack_success_rate_total = 0

                # Load the adversarial examples and compare batch by batch
                adv_x_full = np.load(adv_examples_path)
                batch_size = len(next(iter(self.dataloader))[0])  # Get batch size from dataloader

                # Iterate through adversarial examples in batch-sized chunks
                for i in range(0, len(adv_x_full), batch_size):
                    # Extract the corresponding batch from adv_x
                    adv_x_batch = adv_x_full[i:i + batch_size]

                    # Get the corresponding clean batch from the dataloader
                    x_clean, y_test = next(iter_dataloader, (None, None))

                    # Check if dataloader runs out of batches
                    if x_clean is None or y_test is None:
                        break

                    # Adjust for partial batches
                    if len(adv_x_batch) != len(x_clean):
                        print(f"Partial batch detected: adv_x_batch={len(adv_x_batch)}, x_clean={len(x_clean)}")
                        adv_x_batch = adv_x_batch[:len(x_clean)]

                    # Compute attack success rate for the batch
                    attack_success_rate = Attacker.attack_eval(
                        estimator,
                        x_clean,
                        y_test,
                        adv_x_batch,
                        self.ml_type
                    )
                    attack_success_rate_total += attack_success_rate
                    batches += 1

                if batches == 0:
                    return 0

                # Return the negative of the average attack success rate (for optimization purposes)
                return -attack_success_rate_total / batches

            except Exception as e:
                print(f"Error during optimization: {e}")
                raise


        search_space = [v for k, v in self.defenses_hyperparameters_ranges[defense].items()]
        print("here post")
        func = lambda params: _optimize(estimator, defense, params)
        result = gp_minimize(func, search_space, n_calls=20, acq_optimizer='sampling')
        final_HP = {k: v for k, v in zip(self.defenses_hyperparameters[defense], result.x)}
        # if 'length' in defense.params:
        #     final_HP['length'] = int(estimator.input_shape[1] * .2)
        if 'estimator' in defense.params:
            final_HP['estimator'] = estimator
        if 'num_classes' in defense.params:
            final_HP['num_classes'] = estimator.nb_classes  # MNIST
        if 'clip_values' in defense.params:
            final_HP['clip_values'] = estimator.clip_values
        optimized_defense = defense(**final_HP)
        return optimized_defense, final_HP

    def optimize_defense_hyperparameters(self,estimator, defense, adv_examples_path):
        """
        This function checks the type of the defense in order to send it to the specific optimization function.
        :param defense: the defense we wish to optimize its hyper paraemeters.
        :param adv_examples_path: path to the adversarial examples we want to defend from
        :return: an instance of the optimized defense.
        """
        self.defenses_hyperparameters = {GaussianNoise: ['scale'], ReverseSigmoid: ['beta'], Rounded: ['decimals'],
                                         HighConfidence: ['cutoff'], CutMix: ['alpha', 'probability'], Cutout: ['length'],
                                         FeatureSqueezing: ['bit_depth'], GaussianAugmentation: ['sigma', 'ratio'],
                                         LabelSmoothing: ['max_value'], TotalVarMin: ['norm', 'prob'],
                                         JpegCompression: ['channels_first'],
                                         ClassLabels: ['apply_predict'], SpatialSmoothing: ['window_size']}

        self.defenses_hyperparameters_ranges = {TotalVarMin: {'norm': Integer(1, 2), 'prob': (0.1, 0.7)},
                                                GaussianNoise: {'scale': (0.001, 10)},
                                                ReverseSigmoid: {'beta': (0.001, 10), 'gamma': (0.001, 10)},
                                                Rounded: {'decimals': Integer(1, 5)},
                                                HighConfidence: {'cutoff': (0.000001, 1)},
                                                CutMix: {'alpha': (0.1, 0.5), 'probability': (0.3, 0.7)},
                                                Cutout: {'length': Integer(int(estimator.input_shape[1]*.1), int(estimator.input_shape[1]*.7)) },
                                                FeatureSqueezing: {'bit_depth': Integer(1, 8)},
                                                GaussianAugmentation: {'sigma': (0.001, 1), 'ratio': (0.1, 1)},
                                                LabelSmoothing: {'max_value': (0.5, 0.9)}, # assumes that model returns softmax no log prob or logits
                                                JpegCompression: {'channels_first': Categorical([False, True])},
                                                ClassLabels: {'apply_predict': Categorical([True])},

                                                SpatialSmoothing: {'window_size': Integer(1, 6)}}
        defenses_objects = [d['obj'] for d in self.defenses]
        if len(defense.params) == 0:
            return defense()
        elif defense in defenses_objects and issubclass(defense, Postprocessor):
            return self.optimize_post_processor_optimization(estimator, defense, adv_examples_path)
        elif defense in defenses_objects and issubclass(defense, Preprocessor):
            return self.optimize_pre_processor_optimization(estimator, defense, adv_examples_path)


class ModelEvaluator:
    """
    This class is responsible for the evaluation process of the model.
    Attributes:
    ----------
    estimator: The estimator to evaluate.
    dataloader: The dataloader to use for the evaluation.
    attacks: The attacks to use for the evaluation.
    defenses: The defenses to use for the evaluation.
    ml_type: The type of the estimator (classification/regression).
    framework: The framework used for the estimator (e.g., 'pytorch', 'tensorflow', 'sklearn').
    opt_params_defense: A boolean indicating whether to optimize the hyperparameters of the defense.
    opt_params_attack: A boolean indicating whether to optimize the hyperparameters of the attack.
    epsilon: The epsilon value for the attack.
    defender: An instance of the Defender class.
    attacker: An instance of the Attacker class.

    Methods:
    --------
    copy_tf_estimator(): Copy a TensorFlow estimator.
    calculate_metrics(): Calculate the metrics of the model.
    get_adv_predictions(): Get the predictions of the adversarial examples.
    get_y(): Get the labels of the data.
    calculate_attack_success_rate(): Calculate the success rate of the attack.
    get_pred_size(): Get the size of the predictions.
    set_y(): Sets the labels of the data.
    attack_evaluation(): Evaluate the success of an attack.
    shuffle_adv(): Shuffle the adversarial examples.
    defense_evaluation(): Evaluate the success of a defense.
    evaluate(): Evaluate the model.
    """

    def __init__(self, estimator, dataloader, attacks, defenses, ml_type, framework, opt_params_defense,
                 opt_params_attack, epsilon, max_attack_iterations):

        self.estimator = estimator
        self.dataloader = dataloader
        self.attacks = [attack['obj'] for attack in attacks if attack['obj'] is not None]
        print(self.attacks)
        self.defenses = [defense['obj'] for defense in defenses if defense['obj'] is not None]
        print(self.defenses)
        self.ml_type = ml_type
        self.framework = framework
        self.opt_params_defense = opt_params_defense
        self.opt_params_attack = opt_params_attack
        self.max_attack_iterations = max_attack_iterations
        self.defender = Defender(dataloader, defenses, ml_type)
        self.attacker = Attacker(dataloader=dataloader, ml_type=ml_type, framework=framework,
                                 opt_params_attack=opt_params_attack, epsilon=epsilon)

    @staticmethod
    def copy_tf_estimator(estimator, preprocess_defenses=None, postprocess_defenses= None):
        """
        Copy a TensorFlow estimator.
        """
        from art.estimators.classification.tensorflow import TensorFlowV2Classifier
        # from art.estimators.classification.pytorch import PyTorchClassifier

        model = estimator._model
        loss = estimator._loss_object
        optimizer = estimator._optimizer
        nb_classes = estimator.nb_classes
        clip_values = estimator._clip_values
        input_shape = estimator._input_shape
        preprocess_defenses = preprocess_defenses
        postprocess_defenses = postprocess_defenses

        new_estimator = TensorFlowV2Classifier(model=model, nb_classes=nb_classes, clip_values=clip_values,
                                               input_shape=input_shape, postprocessing_defences=postprocess_defenses,
                                               preprocessing_defences=preprocess_defenses, loss_object=loss,
                                               optimizer=optimizer)
        return new_estimator

    def calculate_metrics(self, estimator, y_prob,y_true=None):
        """
        Calculate the metrics of the model.
        parameters:
        y_prob: an m-by-c np array of the c class probabilities for each of the m samples.
        return: The metrics of the model, evaluation dictionary.
        """
        def get_metric(metric=None):
            """
            Get a specific metric.
            """
            if metric and metric == "auc":

                # Binarize true labels and predicted labels
                y_true_bin = label_binarize(y_true, classes=classes)
                # y_pred_bin = label_binarize(y_pred, classes=classes)

                all_auc_scores = []
                all_fpr = []
                all_tpr = []
                all_thresholds = []

                for i in range(n_classes):
                    # compute for i-th class

                    if i not in classes:  # this class is not in the data

                        all_fpr.append({f"class_f{classes[i]}": list(np.nan)})

                        all_tpr.append({f"class_f{classes[i]}": list(np.nan)})

                        all_thresholds.append(

                            {f"class_f{classes[i]}": list(np.nan)})

                        continue
                    # Calculate ROC curve for the i-th class
                    fpr_i, tpr_i, thresholds_i = roc_curve(y_true_bin[:, i], y_prob[:, i])

                    # Calculate AUC for the i-th class
                    try:
                        auc_score_i = roc_auc_score(y_true_bin[:, i], y_prob[:, i])
                    except Exception as err:
                        # logging.error(f"Could not get any acu for class {i} Error:\n{err}")

                        # traceback.format_exc()

                        auc_score_i = np.nan
                    if n_classes == 2:
                        return round(auc_score_i, 3), tpr_i.round(3), fpr_i.round(3), thresholds_i.round(3)
                    # Append results to lists
                    all_fpr.append({f"class_f{classes[i]}": list(np.nan_to_num(fpr_i.round(3).astype(float)))})
                    all_tpr.append({f"class_f{classes[i]}": list(np.nan_to_num(tpr_i.round(3).astype(float)))})
                    all_thresholds.append(
                        {f"class_f{classes[i]}": list(np.nan_to_num(thresholds_i.round(3).astype(float)))})
                    all_auc_scores.append(round(float(auc_score_i), 3))

                # Return the average AUC, TPR, FPR, and thresholds across all classes
                return np.nanmean(all_auc_scores), all_tpr, all_fpr, all_thresholds
            else:
                metric_map = {"accuracy": round(accuracy_score(y_true=y_true, y_pred=y_pred), 3),
                              "precision": round(precision_score(y_true=y_true, y_pred=y_pred, average="weighted",
                                                                 zero_division=0), 3),
                              "recall": round(recall_score(y_true=y_true, y_pred=y_pred, average="weighted"), 3),
                              "f1score": round(f1_score(y_true=y_true, y_pred=y_pred, average="weighted"), 3),
                              }
                print("metric_map: ",metric_map)

                return metric_map

        def get_report_per_class():
            """
            Get the classification report per class.
            return: The classification report per class.
            """
            target_names = [f"class_{class_i}" for class_i in classes]
            report = classification_report(y_true=y_true, y_pred=y_pred, labels=classes, target_names=target_names,
                                           output_dict=True, zero_division=0)
            precision_class = []
            recall_class = []
            f1score_class = []
            for target_name in target_names:
                class_report = report[target_name]
                precision_class.append({target_name: float(round(class_report['precision'], 3))})
                recall_class.append({target_name: float(round(class_report['recall'], 3))})
                f1score_class.append(({target_name: float(round(class_report['f1-score'], 3))}))
            cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
            accuracy_class = cm.diagonal() / (cm.sum(axis=1) + 10 ** -6)
            accuracy_class = list(map(lambda x: float(round(x, 3)), accuracy_class))
            accuracy_class = zip(target_names, accuracy_class)
            accuracy_class = list(map(lambda x: {x[0]: x[1]}, accuracy_class))

            return accuracy_class, precision_class, recall_class, f1score_class

        if y_true is None:
            y_true = self.get_y()
        classes = np.arange(estimator.nb_classes)  # fix, was based on subset before

        n_classes = len(classes)

        y_pred = np.argmax(y_prob, axis=1)
        if n_classes == 1:
            raise Exception("Model evaluation failed because Y set has only one label")
        try:
            accuracy_class, precision_class, recall_class, f1score_class = get_report_per_class()
        except Exception as err:
            print(err)
            logging.error(err)
            traceback.format_exc()
            accuracy_class, precision_class, recall_class, f1score_class = None, None, None, None
        try:
            acu_score, fpr, tpr, thresholds = get_metric("auc")
        except Exception as err:
            logging.error(err)
            traceback.format_exc()
            acu_score, fpr, tpr, thresholds = None, [], [], []
        try:
            accuracy, precision, recall, f1 = get_metric().values()
        except Exception as err:
            logging.error(err)
            traceback.format_exc()
            accuracy, precision, recall, f1 = None, None, None, None

        eval_dict = {"accuracy": accuracy,
                     "precision": precision,
                     "recall": recall,
                     "f1score": f1,
                     "auc": acu_score,
                     "thresholds": list(thresholds),
                     "tprs": list(tpr),
                     "fprs": list(fpr),
                     "accuracy_class": accuracy_class,  # accuracy per class
                     "precision_class": precision_class,
                     "recall_class": recall_class,
                     "f1score_class": f1score_class,
                  }
        print("eval_dict: ",eval_dict)
        return eval_dict

    def get_adv_predictions(self, estimator, adv_examples_path):
        """
        Get the predictions of the adversarial examples.
        parameters:
        estimator: The estimator to use for the predictions.
        adv_examples_path: The path to the adversarial examples.
        return: The predictions of the adversarial examples.
        """
        predictions = []

        y_values = self.get_y()
        y_values = y_values
        y_range = (min(y_values), max(y_values))

        with open(adv_examples_path, 'rb') as f:
            size_max = os.fstat(f.fileno()).st_size
            while f.tell() < size_max:
                adv_x = np.load(f)
                pred = estimator.predict(adv_x)
                predictions.append(pred)
                # if torch.any(torch.isnan(torch.Tensor(adv_x))) or  torch.any(torch.isnan(torch.Tensor(pred))):
                #     print(pred, adv_x)

                # pred_size = self.get_pred_size(pred)
                # if pred_size > 1:
                #     # if pred_size - 1 == y_range[1] and all([x >= 0 and x <= 1 for p in pred for x in p]):
                #     # pred = [np.argmax(p) for p in pred]
                #     pred = np.argmax(pred,axis=1)

                # else:

                # for prediction in pred:
                #     predictions.append(prediction)
                # predictions += pred
        return np.vstack(predictions)

    def set_y(self, y_true):

        self.y_true = y_true

    def get_y(self):  # fixed

        """

        get the true labels of the dataset.

        """

        return np.array(self.y_true)
    # def get_y(self):
    #     """
    #     get the true labels of the dataset.
    #     """
    #     y_true = []
    #     for data, label in self.dataloader:
    #         if self.framework == 'pytorch':
    #             import torch
    #             if isinstance(label, torch.Tensor):
    #                 label = [l.item() for l in label]
    #         else:
    #             label = [l for l in np.array(label)]
    #         y_true += label
    #     print(y_true)
    #     return np.array(y_true)

    def calculate_attack_success_rate(self, estimator, adv_examples_path):
        """
        Calculate the attack success rate.
        Parameters:
        adv_examples_path: The path to the adversarial examples.
        Returns: The attack success rate.
        """
        try:
            # Load the full set of adversarial examples
            adv_x_full = np.load(adv_examples_path)
            # print(f"Loaded adversarial examples: {adv_x_full.shape}")

            # Initialize variables
            asr = 0
            batches = 0

            # Ensure dataloader and adversarial examples are consistent
            iter_loader = iter(self.dataloader)
            batch_size = len(next(iter(self.dataloader))[0])  # Get batch size from dataloader

            # Iterate through the adversarial examples in batch-sized chunks
            for i in range(0, len(adv_x_full), batch_size):
                # Extract the corresponding batch from adv_x
                adv_x_batch = adv_x_full[i:i + batch_size]

                # Get the corresponding batch from the dataloader
                try:
                    x_test, y_test = next(iter_loader)
                except StopIteration:
                    # Handle cases where the dataloader has no more batches but adv_x does
                    print(f"Partial batch at index {i}. Skipping further adversarial examples.")
                    break

                # Adjust for partial batches in the last chunk
                if len(adv_x_batch) != len(x_test):
                    print(f"Partial batch detected: adv_x_batch={len(adv_x_batch)}, x_test={len(x_test)}")
                    adv_x_batch = adv_x_batch[:len(x_test)]

                # Compute success for this batch
                batch_asr = compute_success(
                    classifier=estimator,
                    x_clean=x_test,
                    labels=y_test,
                    x_adv=adv_x_batch
                )
                asr += batch_asr
                batches += 1

            if batches == 0:
                return 0

            return asr / batches

        except Exception as e:
            print(f"Error calculating attack success rate: {e}")
            raise



    def get_pred_size(self, pred):
        """
        Get the size of the predictions.
        parameters:
        pred: The predictions.
        return: The size of the predictions.
        """
        try:

            pred_size = pred.shape[1]
        except IndexError as index_err:
            logging.error(f"INDEX Error occurred while trying to get pred size .\n Error: {index_err}")
            traceback.format_exc()
            pred_size = 1
        except Exception as err:
            logging.error(f"Error occurred while trying to get pred size .\n Error: {err}")
            traceback.format_exc()
            raise Exception("Could not get pred shape").with_traceback(err.__traceback__)
        return pred_size

    def attack_evaluation(self, estimator, optimize = False):
        """
        This function is responsible for the attack evaluation process.
        :param estimator: The estimator to use for the attack evaluation.
        :param optimize: A boolean indicating whether to optimize the attack hyperparameters.
        return: The evaluation of the attacks, the paths to the adversarial examples, and the number of predictions.
        """
        # import func_timeout
        optim_attacks = []
        eval_attacks = {}
        adv_paths = []
        try:
            if len(self.attacks) == 0:
                raise Exception("No attacks to optimize")
            for attack in self.attacks:
                optim_attacks.append(self.attacker.optimize_evasion_attack(attack=attack, estimator=estimator,optimize=optimize))
            for opt_attack in optim_attacks:
                attack_name = opt_attack.__str__().split('(')[0]
                print("attacking: ", attack_name)
                max_iterations = self.max_attack_iterations.get(attack_name, 0)
                print("max_iterations: ", max_iterations)
                if max_iterations > 0:
                    opt_attack.max_iter = max_iterations
                adv_path = self.attacker.attack_estimator(optim_attack=opt_attack, dataloader=self.dataloader)
                adv_paths.append(adv_path)
                y_prob = self.get_adv_predictions(estimator=estimator, adv_examples_path=adv_path)
                attack_evaluation = self.calculate_metrics(estimator, y_prob=y_prob)
                attack_success_rate = self.calculate_attack_success_rate(estimator, adv_examples_path=adv_path)
                attack_evaluation['attack_success_rate'] = attack_success_rate

                eval_attacks[attack_name] = attack_evaluation
            print("finished attacking")
            return eval_attacks, adv_paths, len(y_prob)
        except Exception as err:
            print(f"Error occurred while trying to evaluate the attack .\n Error: {err}")
            traceback.format_exc()
            raise Exception("Could not evaluate the attack").with_traceback(err.__traceback__)


    def shuffle_adv(self, adv_paths):
        """
        Create a new shuffled npy file where each i-th sample is selected randomly
        from the i-th entry of the loaded npy files, and save it to the same directory
        as the input files.

        :param adv_paths: List of paths to the npy files.
        :return: Path to the saved shuffled npy file.
        """
        try:
            # Load all the npy files into memory
            all_samples = [np.load(path) for path in adv_paths]

            # Ensure all files have the same number of samples
            num_samples = len(all_samples[0])
            if not all(len(samples) == num_samples for samples in all_samples):
                raise ValueError("All npy files must have the same number of samples.")

            # Create the shuffled output
            shuffled_samples = []
            for i in range(num_samples):
                # Collect the i-th sample from each file
                ith_samples = [samples[i] for samples in all_samples]

                # Randomly select one of the i-th samples
                shuffled_samples.append(random.choice(ith_samples))

            # Save the shuffled samples to the same directory as the input files
            save_directory = os.path.dirname(adv_paths[0])
            output_path = os.path.join(save_directory, "shuffled_output.npy")
            np.save(output_path, np.array(shuffled_samples))

            print(f"Shuffled file saved to {output_path}")
            return output_path

        except Exception as e:
            print(f"Error while creating shuffled npy file: {e}")
            raise

    def clone_with_updates(self, estimator, **updated_params):
        """
        Create a copy of the given estimator and initialize it with updated parameters.

        :param estimator: The original estimator to copy.
        :param updated_params: Parameters to override in the copied estimator.
        :return: A new estimator object with updated parameters.
        """
        # Get the original parameters of the estimator (assumes the estimator exposes them via __dict__)
        init_signature = inspect.signature(estimator.__class__.__init__)
        init_params = init_signature.parameters

        # Extract valid parameters for the constructor
        valid_params = {}
        for key, value in estimator.__dict__.items():
            # Convert private attributes (_attr) to their public counterparts (attr)
            if key == '_input_shape' and 'nb_features' in init_params:
                key = '_nb_features'
                value = value[0]
            public_key = key.lstrip('_')
            if public_key in init_params:
                # If the parameter is a ModelWrapper, extract the model
                if public_key == 'model' and hasattr(value, '_model'):
                    value = value._model
                valid_params[public_key] = value

        # Update the valid parameters with the provided overrides
        valid_params.update(updated_params)

        # Reinitialize the estimator with the updated parameters
        return estimator.__class__(**valid_params)
    def defense_evaluation(self,estimator, defense, adv_path):
        """
        This function is responsible for the defense evaluation process.
        :param defense: The defense to evaluate.
        :param shuffle_adv_path: The path to the shuffled adversarial examples.
        return: The evaluation of the defense.
        """
        try:
            # step 2 - defense evaluation
            if self.opt_params_defense:
                print("optimizing: ", defense.__name__)
                optim_defense, HP = self.defender.optimize_defense_hyperparameters(estimator=estimator,defense=defense,
                                                                                   adv_examples_path=adv_path)
            else:
                HP = {}
                if 'length' in defense.params:
                    HP['length'] = int(estimator.input_shape[1] * .2)
                if 'estimator' in defense.params:
                    HP['estimator'] = estimator
                if 'num_classes' in defense.params:
                    HP['num_classes'] = estimator.nb_classes  # MNIST
                if 'clip_values' in defense.params:
                    HP['clip_values'] = estimator.clip_values
                optim_defense = defense(**HP)

            preprocessing_defences, postprocessing_defences = None, None
            if issubclass(type(optim_defense), Preprocessor):
                preprocessing_defences= [optim_defense]
                # estimator._update_preprocessing_operations()
            elif issubclass(type(optim_defense), Postprocessor):
                postprocessing_defences = [optim_defense]

            # preprocessing_defences = preprocessing_defences or []
            # postprocessing_defences = postprocessing_defences or []
            if self.framework == 'tensorflow':
                defense_estimator = ModelEvaluator.copy_tf_estimator(estimator, preprocessing_defences, postprocessing_defences)
            else:
                defense_estimator = self.clone_with_updates(
                    estimator,
                    preprocessing_defences=preprocessing_defences,
                    postprocessing_defences=postprocessing_defences
                )


            defense_report = {}
            defense_report["params"] = HP
            labels = []
            probs = []
            for data, label in self.dataloader:
                prob = defense_estimator.predict(data)  # Shape: batch_size x c

                probs.append(prob)

            probs = np.vstack(probs)

            clean_task_defense_eval = self.calculate_metrics(defense_estimator,probs)
            defense_report["defense_clean_task_performance"] = clean_task_defense_eval
            print("calculating metrics of static defense: ", defense.__name__)
            # defense_eval = {}
            # for attack in self.attacks:
            # #
            #     adv_path = self.attacker.attack_estimator(optim_attack=attack(estimator), dataloader=self.dataloader)
            #     y_pred = self.get_adv_predictions(estimator=estimator, adv_examples_path=adv_path)
            #     static_defense_res = self.calculate_metrics(y_pred=y_pred)
            #     defense_eval[attack(estimator).__str__().split('(')[0]] = static_defense_res

            # y_pred = self.get_adv_predictions(estimator=estimator, adv_examples_path=shuffle_adv_path)
            print("calculating metrics: ", defense.__name__)
            # optim_defense_evaluation = self.calculate_metrics(y_pred)
            attack_defense_unoptimized, _, _ = self.attack_evaluation(estimator=defense_estimator)
            defense_report["defense_evaluation"] = attack_defense_unoptimized
            # step 3 - optimize attacks on defense
            if self.opt_params_attack:
                print("attacking on defense:", defense.__name__)
                optimize_attacks_eval, _, _ = self.attack_evaluation(estimator=defense_estimator, optimize = self.opt_params_attack)
                defense_report["optimize_attacks_on_defense_reports"] = optimize_attacks_eval

            return defense_report
        except Exception as err:
            print(f"Error occurred while trying to evaluate the defense .\n Error: {err}")
            traceback.format_exc()
            raise Exception("Could not evaluate the defense").with_traceback(err.__traceback__)

    def evaluate(self):
        """
        This function is responsible for the evaluation process of the model and to call to all the evaluation methods.
        return: The reports of the evaluation.
        """

        print("starting clean model evaluation")
        ## Clean task Perfomance
        labels = []
        probs = []  # To store the m-by-c array of predictions
        for data, label in self.dataloader:
            # Collect raw probabilities (m-by-c array)
            prob = self.estimator.predict(data)  # Shape: batch_size x c
            probs.append(prob)
            # Collect ground truth labels
            if self.framework == 'pytorch':
                import torch
                if isinstance(label, torch.Tensor):
                    label = [l.item() for l in label]
            else:
                label = [l for l in np.array(label)]
            labels += label
            # Concatenate probabilities into a single m-by-c array
        probs = np.vstack(probs)
        # Convert labels and preds to NumPy arrays
        labels = np.array(labels)
        self.set_y(labels)
        clean_task = self.calculate_metrics(self.estimator, probs)
        print("clean model evaluation done", clean_task)
        reports = {}
        try:
            clean_model_eval, adv_paths, batches = self.attack_evaluation(estimator=self.estimator)
            shuffle_adv_path = self.shuffle_adv(adv_paths=adv_paths)

            print("going over all defenses", self.defenses)
            for d in self.defenses:
                print("evaluating: ", d.__name__)
                report = self.defense_evaluation(self.estimator, d, shuffle_adv_path)
                print(report)
                reports[d.__name__] = report
            clean_model_eval['Clean_task_perfomance'] = clean_task
            # Add the number_of_samples here
            number_of_samples = len(labels)
            clean_model_eval['Clean_task_perfomance']['dataset_statistics'] = {'number_of_samples': number_of_samples}
            reports["clean_model_evaluation"] = clean_model_eval
            return reports

        except Exception as err:
            # reports["error"] = err
            # reports["stack trace"] = traceback.format_exc()
            print(f"Error occurred while trying to evaluate the model .\n Error: {err}")
            raise Exception("Could not evaluate the model").with_traceback(err.__traceback__)
