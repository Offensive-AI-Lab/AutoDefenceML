import numpy as np
import pandas as pd
import inspect
import traceback
import logging
from art.attacks import EvasionAttack, InferenceAttack
import func_timeout
import datetime
import time
from art_handler.handler import ArtHandler
from art_attacks_plugin import *
from sklearn.preprocessing import label_binarize


class AttackDefenseValidator:
    """
        A utility class for validating compatible attacks and defenses for a given estimator.

        Attributes:
        ----------
        art_handler: An ArtHandler instance for handling ART-related functionality.
        metadata: Metadata related to the machine learning model.
        _file_loader: A FileLoader object for loading metadata.

        Methods:
        --------

        get_compatible_attacks(estimator): Get a list of compatible attacks for a given estimator.


        get_compatible_defenses(): Get a list of compatible defenses.

        """

    def __init__(self, fileloader_obj):
        """
                Initialize an AttackDefenseValidator instance.

                Parameters:
                -----------
                metadata (dict): Metadata related to the machine learning model.

                """
        print('Initializing AttackDefenseValidator. 39')
        self.art_handler = ArtHandler()
        self._file_loader = fileloader_obj
        self.metadata = fileloader_obj.metadata
        with open("attack_defense_validation-logs.txt", "w") as f:
            start_time = time.time()
            f.write(f"Starting validation at: {datetime.datetime.now()}\n")
        print("AttackDefenseValidator initialized. 46")


    def get_compatible_attacks(self, estimator):
        """
                Get a list of compatible attacks for a given estimator.

                Parameters:
                -----------
                estimator: An estimator object.

                Returns:
                --------
                list of dict: List of compatible attack dictionaries.

                """
        print("Fetching compatible attacks. 62")
        compatible_attacks = []
        attacks = self.art_handler.get("attack")
        print(attacks)
        Zoo =False
        dataloader = self._file_loader.get_dataloader()

        if self.metadata['ml_model']['meta']['framework'] == 'pytorch':
            sample_data_x, sample_data_y = next(iter(dataloader))
            # sample_data_x = sample_data_x[0].reshape(estimator.input_shape)
            # sample_data_x = sample_data_x.unsqueeze(0)
            print("Sample data for PyTorch loaded. 72")
        else:
            sample_data_x, sample_data_y = dataloader.get_next_batch()
            print("Sample data for other framework loaded. 75")
        for attack_dict in attacks:
            print(f"Evaluating attack {attack_dict['name']}. 77")
            # with open("attack_defense_validation-logs.txt", "a") as f:
            #     f.write(f"Starting to evaluate attack {attack_dict['name']} at : {datetime.datetime.now()}\n")
            adv = []
            try:
                print("Start Try. 82")
                attack = attack_dict['obj']

                if not issubclass(type(estimator), attack._estimator_requirements):
                    print(f"Attack {attack} if not compatible. 86")
                    continue
                # dataloader = self._file_loader.get_dataloader()
                #
                # if self.metadata['ml_model']['meta']['framework'] == 'pytorch':
                #     sample_data_x, sample_data_y = next(iter(dataloader))
                #     sample_data_x=sample_data_x[0].reshape(estimator.input_shape)
                # else:
                #     sample_data_x, sample_data_y = dataloader.get_next_batch()
                attack_function_inspect = dict(inspect.signature(attack).parameters)
                params = {}
                if attack_function_inspect.get('targeted'):
                    params['targeted'] = False
                # Zoo attack parameter 'nb_parallel' depends on the dataset size , check for different values
                if attack_dict['name'] == 'ZooAttack':
                    for param in [1, 10, 100]:
                        try:
                            params['nb_parallel'] = param
                            attack_init = attack(classifier=estimator, **params)
                            # time_to_attack = (time.time() - T ) * len(dataloader)
                            func_timeout.func_timeout(timeout=20, func=attack_init.generate, args=[np.array(sample_data_x)])
                            Zoo = True
                            break
                        except func_timeout.exceptions.FunctionTimedOut:
                            print(f"Zoo attack timed out with nb_parallel={param}. 110")
                            pass
                        except Exception as err:
                            logging.error(f"Error occurred while getting compatible attacks 113.\n Error: {err}")
                            logging.error(traceback.format_exc())
                            pass
                    if Zoo:
                        attack_dict.pop('obj')
                        attack_dict["has_max_iter"] = True
                        compatible_attacks.append(attack_dict)
                        print(f"Zoo attack added to compatible attacks. 120")
                else:
                    if attack_function_inspect.get('estimator') is not None:
                        attack_init = attack(estimator=estimator, **params)
                    elif attack_function_inspect.get('classifier') is not None:
                        attack_init = attack(classifier=estimator, **params)
                    else:
                        continue

                    print('Line 129')
                    generate_params = {}
                    if self.metadata['ml_model']['meta']['framework'] == 'xgboost' or self.metadata['ml_model']['meta']['framework'] == 'sklearn' or self.metadata['ml_model']['meta']['framework'] == 'catboost':
                        #attach togeter x and y to one dataset

                        dataframe = dataloader.get_dataset()
                        column_names =  dataframe.columns
                        # dataframe.columns = column_names
                        generate_params = { 'dataset': dataframe , 'mask':None, 'columns_names':column_names}
                    if attack_dict['name'] == 'HopSkipJump':
                        print("generating hopskipjump")
                    func_timeout.func_timeout(timeout=2, func=attack_init.generate,
                                                  args=[np.array(sample_data_x)],kwargs=generate_params)

                    print('Attack finished ahead of time. 132')
                    attack_dict.pop('obj')
                    attack_dict["has_max_iter"] = bool(attack_function_inspect.get('max_iter'))
                    compatible_attacks.append(attack_dict)
                    #
                    # attack_dict.pop('obj')
                    # attack_dict["has_max_iter"] = bool(attack_function_inspect.get('max_iter'))
                    # compatible_attacks.append(attack_dict)

            except func_timeout.FunctionTimedOut:
                    attack_dict.pop('obj')
                    attack_dict["has_max_iter"] = bool(attack_function_inspect.get('max_iter'))
                    compatible_attacks.append(attack_dict)
                    print(f"Attack {attack_dict['name']} timed out. 145")
                    continue

            except Exception as err:
                logging.error(f"Error occurred while getting compatible attacks. 149\n Error: {err}")
                # with open("attack_defense_validation-logs.txt", "a") as f:
                #     f.write(f"Error on attack {attack_dict['name']} at : {datetime.datetime.now()},"
                #             f"Error: {err}")
                logging.error(traceback.format_exc())
                continue
            # finally:
            #     if len(adv) >  0:
            #         print('Attack finished ahead of time')
            #         attack_dict.pop('obj')
            #         attack_dict["has_max_iter"] = bool(attack_function_inspect.get('max_iter'))
            #         compatible_attacks.append(attack_dict)

        print("Compatible attacks fetching completed. 162")
        return compatible_attacks

    def get_compatible_defenses(self,estimator):
        """
                Get a list of compatible defenses.

                Returns:
                --------
                list of dict: List of compatible defense dictionaries.

                """
        print("Fetching compatible defenses. 174")
        defenses = []
        dataloader = self._file_loader.get_dataloader()
        if self.metadata['ml_model']['meta']['framework'] == 'pytorch':
            sample_data_x, sample_data_y = next(iter(dataloader))

            print("Sample data for PyTorch loaded. 72")
        else:
            sample_data_x, sample_data_y = dataloader.get_next_batch()
        for defense_dict in self.art_handler.get(('defense')):

            if defense_dict['type'] == 'preprocessor':
                obj = defense_dict['obj']
                try:
                    HP = {}
                    if 'length' in obj.params:
                        HP['length'] = int(estimator.input_shape[1]*.2)
                    if 'estimator' in obj.params:
                        HP['estimator'] = estimator
                    if 'num_classes' in obj.params:
                        HP['num_classes'] = estimator.nb_classes  # MNIST
                    if 'clip_values' in obj.params:
                        HP['clip_values'] = estimator.clip_values
                    defense = obj(**HP)
                    if defense_dict['name'] == 'LabelSmoothing':
                        classes = np.arange(estimator.nb_classes)
                        y_bin = label_binarize(np.array(sample_data_y), classes=classes)
                        func_timeout.func_timeout(timeout=4, func=defense,
                                                  args=[np.array(sample_data_x), y_bin])
                    else:
                        func_timeout.func_timeout(timeout=4, func=defense,
                                              args=[np.array(sample_data_x), np.array(sample_data_y)])
                    del defense_dict['obj']
                    defenses.append(defense_dict)
                    print("defense added to compatible defenses. 192")

                except func_timeout.exceptions.FunctionTimedOut:
                    del defense_dict['obj']
                    defenses.append(defense_dict)
                    print("defense added to compatible defenses. 192")
                    continue
                except Exception as err:
                    print(f"Error occurred while getting compatible attacks. 149\n Error: {err}")
                    # with open("attack_defense_validation-logs.txt", "a") as f:
                    #     f.write(f"Error on attack {attack_dict['name']} at : {datetime.datetime.now()},"
                    #             f"Error: {err}")
                    print(traceback.format_exc())
                    continue
            else:
                del defense_dict['obj']
                defenses.append(defense_dict)
        if self.metadata['ml_model']['meta']['framework'] == 'sklearn':
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.ensemble import RandomForestClassifier
            if isinstance(estimator.model, DecisionTreeClassifier):
                dict_monte = {
                    "description": "TTTS-Tree Test Time Simulation for Enhancing Decision Tree Robustness Against Adversarial Examples,' authored by Cohen Seffi, Arbili Ofir, Mirsky Yisroel, and Rokach Lior, and published in the proceedings of the AAAI 2024 conference. ",
                    "class_name": "TTTS.MonteCarloDecisionTreeClassifier",
                    "name": "MonteCarloDecisionTreeClassifier",
                    "type": "preprocessor"
                }
                defenses.append(dict_monte)
            elif isinstance(estimator.model, RandomForestClassifier):
                dict_monte = {"name": "MonteCarloRandomForestClassifier",
                              "class_name": "TTTS.MonteCarloRandomForestClassifier",
                              "description": "TTTS-Tree Test Time Simulation for Enhancing Decision Tree Robustness Against Adversarial Examples,' authored by Cohen Seffi, Arbili Ofir, Mirsky Yisroel, and Rokach Lior, and published in the proceedings of the AAAI 2024 conference. ",
                              "type": "preprocessor"}
                defenses.append(dict_monte)
        print("Compatible defenses fetching completed. 179")
        return defenses

# from sklearn.tree import DecisionTreeClassifier
# model = DecisionTreeClassifier(random_state=123)
# #train on iris dataset
# from sklearn.datasets import load_iris
# iris = load_iris()
# X = iris.data
# y = iris.target
# model.fit(X, y)
# #save to pkl
# import dill
# with open("model_iris.pkl", "wb") as f:
#     dill.dump(model, f)
#
# with open("model_iris.pkl", 'rb') as f:
#     loaded_obj = dill.load(f)
# from art.estimators.classification import SklearnClassifier
# estimator = SklearnClassifier(model=loaded_obj)
# if isinstance(estimator.model, DecisionTreeClassifier):
#     print('true')
#
# from TTTS import MonteCarloDecisionTreeClassifier
# #model parametes
# params = estimator.model.get_params()
#
# model = MonteCarloDecisionTreeClassifier(**params)
# dict = estimator.get_params()
# dict['model'] = model
#
# estimator(**dict)
#
#
