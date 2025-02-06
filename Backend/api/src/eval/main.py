import os
import traceback
from .evaluation.get_subscriber import get_subscriber
from google.cloud import pubsub_v1
import logging
from .user_files_eval.helpers import *
from file_loader.file_handler import FileLoader
from art_handler.handler import ArtHandler
from art_attacks_plugin import *
from .evaluation.model_evaluation import ModelEvaluator
from .firestore import set_report
import json
from dotenv import load_dotenv
import concurrent.futures
from ..api.routers.helpers import  get_from_db
FIRESTORE_EVAL_STATUS_COLLECTION =  os.getenv("FIRESTORE_EVAL_STATUS_COLLECTION")

"""
This module is responsible for evaluating the model.
It listens to the evaluation topic and when a message arrives, it evaluates the model and sends the report to the firestore.
"""

load_dotenv()
PROJECT_ID = os.getenv("PROJECT_ID")
TOPIC_PING = os.getenv("TOPIC_PING")
TOPIC_EVAL = os.getenv("TOPIC_EVAL")
FIRESTORE_DB = os.getenv("FIRESTORE_DB")
FIRESTORE_COLLECTION = os.getenv("FIRESTORE_REPORTS_COLLECTION")
# os.environ["FILES_PATH"] = os.getenv("FILES_PATH_EVAL")
environment_id = os.getenv("ENVIRONMENT") or "MainServer"
if environment_id == "cloud_dev":
    environment_id = "MainServer"

def clean_env():
    def clean_dir(directory_path,top_level_directory):
        try:
            # Iterate through all items in the directory
            for item in os.listdir(directory_path):
                item_path = os.path.join(directory_path, item)

                # Check if the item is a file or directory
                if os.path.isfile(item_path):
                    # Delete the file if it's not __init__.py
                    if item != "__init__.py":
                        os.remove(item_path)
                elif os.path.isdir(item_path):
                    # Recursively call the function for subdirectories
                    clean_dir(item_path, top_level_directory)

            # Remove the directory itself if it's not the top-level directory
            if directory_path != top_level_directory:
                os.rmdir(directory_path)

            print(f"All files and directories in {directory_path} deleted, except __init__.py")
        except Exception as e:
            print(f"Error: {e}")
    try:
        # set paths
        model_dir_path = get_model_package_root()
        dataset_dir_path = get_dataset_package_root()
        dataloader_dir_path = get_dataloader_package_root()
        loss_dir_path = get_loss_package_root()
        req_dir_root = get_req_files_package_root()
        run_time_dir_path = get_run_files()
        # clean  dirs
        clean_dir(model_dir_path,model_dir_path)
        clean_dir(dataset_dir_path,dataset_dir_path)
        clean_dir(dataloader_dir_path,dataloader_dir_path)
        clean_dir(loss_dir_path,loss_dir_path)
        clean_dir(req_dir_root,req_dir_root)
        clean_dir(run_time_dir_path,run_time_dir_path)
        # clean Estimator_params
        user_file_dir_path = get_files_package_root()
        os.remove(user_file_dir_path + r"/Estimator_params.json")
    except FileNotFoundError:
        pass
def evaluate(request, job_id, defense):

    """
    Evaluates the model and sends the report to the firestore.
    :param request: The request to evaluate the model.
    :param job_id: The id of the job.
    :param defense: The defense to evaluate.
    :param defense: The defense to evaluate.
    :return: None
    """

    print("Enter evaluate")
    print("defense: ", defense)
    try:
        os.environ["FILES_PATH"] = os.getenv("FILES_PATH_EVAL")
        # Down load all of the content from GCP and store it locally
        fileloader = FileLoader(metadata=request,
                       path_to_files_dir=get_files_package_root(),
                       path_to_model_files_dir=get_model_package_root(),
                       path_to_dataloader_files_dir=get_dataloader_package_root(),
                       path_to_dataset_files_dir=get_dataset_package_root(),
                       path_to_loss_files_dir=get_loss_package_root(),
                       path_to_req_files_dir=get_req_files_package_root(),
                       from_bucket=os.getenv("FROM_BUCKET"),
                       bucket_name=os.getenv("BUCKET_NAME"),
                       account_service_key_name=os.getenv("ACCOUNT_SERVICE_KEY")
                       )
        
        # import ART and all of our plugins
        art_handler = ArtHandler()

        estimator = fileloader.get_estimator()
        dataloader = fileloader.get_dataloader()

        attacks = art_handler.get("attack", specifics=request['attacks'])
        print(attacks)
        defenses = art_handler.get("defense", specifics=defense)
        print(defenses)

        # special case for TTTS
        if not defenses:
            if request['defense']['class_name'] == "TTTS.MonteCarloDecisionTreeClassifier":
                from  TTTS import MonteCarloDecisionTreeClassifier
                defenses = [{"class_name": "TTTS.MonteCarloDecisionTreeClassifier" , "obj": MonteCarloDecisionTreeClassifier}]
            elif request['defense']['class_name'] == "TTTS.MonteCarloRandomForestClassifier":
                from TTTS import MonteCarloRandomForestClassifier
                defenses = [{"class_name": "TTTS.MonteCarloRandomForestClassifier", "obj": MonteCarloRandomForestClassifier}]
        
        ml_type = request['ml_model']['meta']['ml_type']
        framework = request['ml_model']['meta']['framework']
        # print(request)
        
        # get information on hyperparam optimization
        opt_params_defense = request['HyperparametersOptimization']['hyperparameters_optimization_defense']
        opt_params_attack = request['HyperparametersOptimization']['hyperparameters_optimization_attack']

        max_attack_iterations = request['HyperparametersOptimization']["max_attack_iterations"]
        epsilon = request['HyperparametersOptimization']['epsilon']
        clip_values = request['ml_model']['dim']['clip_values']
        range = clip_values[1] - clip_values[0]
        if epsilon == None:
            epsilon = 0.01  # default value for optimized epsilon
        new_epsilon = epsilon * range #normalize epsilon budget to feature range

        eval_object = ModelEvaluator(estimator=estimator, dataloader=dataloader,
                                     attacks=attacks, defenses=defenses,
                                     ml_type=ml_type, framework=framework, opt_params_defense=opt_params_defense,
                                     opt_params_attack=opt_params_attack, epsilon=new_epsilon,
                                     max_attack_iterations=max_attack_iterations)

        print("start....")
        report = eval_object.evaluate()
        print(report)

        set_report(project_id=PROJECT_ID,database=FIRESTORE_DB,collection_name=FIRESTORE_COLLECTION,
                   document_key=job_id,report=report)
        print("finished")
        evaluation_status = get_from_db(project_id=PROJECT_ID,
                                        database=FIRESTORE_DB,
                                        collection_name=FIRESTORE_EVAL_STATUS_COLLECTION,
                                        document_id=job_id)
        print("evaluation_status: ", evaluation_status)
        # if report is not None \
        #         and len(report.keys()) == evaluation_status["num_of_defenses"] + 1:
        clean_env()
    except Exception as e:
        print(e)
        error_traceback = traceback.format_exc()
        report = {d: str(error_traceback) for d in defense["class_name"]}
        report["clean_model_evaluation"] = ""
        print(error_traceback)
        set_report(project_id=PROJECT_ID,database=FIRESTORE_DB,collection_name=FIRESTORE_COLLECTION,
                   document_key=job_id,report=report)
        clean_env()
import logging

def start_pubsub_listener():
    print("Starting listener in node")
    
    def callback(message):
        try:
            print(f"Received message in callback")
            print(f"Message with attributes: {message.attributes}")
            
            if message.data.decode("utf8") == "ping":
                print("Ping received")
                publisher_ping = pubsub_v1.PublisherClient()
                ping_topic_name = f'projects/{PROJECT_ID}/topics/{TOPIC_PING}'
                publisher_ping.publish(ping_topic_name, eval_sub_name.encode('utf8'))
            elif message.data.decode("utf8") == "stop":
                print("Stopping process")
                clean_env()
            # רק אם לא ping, בדוק את המזהה
            validation_id = message.attributes.get("validation_id")
            # if not validation_id:
            #     print("Message without validation_id, ignoring")
            #     message.ack()
            #     return
            
            print(f"****** validation_id: {validation_id} environment_id: {environment_id}")
            # if validation_id and environment_id not in validation_id:
            #     print(f"Validation ID not found or does not match environment. validation_id: {validation_id} environment_id: {environment_id}")
            #     message.ack()
            #     return

            if message.attributes.get('target_subscription') == eval_sub_name:
                print(f"Processing job: {message.attributes['job_id']}")
                with open(get_files_package_root() + "/Estimator_params.json", 'w') as f:
                    json.dump(message.attributes["estimator_params"], f)
                dict_request = json.loads(message.data.decode("utf-8"))
                print("defense: ", {"class_name": eval(message.attributes["target_defense"])} )
                evaluate(dict_request, message.attributes["job_id"],
                         {"class_name": eval(message.attributes["target_defense"])})
            message.ack()
            
        except Exception as e:
            print(f"Error in callback: {str(e)}")
            print(traceback.format_exc())
            message.nack()

    try:
        eval_sub_name = get_subscriber(TOPIC_EVAL)
        print(f"Created subscriber: {eval_sub_name}")
        
        subscription_path = f'projects/{PROJECT_ID}/subscriptions/{eval_sub_name}'
        print(f"Using subscription path: {subscription_path}")
        
        subscriber = pubsub_v1.SubscriberClient()
        streaming_pull_features = subscriber.subscribe(subscription_path, callback=callback)
        print(f"Successfully subscribed to {subscription_path}")

        with subscriber:
            try:
                print("Starting to process messages...")
                streaming_pull_features.result()
            except Exception as e:
                print(f"Error in streaming_pull_features: {str(e)}")
                print(traceback.format_exc())
                streaming_pull_features.cancel()
                streaming_pull_features.result()
                
    except Exception as e:
        print(f"Error setting up subscriber: {str(e)}")
        print(traceback.format_exc())


