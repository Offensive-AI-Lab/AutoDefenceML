import json
import logging
import traceback
import uuid
import os
from fastapi import BackgroundTasks, APIRouter
from fastapi.encoders import jsonable_encoder
from .request_classes import ModelEvalRequestBody
from google.cloud import pubsub_v1
from ...eval.user_files_eval.helpers import *
from .helpers import get_listening_subs, get_from_db, store_on_db, get_validation_id_from_requests
import func_timeout
from dotenv import load_dotenv
import time
from datetime import datetime
# from google.cloud import container_v1

load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")
TOPIC = os.getenv("EVAL_TOPIC")
TOPIC_PING = os.getenv("TOPIC_PING")
TOPIC_EVAL = os.getenv("TOPIC_EVAL")
FIRESTORE_DB = os.getenv("FIRESTORE_DB")
FIRESTORE_REPORTS_COLLECTION = os.getenv("FIRESTORE_REPORTS_COLLECTION")
FIRESTORE_ESTIMATOR_COLLECTION=os.getenv("FIRESTORE_ESTIMATOR_COLLECTION")
FIRESTORE_EVAL_STATUS_COLLECTION = os.getenv("FIRESTORE_EVAL_STATUS_COLLECTION")

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
        req_dir_path = get_req_files_package_root()
        # clean  dirs
        clean_dir(model_dir_path, model_dir_path)
        clean_dir(dataset_dir_path, dataloader_dir_path)
        clean_dir(dataloader_dir_path, dataloader_dir_path)
        clean_dir(loss_dir_path, loss_dir_path)
        clean_dir(req_dir_path, req_dir_path)
        # clean Estimator_params
    except FileNotFoundError:
        pass

eval_topic_path = 'projects/{project_id}/topics/{topic}'.format(
    project_id=PROJECT_ID,
    topic=TOPIC_EVAL,
)

publisher = pubsub_v1.PublisherClient()
# container_client = container_v1.ClusterManagerClient()

router = APIRouter()
# async def create_kubernetes_cluster():
#     # Define cluster parameters
#     project = PROJECT_ID
#     zone = "us-central1-a"  # Specify your preferred zone
#     cluster_name = f"evaluation-cluster-{str(uuid.uuid4())[:8]}"  # Generate a unique cluster name
#
#     # Define cluster configuration
#     cluster = {
#         "name": cluster_name,
#         "initial_node_count": 3,  # Adjust node count as needed
#         "node_config": {
#             "machine_type": "n1-standard-1",  # Specify machine type
#             "disk_size_gb": 100,  # Specify disk size
#             "oauth_scopes": [
#                 "https://www.googleapis.com/auth/cloud-platform"
#             ]
#         }
#     }
#
#     # Create the cluster
#     operation = await container_client.create_cluster(project=project, zone=zone, cluster=cluster)
#     operation.result()
#
#     # Return cluster name and zone
#     return cluster_name, zone
async def perform_evaluation(job_id, request):
    clean_env()
    request = request.dict()
    # Initializing list of listening subscribers (PUB\SUB architecture)
    subs = []
    try:
        print("Before calling get_listening_subs")  # debug log
        func_timeout.func_timeout(timeout=30,func=get_listening_subs,args=[subs])
        print(f"After calling get_listening_subs, subs: {subs}")  # debug log
    except func_timeout.FunctionTimedOut:
        print("get_listening_subs timed out")  # debug log
    except Exception as e:
        print(f"Error calling get_listening_subs: {str(e)}")
        print(traceback.format_exc())
    try:
        subs_amount = len(subs)
        print(f"subs_amount: {subs_amount}")  # debug log
        if subs_amount == 0:
            raise Exception("no subscriber is listening to the topic...")

        defenses = list(request["defense"].values())[0]
        print(f"defenses: {defenses}")  # debug log
        validation_id = request["validation_id"]
        estimator_params = get_from_db(project_id=PROJECT_ID, database=FIRESTORE_DB,
                                       collection_name=FIRESTORE_ESTIMATOR_COLLECTION, document_id=validation_id)
        if estimator_params is None:
             raise Exception(f"Document ID {validation_id} not found.")
        validation_id = get_validation_id_from_requests(
            project_id=PROJECT_ID,
            database=FIRESTORE_DB,
            document_id=job_id
        )

        def distribute_tasks(subscribers, tasks):

            # Create a dictionary to store assignments
            assignments = {subscriber: [] for subscriber in subscribers}
            print(f"assignments: {assignments}")  # debug log
            # Assign tasks in a round-robin manner
            subscriber_index = 0
            for task in tasks:
                subscriber = subscribers[subscriber_index]
                assignments[subscriber].append(task)
                subscriber_index = (subscriber_index + 1) % len(subscribers)

            # def extract_short_id(job_id: str) -> str:
            #     # Split by '-' and take the second part
            #     parts = job_id.split('-')
            #     if len(parts) > 1:
            #         # Take first 6 characters of the second part
            #         return parts[1][:6]
            #     return ""
            # print(f"Job_id attributes full1: {job_id}")

            # Publish tasks to topic with subscriber name as an attribute
            for subscriber, tasks in assignments.items():
                if not tasks:  # Skip if the tasks list is empty
                    logging.info(f"Skipping publishing for {subscriber} as tasks are empty")
                    continue
                try:
                    publisher.publish(
                        eval_topic_path,
                        json.dumps(request).encode("utf-8"),
                        target_subscription=subscriber,
                        target_defense=str(tasks).encode('utf8'),
                        estimator_params=json.dumps(estimator_params).encode("utf-8"),
                        job_id=job_id,
                        validation_id=validation_id
                    )
                except Exception as err:
                    logging.error(err)
                    logging.error(traceback.format_exc())
                    # sort subscribers by the amount of tasks assigned to them asc order
                    sorted_assigned_subs = sorted(list(assignments.keys()), key=lambda x: len(assignments[x]))
                    for i, sub in enumerate(sorted_assigned_subs):
                        try:
                            publisher.publish(
                                eval_topic_path,
                                json.dumps(request).encode("utf-8"),
                                target_subscription=sub,
                                target_defense=str(tasks).encode('utf8'),
                                estimator_params=json.dumps(estimator_params).encode("utf-8"),
                                job_id=job_id,
                                validation_id=validation_id
                            )
                            break
                        except Exception as err:
                            logging.error(err)
                            logging.error(traceback.format_exc())
                            if i == len(sorted_assigned_subs) - 1:
                                raise Exception("could not assign tasks to subscriber")
                            else:
                                continue

            return assignments

        distribute_tasks(subscribers=subs, tasks=defenses)

        evaluation_status = get_from_db(project_id=PROJECT_ID,
                                        database=FIRESTORE_DB,
                                        collection_name=FIRESTORE_EVAL_STATUS_COLLECTION,
                                        document_id=job_id)

        evaluation_status['process_stage'] = 'Evaluation'
        # evaluation_status['cluster_name'] = cluster_name  # Store cluster name in evaluation status
        # evaluation_status['zone'] = zone  # Store zone in evaluation status
        store_on_db(project_id=PROJECT_ID,
                    database=FIRESTORE_DB,
                    collection_name=FIRESTORE_EVAL_STATUS_COLLECTION,
                    document_key=job_id,
                    params=evaluation_status)

    except Exception as e:
        error_traceback = traceback.format_exc()
        evaluation_status = {"job_id": job_id, "process_status": "Failed",
                                     "process_stage": "None",
                                     "error": str(e),
                                     "stack trace": str(error_traceback),
                                     "report": None,
                             "pdf": None}


        store_on_db(project_id=PROJECT_ID,
                    database=FIRESTORE_DB,
                    collection_name=FIRESTORE_EVAL_STATUS_COLLECTION,
                    document_key=job_id,
                    params=evaluation_status)
        # clean_env()


async def perform_evaluation_wrapper(job_id, request):
    await perform_evaluation(job_id, request)

def set_eval_response(job_id, request):
    request = request.dict()
    evaluation_status = {"job_id": job_id, "process_status": 'Running',
                "process_stage": "Distributing",
                "error": None,
                "stack trace": None,
                "num_of_defenses": len(request["defense"]["class_name"]),
                "report": None,
                         "pdf": None,
                         "start_time": datetime.now().strftime("%d/%m/%Y %H:%M:%S")}

    store_on_db(project_id=PROJECT_ID,
                database=FIRESTORE_DB,
                collection_name=FIRESTORE_EVAL_STATUS_COLLECTION,
                document_key=job_id,
                params=evaluation_status)


@router.post("/evaluate/")
async def evaluate(request: ModelEvalRequestBody , background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())  # random uuid
    request_dict = request.dict()
    user_id = request_dict['user_id']
    job_id = user_id + "-" + job_id

    # Add by Eran Simtob for server use - start
    document = {
                "user_id": request.user_id,
                "job_id": job_id,
                "request": jsonable_encoder(request),
                "gpu_num": '',
                "request_type": 'evaluate'}
    store_on_db(project_id=PROJECT_ID,
                database=FIRESTORE_DB,
                collection_name="Requests",
                document_key=job_id,
                params=document)
    # Add by Eran Simtob for server use - end

    set_eval_response(job_id, request)
    background_tasks.add_task(perform_evaluation_wrapper, job_id, request)
    # Return the job ID to the client
    return {"job_id": job_id}
