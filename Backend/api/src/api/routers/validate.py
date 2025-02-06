import os
import subprocess
import time

from fastapi import BackgroundTasks, APIRouter
from fastapi.encoders import jsonable_encoder
from .request_classes import ValidationRequestBody
from .helpers import clean_env, store_on_db, get_from_db
from ...pre_run.input_validation import InputValidator
from ...pre_run.model_validation import ModelDatasetValidator
from ...pre_run.estimator_match import EstimatorHandler
from ...pre_run.attack_defense_validation import AttackDefenseValidator
from ...user_files.helpers import *
from file_loader.file_handler import FileLoader
import uuid
import traceback
import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

load_dotenv()
router = APIRouter()

PROJECT_ID = os.getenv("PROJECT_ID")
TOPIC = os.getenv("EVAL_TOPIC")
TOPIC_PING = os.getenv("TOPIC_PING")
TOPIC_EVAL = os.getenv("TOPIC_EVAL")
FIRESTORE_DB = os.getenv("FIRESTORE_DB")
FIRESTORE_REPORTS_COLLECTION = os.getenv("FIRESTORE_REPORTS_COLLECTION")
FIRESTORE_VAL_STATUS_COLLECTION = os.getenv("FIRESTORE_VAL_STATUS_COLLECTION")

import ipaddress, struct
import random


# # Define the base CIDR and subnet size
# base_cidr = ipaddress.IPv4Network("10.0.1.0/24")
# subnet_size = 24  # This corresponds to /24 subnet mask

def random_ip(network):
    network = ipaddress.IPv4Network(network)
    network_int, = struct.unpack("!I", network.network_address.packed)  # make network address into an integer
    rand_bits = network.max_prefixlen - network.prefixlen  # calculate the needed bits for the host part
    rand_host_int = random.randint(0, 2 ** rand_bits - 1)  # generate random host part
    ip_address = ipaddress.IPv4Address(network_int + rand_host_int)  # combine the parts
    return ip_address.exploded


# def run_terraform(cluster_name, req_file, tf_dir):
#     # change the directory to the terraform files
#     root_dir = os.path.dirname(os.path.dirname(os.getcwd()))
#     os.chdir(os.path.join(root_dir, tf_dir))
#     print(os.getcwd())
#     # initialize the terraform
#     subprocess.run(["terraform", "init"], check=True)
#     try:
#         subprocess.run(["terraform", "apply", "-auto-approve", "-var", f"cluster_name={cluster_name}", "-var",
#                         f"req_file={req_file}"])
#     except subprocess.CalledProcessError as e:
#         logging.error(f"Error while running terraform: {e}")


def add_to_db(user_id, job_id, process_type, timestamp, cluster_name):
    # Add the job details to the MySQL database

    pass


def perform_validation_sync(job_id, request):
    try:
        clean_env()
        # Extract the request data as a dictionary
        request = request.dict()
        user_id = request.get("user_id")

        # Log the run ID for tracking
        print(f"job id: {job_id}")
        cluster_name = f"k8c-{job_id}"
        os.environ["FILES_PATH"] = os.getenv("FILES_PATH_VAL")
        # Todo add requirements.txt file
        fileloader = FileLoader(metadata=request,
                                path_to_files_dir=get_files_package_root(),
                                path_to_model_files_dir=get_model_package_root(),
                                path_to_dataset_files_dir=get_dataset_package_root(),
                                path_to_dataloader_files_dir=get_dataloader_package_root(),
                                path_to_loss_files_dir=get_loss_package_root(),
                                path_to_req_files_dir=get_req_files_package_root(),
                                from_bucket=os.getenv("FROM_BUCKET"),
                                bucket_name=os.getenv("BUCKET_NAME"),
                                account_service_key_name=os.getenv("ACCOUNT_SERVICE_KEY"))
        # req_file = fileloader.get_req_file()
        # # req_file = r"C:\Users\user\PycharmProjects\api\requirements-torch.txt"
        # tf_dir = "terraform_files" #todo change
        # # dev_subnet_cidr = random_ip("10.0.1.0/24")
        # # dev_k8_secondary_subnet_cidr =random_ip("10.0.2.0/24")
        # # dev_k8_service_secondary_subnet_cidr =random_ip("10.0.3.0/24")
        # run_terraform(cluster_name, req_file,tf_dir )
        #
        # timestamp = time.time()
        # # put inside mysql db
        # add_to_db(user_id, job_id, "model_validation", timestamp,cluster_name)
        #           # Create instances of InputValidator and ModelDatasetValidator
        input_validator = InputValidator(fileloader_obj=fileloader)
        model_dataset_validator = ModelDatasetValidator(fileloader_obj=fileloader)

        # Validate user input data
        if input_validator.validate():
            validation_status = get_from_db(project_id=PROJECT_ID,
                                            database=FIRESTORE_DB,
                                            collection_name=FIRESTORE_VAL_STATUS_COLLECTION,
                                            document_id=job_id)
            validation_status['process_stage'] = 'Model and dataset validation'
            store_on_db(project_id=PROJECT_ID,
                        database=FIRESTORE_DB,
                        collection_name=FIRESTORE_VAL_STATUS_COLLECTION,
                        document_key=job_id,
                        params=validation_status)
        else:
            raise Exception("User's input not valid")
        # Validate model and dataset compatibility
        if model_dataset_validator.validate():
            validation_status = get_from_db(project_id=PROJECT_ID,
                                            database=FIRESTORE_DB,
                                            collection_name=FIRESTORE_VAL_STATUS_COLLECTION,
                                            document_id=job_id)
            validation_status['process_stage'] = 'Finding compatible attacks and defenses'
            store_on_db(project_id=PROJECT_ID,
                        database=FIRESTORE_DB,
                        collection_name=FIRESTORE_VAL_STATUS_COLLECTION,
                        document_key=job_id,
                        params=validation_status)
        else:
            raise Exception("Model dose not match with the data set!")

        # Get the input parameters and wrap the ML model in an estimator
        input = input_validator.get_input()
        print(f"Input to wrapp estimator: {input}")
        wrapper = EstimatorHandler(input, fileloader_obj=fileloader)
        print(f"Starting to wrap model...")
        wrapper.wrap()
        print(f"Wrap went successfully...")
        estimator = fileloader.get_estimator()

        # Use AttackDefenseValidator to find compatible attacks and defenses
        attack_defense_validator = AttackDefenseValidator(fileloader_obj=fileloader)
        compatible_attacks = attack_defense_validator.get_compatible_attacks(estimator=estimator)
        compatible_defenses = attack_defense_validator.get_compatible_defenses(estimator=estimator)
        # Update validation_status with the results
        validation_status = {"job_id": job_id, "process_status": "Done",
                             "process_stage": "None",
                             "error": "",
                             "stack trace": "",
                             "compatible_attacks": compatible_attacks,
                             "compatible_defenses": compatible_defenses}
        store_on_db(project_id=PROJECT_ID,
                    database=FIRESTORE_DB,
                    collection_name=FIRESTORE_VAL_STATUS_COLLECTION,
                    document_key=job_id,
                    params=validation_status)

        print(f"Saved status : {validation_status}")
        # setting estimator params to firestore db
        with open(get_files_package_root() + "/Estimator_params.json", 'r') as f:
            estimator_params = json.load(f)
        store_on_db(project_id=os.getenv("PROJECT_ID"),
                    database=os.getenv("FIRESTORE_DB"),
                    collection_name=os.getenv("FIRESTORE_ESTIMATOR_COLLECTION"),
                    document_key=job_id,
                    params=estimator_params)
        clean_env()
    except Exception as e:
        # Handle exceptions and record error details
        error_traceback = traceback.format_exc()
        validation_status = {"job_id": job_id, "process_status": "Failed",
                             "process_stage": "None",
                             "error": str(e),
                             "stack trace": str(error_traceback),
                             "compatible_attacks": [],
                             "compatible_defenses": []}
        store_on_db(project_id=PROJECT_ID,
                    database=FIRESTORE_DB,
                    collection_name=FIRESTORE_VAL_STATUS_COLLECTION,
                    document_key=job_id,
                    params=validation_status)
        if validation_status["process_status"] == "Failed":
            clean_env()


async def perform_validation(job_id, request):
    """
       Asynchronous function responsible for performing a validation process.

       Parameters:
       - job_id (str): A unique identifier for the validation task.
       - request: An instance of ValidationRequestBody containing validation metadata.

       Returns:
       - None: The results of the validation process are stored in the validation_status dictionary.

      Note:
    - This function is asynchronous and runs in the background to avoid blocking the main thread.

       """

    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        await loop.run_in_executor(executor, perform_validation_sync, job_id, request)


def set_val_response(job_id):
    validation_status = {"job_id": job_id, "process_status": 'Running',
                         "process_stage": 'Building node pool',
                         "error": None,
                         "stack trace": None,
                         "compatible_attacks": None,
                         "compatible_defenses": None}

    store_on_db(project_id=PROJECT_ID,
                database=FIRESTORE_DB,
                collection_name=FIRESTORE_VAL_STATUS_COLLECTION,
                document_key=job_id,
                params=validation_status)


@router.post("/validatetest/")
async def validate(request: ValidationRequestBody, background_tasks: BackgroundTasks):
    job_idUuid = str(uuid.uuid4())  # random uuid
    request_dict = request.dict()
    user_id = request_dict['user_id']
    job_id = user_id + "-" + job_idUuid
    short_id = job_idUuid[:6]

    # Add by Eran Simtob for server use - start
    document = {"user_id": request.user_id,
                "job_id": job_id,
                "request": jsonable_encoder(request),
                "background_tasks": jsonable_encoder(background_tasks),
                "short_id": short_id,
                "request_type": 'validate'}
    store_on_db(project_id=PROJECT_ID,
                database=FIRESTORE_DB,
                collection_name="Requests",
                document_key=job_id,
                params=document)

    # Add by Eran Simtob for server use - end

    set_val_response(job_id)
    # timestamp = time.time()
    # #put inside mysql db
    # add_to_db(user_id,job_id, "model_validation",timestamp,
    # Call the asynchronous validation function with the job ID
    background_tasks.add_task(perform_validation, job_id, request)
    # Return the job ID to the client
    return {"job_id": job_id}
