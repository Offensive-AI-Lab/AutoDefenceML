import os

import func_timeout
from fastapi import BackgroundTasks, APIRouter
from fastapi.encoders import jsonable_encoder
from .helpers import store_on_db, get_from_db
from .request_classes import DatasetEvaluationRequestBody
from dotenv import load_dotenv

import uuid
from ...data_eval.user_files_data.helpers import *
import traceback
import asyncio
import json
import logging

from ...data_eval.data_validator import FileHandlerDataset, DataValidator, evaluate
from concurrent.futures import ThreadPoolExecutor
load_dotenv()
router = APIRouter()
PROJECT_ID = os.getenv("PROJECT_ID")
# TOPIC = os.getenv("EVAL_TOPIC")
# TOPIC_PING = os.getenv("TOPIC_PING")
# TOPIC_EVAL = os.getenv("TOPIC_EVAL")
FIRESTORE_DB = os.getenv("FIRESTORE_DB")
FIRESTORE_REPORTS_COLLECTION = os.getenv("FIRESTORE_REPORTS_COLLECTION")
FIRESTORE_DATA_EVAL_STATUS_COLLECTION = os.getenv("FIRESTORE_DATA_EVAL_STATUS_COLLECTION")

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
        dataset_dir = get_dataset_package_root()
        dataloader_dir = get_dataloader_package_root()
        files_dir = get_files_package_root()
        clean_dir(dataset_dir,dataset_dir)
        clean_dir(dataloader_dir,dataloader_dir)
    except FileNotFoundError:
        pass
def perform_evaluation(job_id, request):
    request = request.dict()

    file_loader = FileHandlerDataset(request=request, path_to_files_dir=get_files_package_root(),
                                     path_to_dataset_files_dir=get_dataset_package_root(),
                                     path_to_dataloader_files_dir=get_dataloader_package_root(),
                                     from_bucket=os.getenv("FROM_BUCKET"),
                                     bucket_name=os.getenv("BUCKET_NAME"),
                                     account_service_key_name=os.getenv("ACCOUNT_SERVICE_KEY"))
    data_validator = DataValidator(file_loader)
    validate = data_validator.validate_dataset()
    dataframe = data_validator.get_dataframe()
    target_name = data_validator.get_target_name()
    try:
        data_eval_status = {"job_id": job_id, "process_status": 'Running',
                                  "process_stage": 'Data Poisoning',
                                  "error": None,
                                  "stack trace": None,
                                  "report": {},
                                  "pdf": None}
        report, pdf_string = evaluate(dataframe, target_name)
        if report is not None and pdf_string is not None:
            data_eval_status["report"] = report
            data_eval_status["pdf"] = pdf_string
            data_eval_status["process_status"] = 'Completed'
            store_on_db(project_id=PROJECT_ID,
                        database=FIRESTORE_DB,
                        collection_name=FIRESTORE_DATA_EVAL_STATUS_COLLECTION,
                        document_key=job_id,
                        params=data_eval_status)
            clean_env()
        else:
            raise ValueError("Error in data evaluation")
    except Exception as e:
        data_eval_status = {"job_id": job_id, "process_status": 'Failed',
                                  "process_stage": 'None',
                                  "error": str(e),
                                  "stack trace": traceback.format_exc(),
                                  "report": None,
                                  "pdf": None}
        store_on_db(project_id=PROJECT_ID,
                    database=FIRESTORE_DB,
                    collection_name=FIRESTORE_DATA_EVAL_STATUS_COLLECTION,
                    document_key=job_id,
                    params=data_eval_status)
        clean_env()





async def perform_eval_wrapper(job_id, request):
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        await loop.run_in_executor(executor, perform_evaluation, job_id, request)

def set_val_response(job_id):
    data_eval_status = {"job_id": job_id, "process_status": 'Running',
                "process_stage": 'Data Poisoning',
                "error": None,
                "stack trace": None,
                "report": None,
                             "pdf": None}

    store_on_db(project_id=PROJECT_ID,
                database=FIRESTORE_DB,
                collection_name=FIRESTORE_DATA_EVAL_STATUS_COLLECTION,
                document_key=job_id,
                params=data_eval_status)


@router.post("/dataset_evaluate/")
async def dataset_evaluate(request: DatasetEvaluationRequestBody, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())  # random uuid
    request_dict = request.dict()
    user_id = request_dict['user_id']
    job_id = user_id + "-" + job_id
    
    # Add by Eran Simtob for server use - start
    document = {"user_id": request.user_id,
                "job_id": job_id,
                "request": jsonable_encoder(request),
                "background_tasks": jsonable_encoder(background_tasks),
                "gpu_num": '',
                "request_type": 'dataset_evaluate'}
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
    background_tasks.add_task(perform_eval_wrapper, job_id, request)
    # Return the job ID to the client
    return {"job_id": job_id}