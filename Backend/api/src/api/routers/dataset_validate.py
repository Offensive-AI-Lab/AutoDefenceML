import os
from concurrent.futures import ThreadPoolExecutor
import func_timeout
from fastapi import BackgroundTasks, APIRouter
from fastapi.encoders import jsonable_encoder
from .helpers import store_on_db, get_from_db
from .request_classes import DatasetValidationRequestBody
from dotenv import load_dotenv

import uuid
from ...data_eval.user_files_data.helpers import *
import traceback
import asyncio
import json
import logging
from ...data_eval.data_validator import DataValidator, FileHandlerDataset


load_dotenv()
router = APIRouter()
PROJECT_ID = os.getenv("PROJECT_ID")
# TOPIC = os.getenv("EVAL_TOPIC")
# TOPIC_PING = os.getenv("TOPIC_PING")
# TOPIC_EVAL = os.getenv("TOPIC_EVAL")
FIRESTORE_DB = os.getenv("FIRESTORE_DB")
FIRESTORE_REPORTS_COLLECTION = os.getenv("FIRESTORE_REPORTS_COLLECTION")
FIRESTORE_DATA_VAL_STATUS_COLLECTION = os.getenv("FIRESTORE_DATA_VAL_STATUS_COLLECTION")


def perform_validation(job_id, request):
    request = request.dict()

    file_loader= FileHandlerDataset(request=request ,path_to_files_dir=get_files_package_root(),
                               path_to_dataset_files_dir=get_dataset_package_root(),
                                path_to_dataloader_files_dir=get_dataloader_package_root(),
                               from_bucket=os.getenv("FROM_BUCKET"),
                               bucket_name=os.getenv("BUCKET_NAME"),
                               account_service_key_name=os.getenv("ACCOUNT_SERVICE_KEY"))
    data_validator = DataValidator(file_loader)
    try:
        if data_validator.validate_dataloader():
            data_validate_status= get_from_db(project_id=PROJECT_ID,
                                              database=FIRESTORE_DB,
                                              collection_name=FIRESTORE_DATA_VAL_STATUS_COLLECTION,
                                              document_id=job_id)
            data_validate_status['process_stage'] = 'Dataset validation'
            store_on_db(project_id=PROJECT_ID,
                        database=FIRESTORE_DB,
                        collection_name=FIRESTORE_DATA_VAL_STATUS_COLLECTION,
                        document_key=job_id,
                        params=data_validate_status)
            if data_validator.validate_dataset():

                data_validate_status = {"job_id": job_id, "process_status": 'Done',
                                        "process_stage": 'None',
                                        "error": "",
                                        "stack trace": ""}
                store_on_db(project_id=PROJECT_ID,
                            database=FIRESTORE_DB,
                            collection_name=FIRESTORE_DATA_VAL_STATUS_COLLECTION,
                            document_key=job_id,
                            params=data_validate_status)

                logging.info(f"Saved status : {data_validate_status}")
            else:
                raise Exception("Validation  Dataset failed")
        else:
            raise Exception("Validation  Dataloader failed")

    except Exception as e:
        # Handle exceptions and record error details
        error_traceback = traceback.format_exc()
        data_validate_status = {"job_id": job_id, "process_status": "Failed",
                                     "process_stage": "None",
                                     "error": str(e),
                                     "stack trace": str(error_traceback)
                                     }

        store_on_db(project_id=PROJECT_ID,
                    database=FIRESTORE_DB,
                    collection_name=FIRESTORE_DATA_VAL_STATUS_COLLECTION,
                    document_key=job_id,
                    params=data_validate_status)
        # if validation_status["process_status"] == "Failed":
        #     # clean_env()


async def perform_validation_wrapper(job_id, request):
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        await loop.run_in_executor(executor, perform_validation, job_id, request)


def set_val_response(job_id):
    data_validate_status = {"job_id": job_id, "process_status": 'Running',
                "process_stage": 'Data validation',
                "error": None,
                "stack trace": None
                }

    store_on_db(project_id=PROJECT_ID,
                database=FIRESTORE_DB,
                collection_name=FIRESTORE_DATA_VAL_STATUS_COLLECTION,
                document_key=job_id,
                params=data_validate_status)
@router.post("/dataset_validate/")
async def bias_validate(request: DatasetValidationRequestBody, background_tasks: BackgroundTasks):
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
                "request_type": 'dataset_validate'}
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
    background_tasks.add_task(perform_validation_wrapper, job_id, request)
    # Return the job ID to the client
    return {"job_id": job_id}