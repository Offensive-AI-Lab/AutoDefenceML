import os

import func_timeout
from fastapi import BackgroundTasks, APIRouter
from fastapi.encoders import jsonable_encoder
from .helpers import store_on_db, get_from_db
from .request_classes import BiasMitigationRequestBody
from dotenv import load_dotenv

import uuid
from ...bias_eval.user_files_bias.helpers import *
import traceback
import asyncio
import json
import logging
from ...bias_eval.bias_mitigation import mitigation
from ...bias_eval.bias_validation import FileHandlerDataset, BiasValidator
from concurrent.futures import ThreadPoolExecutor
load_dotenv()
router = APIRouter()
PROJECT_ID = os.getenv("PROJECT_ID")
# TOPIC = os.getenv("EVAL_TOPIC")
# TOPIC_PING = os.getenv("TOPIC_PING")
# TOPIC_EVAL = os.getenv("TOPIC_EVAL")
FIRESTORE_DB = os.getenv("FIRESTORE_DB")
FIRESTORE_REPORTS_COLLECTION = os.getenv("FIRESTORE_REPORTS_COLLECTION")
FIRESTORE_BIAS_MIT_STATUS_COLLECTION = os.getenv("FIRESTORE_BIAS_MIT_STATUS_COLLECTION")

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

def perform_mitigation(job_id, request):
    request = request.dict()
    priv_features = request["priv_features"]
    mitigations = request["mitigations"] # list of dicts of name description
    download_url = request.get("download_url", None)
    print(download_url)
    file_loader = FileHandlerDataset(request=request, path_to_files_dir=get_files_package_root(),
                                     path_to_dataset_files_dir=get_dataset_package_root(),
                                     path_to_dataloader_files_dir=get_dataloader_package_root(),
                                     from_bucket=os.getenv("FROM_BUCKET"),
                                     bucket_name=os.getenv("BUCKET_NAME"),
                                     account_service_key_name=os.getenv("ACCOUNT_SERVICE_KEY"))
    bias_validator = BiasValidator(file_loader)
    validate = bias_validator.validate_dataset()
    dataframe = bias_validator.get_dataframe()
    target_name = bias_validator.get_target_name()
    try:
        bias_mitigation_status = {"job_id": job_id, "process_status": 'Running',
                                  "process_stage": 'Bias Mitigation',
                                  "error": None,
                                  "stack trace": None,
                                  "report": {},
                                  "pdf": None}
        # for mitigate in mitigations:
        #     mitigation_name = mitigate["name"]
        import base64
        from PyPDF2 import PdfReader, PdfWriter
        from io import BytesIO
        pdf_buffers = {}
        for priv_feature in priv_features:

            if mitigations is not None:
                feat_name = priv_feature["name"]
                feat_value = priv_feature["value"]
                feature = f'{feat_name}_{feat_value}'
                report_dict , pdf_base64 = mitigation(mitigations, dataframe, priv_feature , target_name, download_url)
                bias_mitigation_status['report'][feature] = report_dict
                pdf_buffers[feature] = base64.b64decode(pdf_base64)


            else:
                raise Exception("Mitigation not found")
        # Merge all PDFs into one
        pdf_writer = PdfWriter()
        for priv_name, pdf_buffer in pdf_buffers.items():
            pdf_reader = PdfReader(BytesIO(pdf_buffer))
            for page_num in range(len(pdf_reader.pages)):
                pdf_writer.add_page(pdf_reader.pages[page_num])

        # Write the combined PDF to a BytesIO object
        combined_pdf_buffer = BytesIO()
        pdf_writer.write(combined_pdf_buffer)
        combined_pdf_buffer.seek(0)

        # Encode the combined PDF to a base64 string
        combined_pdf_base64 = base64.b64encode(combined_pdf_buffer.read()).decode('utf-8')

        bias_mitigation_status["pdf"] = combined_pdf_base64
        bias_mitigation_status["process_status"] = 'Completed'
        store_on_db(project_id=PROJECT_ID,
                    database=FIRESTORE_DB,
                    collection_name=FIRESTORE_BIAS_MIT_STATUS_COLLECTION,
                    document_key=job_id,
                    params=bias_mitigation_status)
        clean_env()
    except Exception as e:
        bias_mitigation_status = {"job_id": job_id, "process_status": 'Failed',
                                  "process_stage": 'Bias Mitigation',
                                  "error": str(e),
                                  "stack trace": traceback.format_exc(),
                                  "report": None,
                                  "pdf": None}
        store_on_db(project_id=PROJECT_ID,
                    database=FIRESTORE_DB,
                    collection_name=FIRESTORE_BIAS_MIT_STATUS_COLLECTION,
                    document_key=job_id,
                    params=bias_mitigation_status)
        clean_env()





async def perform_mitigation_wrapper(job_id, request):
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        await loop.run_in_executor(executor, perform_mitigation, job_id, request)

def set_val_response(job_id):
    bias_mitigation_status = {"job_id": job_id, "process_status": 'Running',
                "process_stage": 'Bias Mitigation',
                "error": None,
                "stack trace": None,
                "report": {},
                             "pdf": None}

    store_on_db(project_id=PROJECT_ID,
                database=FIRESTORE_DB,
                collection_name=FIRESTORE_BIAS_MIT_STATUS_COLLECTION,
                document_key=job_id,
                params=bias_mitigation_status)


@router.post("/bias_mitigation/")
async def bias_mitigation(request: BiasMitigationRequestBody, background_tasks: BackgroundTasks):
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
                "request_type": 'bias_mitigation'}
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
    background_tasks.add_task(perform_mitigation_wrapper, job_id, request)
    # Return the job ID to the client
    return {"job_id": job_id}