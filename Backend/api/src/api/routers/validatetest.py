import os
import subprocess
import time
import uuid
import traceback
import asyncio
from datetime import datetime
import datetime as dt
import pytz
import json
import base64
import logging
import shutil
import kubernetes
from google.cloud import storage, container_v1, secretmanager
from google.cloud.exceptions import NotFound
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any
from fastapi import BackgroundTasks, APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
from .request_classes import ValidationRequestBody
from .helpers import clean_env, store_on_db, get_from_db, db_log, get_from_db_by_short_id, get_nodepools_from_db, check_nodepool_last_activity
from ...pre_run.input_validation import InputValidator
from ...pre_run.model_validation import ModelDatasetValidator
from ...pre_run.estimator_match import EstimatorHandler
from ...pre_run.attack_defense_validation import AttackDefenseValidator
from ...user_files.helpers import *
from file_loader.file_handler import FileLoader
from dotenv import load_dotenv
from .gke_utils import GKEUtils


load_dotenv()
router = APIRouter()

PROJECT_ID = os.getenv("PROJECT_ID")
TOPIC = os.getenv("EVAL_TOPIC")
TOPIC_PING = os.getenv("TOPIC_PING")
TOPIC_EVAL = os.getenv("TOPIC_EVAL")
FIRESTORE_DB = os.getenv("FIRESTORE_DB")
FIRESTORE_REPORTS_COLLECTION = os.getenv("FIRESTORE_REPORTS_COLLECTION")
FIRESTORE_VAL_STATUS_COLLECTION = os.getenv("FIRESTORE_VAL_STATUS_COLLECTION")

terraform_directory = "/Library/validation/src/Infrastructure/"
environment_id = os.getenv("ENVIRONMENT") or "MainServer"
if environment_id == "cloud_dev":
    environment_id = "MainServer"

def run_terraform_command_background(command: list, cwd: str) -> subprocess.Popen:
    return subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )


def run_terraform_init_background(terraform_directory: str) -> subprocess.Popen:
    return run_terraform_command_background(["terraform", "init"], terraform_directory)


def run_terraform_apply_background(terraform_directory: str) -> subprocess.Popen:
    return run_terraform_command_background(["terraform", "apply", "-auto-approve"], terraform_directory)


def run_terraform_destroy_background(terraform_directory: str, target: str) -> subprocess.Popen:
    return run_terraform_command_background(["terraform", "destroy", "-target=" + target, "-auto-approve"],
                                            terraform_directory)


def run_terraform_apply():
    try:
        result = subprocess.run(
            ["terraform", "apply", "-auto-approve"],
            cwd=terraform_directory,
            check=True,
            capture_output=True,
            text=True
        )
        db_log(environment_id, "66 - Terraform apply completed successfully.")
        db_log(environment_id, result.stdout)
        return result.stdout
    except subprocess.CalledProcessError as e:
        db_log(environment_id, f"70 - Terraform apply failed with error: {e}")
        db_log(environment_id, json.dumps({"error": str(e)}))
        db_log(environment_id, e.stderr)
        return None


def run_terraform_init():
    try:
        result = subprocess.run(
            ["terraform", "init"],
            cwd=terraform_directory,
            check=True,
            capture_output=True,
            text=True
        )
        db_log(environment_id, "66 - Terraform init completed successfully.")
        db_log(environment_id, result.stdout)
        return result.stdout
    except subprocess.CalledProcessError as e:
        db_log(environment_id, f"70 - Terraform init failed with error: {e}")
        db_log(environment_id, json.dumps({"error": str(e)}))
        db_log(environment_id, e.stderr)
        return None


def add_to_db(user_id, job_id, process_type, timestamp, cluster_name):
    # Add the job details to the MySQL database
    pass


def perform_validation_sync(job_id, request):
    try:
        clean_env()
        request = request.dict()
        user_id = request.get("user_id")
        db_log(environment_id, f"84 - job id: {job_id}")
        cluster_name = f"k8c-{job_id}"
        os.environ["FILES_PATH"] = os.getenv("FILES_PATH_VAL")

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

        input_validator = InputValidator(fileloader_obj=fileloader)
        model_dataset_validator = ModelDatasetValidator(fileloader_obj=fileloader)

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
            raise Exception("Model does not match with the dataset!")

        input = input_validator.get_input()
        db_log(environment_id, f"131 - Input to wrap estimator: {input}")
        wrapper = EstimatorHandler(input, fileloader_obj=fileloader)
        db_log(environment_id, f"133 - Starting to wrap model...")
        wrapper.wrap()
        db_log(environment_id, f"135 - Wrap went successfully...")
        estimator = fileloader.get_estimator()

        attack_defense_validator = AttackDefenseValidator(fileloader_obj=fileloader)
        compatible_attacks = attack_defense_validator.get_compatible_attacks(estimator=estimator)
        compatible_defenses = attack_defense_validator.get_compatible_defenses(estimator=estimator)

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

        db_log(environment_id, f"154 - Saved status : {validation_status}")

        with open(get_files_package_root() + "/Estimator_params.json", 'r') as f:
            estimator_params = json.load(f)
        store_on_db(project_id=os.getenv("PROJECT_ID"),
                    database=os.getenv("FIRESTORE_DB"),
                    collection_name=os.getenv("FIRESTORE_ESTIMATOR_COLLECTION"),
                    document_key=job_id,
                    params=estimator_params)
        clean_env()
    except Exception as e:
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
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        await loop.run_in_executor(executor, perform_validation_sync, job_id, request)


def set_val_response(job_id):
    validation_status = {"job_id": job_id, "process_status": 'Running',
                         "process_stage": 'Input validation',
                         "error": None,
                         "stack trace": None,
                         "compatible_attacks": None,
                         "compatible_defenses": None}

    store_on_db(project_id=PROJECT_ID,
                database=FIRESTORE_DB,
                collection_name=FIRESTORE_VAL_STATUS_COLLECTION,
                document_key=job_id,
                params=validation_status)


@router.post("/validate/")
async def validate(request: ValidationRequestBody, background_tasks: BackgroundTasks):
    try:
        await removelock()
        job_idUuid = str(uuid.uuid4())
        request_dict = request.dict()
        user_id = request_dict['user_id']
        job_id = user_id + "-" + job_idUuid
        short_id = job_idUuid[:6]

        document = {"user_id": request.user_id,
                    "job_id": job_id,
                    "request": jsonable_encoder(request),
                    "background_tasks": jsonable_encoder(background_tasks),
                    "short_id": short_id,
                    "destroy_id": "google_container_node_pool.node-gpu_num_" + short_id,
                    "request_type": 'validate'}

        db_log(environment_id, f"216 - Storing document in DB for job_id: {job_id}")
        store_on_db(project_id=PROJECT_ID,
                    database=FIRESTORE_DB,
                    collection_name="Requests",
                    document_key=job_id,
                    params=document)

        # הוספת תיעוד חדש ל-NodePools collection
        nodepool_doc = {
            "nodepool_name": f"sc-gpu-{short_id}",
            "short_id": short_id,
            "job_id": job_id,
            "user_id": request.user_id,
            "status": "active",
            "deleted_time": None,
            "create_time": dt.datetime.now(pytz.timezone('Asia/Jerusalem')).strftime("%Y-%m-%d %H:%M:%S")
        }
        
        store_on_db(project_id=PROJECT_ID,
                    database=FIRESTORE_DB,
                    collection_name="NodePools",
                    document_key=f"nodepool-{short_id}",
                    params=nodepool_doc)

        db_log(short_id, f"NodePool {short_id} is now starting")
        db_log(environment_id, f"NodePool {nodepool_doc['nodepool_name']} info stored in DB")

        validation_status2 = {"job_id": job_id, "process_status": 'Running',
                         "process_stage": 'NodePool is starting',
                         "error": None,
                         "stack trace": None,
                         "compatible_attacks": None,
                         "compatible_defenses": None}

        store_on_db(project_id=PROJECT_ID,
                    database=FIRESTORE_DB,
                    collection_name=FIRESTORE_VAL_STATUS_COLLECTION,
                    document_key=job_id,
                    params=validation_status2)

        db_log(environment_id, "223 - Fetching terraform file from DB")
        terraformFile = get_from_db(project_id=PROJECT_ID,
                                    database=FIRESTORE_DB,
                                    collection_name="Stuff",
                                    document_id="addGpuServer")

        def write_to_file(file_name, content):
            try:
                with open(file_name, 'w') as f:
                    f.write(content)
                db_log(environment_id, f"235 - Content written to {file_name}")
                db_log(environment_id, f"236 - Content: {content}")

                if os.path.exists(file_name):
                    file_size = os.path.getsize(file_name)
                    db_log(environment_id, f"237 - File {file_name} exists after writing. Size: {file_size} bytes")
                else:
                    db_log(environment_id, f"239 - File {file_name} does not exist after supposed writing")

            except Exception as e:
                db_log(environment_id, f"242 - Error writing to file {file_name}: {str(e)}")
                raise

        newText = terraformFile["json"].replace("XXXX", short_id)
        newText = newText.replace("YYYY", short_id)
        newText = newText.replace("MainServer", short_id)
        db_log(environment_id, f"248 - Modified terraform text: {newText[:100]}...")  # Log first 100 chars

        db_log(environment_id, f"250 - Terraform directory: {terraform_directory}")
        if not os.path.exists(terraform_directory):
            os.makedirs(terraform_directory)
            db_log(environment_id, f"253 - Created terraform directory: {terraform_directory}")

        file_path = os.path.join(terraform_directory, f"new-gpu-server-{short_id}.tf")
        db_log(environment_id, f"256 - Attempting to write file: {file_path}")

        write_to_file(file_path, newText)

        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            db_log(environment_id, f"262 - File {file_path} exists after write_to_file. Size: {file_size} bytes")
        else:
            db_log(environment_id, f"264 - File {file_path} does not exist after write_to_file")

        if terraformFile["runTerraform"]:
            db_log(environment_id, "267 - Adding run_terraform_apply to background tasks")
            background_tasks.add_task(run_terraform_apply)

        # set_val_response(job_id)
        # background_tasks.add_task(perform_validation, job_id, request)

        db_log(environment_id, f"273 - Validate function completed successfully for job_id: {job_id}")
        return {"job_id": job_id}

    except Exception as e:
        error_msg = f"277 - Error in validate function: {str(e)}"
        db_log(environment_id, error_msg)
        raise HTTPException(status_code=500, detail=error_msg)


@router.get("/getid/")
async def getid():
    db_log(environment_id, "283 - get id")
    return environment_id


@router.get("/removelock/")
async def removelock():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        # Initialize the Google Cloud Storage client
        storage_client = storage.Client()

        # Get the bucket
        bucket = storage_client.bucket("cbg-api-bucket")

        # Specify the blob (file) to delete
        blob = bucket.blob("Infrastructure/state/default.tflock")

        # Check if the blob exists
        if blob.exists():
            # Delete the blob
            blob.delete()
            logger.info("Lock file successfully removed")
            return {"message": "Lock file successfully removed"}
        else:
            logger.info("Lock file not found, no action needed")
            return {"message": "Lock file not found, no action needed"}

    except Exception as e:
        logger.error(f"Error checking/removing lock file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to check/remove lock file: {str(e)}")


@router.get("/listofgpu/")
async def list_gpu():
    try:
        result = subprocess.run(
            ["terraform", "state", "list"],
            cwd=terraform_directory,
            capture_output=True,
            text=True
        )
        return {"gpu_list": result.stdout.split('\n'), "error": result.stderr.split('\n')}
    except subprocess.CalledProcessError as e:
        db_log(environment_id, f"326 - Error during list GPU: {e.stderr}")
        return {"error": str(e)}


@router.get("/removegpu/{gpu_id}")
async def remove_gpu(gpu_id: str):
    destroy_process = run_terraform_destroy_background(terraform_directory, gpu_id)
    db_log(environment_id, f"332 - Terraform destroy for {gpu_id} started in background.")

    return {"message": f"Terraform destroy for {gpu_id} initiated in background"}


@router.get("/get_all_files/")
async def listget_all_files_gpu():
    try:
        # Check if the provided path is a directory
        if not os.path.isdir(terraform_directory):
            raise ValueError(f"{terraform_directory} is not a valid directory")

        # List all files in the directory
        files = []
        for entry in os.listdir(terraform_directory):
            full_path = os.path.join(terraform_directory, entry)
            if os.path.isfile(full_path):  # Only add files (ignore directories)
                files.append(full_path)

        return files

    except Exception as e:
        db_log(environment_id, f"353 - An error occurred: {e}")
        return []


@router.get("/terraform_apply/")
async def terraform_apply():
    removelock()
    return run_terraform_apply()


@router.get("/terraform_init/")
async def terraform_init():
    removelock()
    return run_terraform_init()


async def process_validation_by_job_id(job_id: str, background_tasks: BackgroundTasks):
    # שליפת הנתונים מה-DB
    document = get_from_db(
        project_id=PROJECT_ID,
        database=FIRESTORE_DB,
        collection_name="Requests",
        document_id=job_id
    )

    if not document:
        return {"error": f"373 - No document found for job_id: {job_id}"}

    # שחזור אובייקט הבקשה
    request = ValidationRequestBody(**document["request"])

    # הרצת הפעולות
    set_val_response(job_id)
    background_tasks.add_task(perform_validation, job_id, request)

    return {"message": f"382 - Validation process started for job_id: {job_id}"}


async def process_validation_by_id():
    db_log(environment_id, "387 - process_validation_by_id start")
    try:
        db_log(environment_id, "389 - log every line")
        document = await get_from_db_by_short_id(
            project_id=PROJECT_ID,
            database=FIRESTORE_DB,
            collection_name="Requests",
            short_id=environment_id
        )
        db_log(environment_id, "396 - log every line")
        if not document:
            db_log(environment_id, "394 - request: " + document["request"])
            return {"error": f"394 - No document found for short_id: " + document["job_id"]}
        db_log(environment_id, "400 - log every line")
        if "request" not in document:
            db_log(environment_id, "399 - 'request' key not found in document")
            return {"error": "400 - 'request' key not found in document"}
        db_log(environment_id, "404 - log every line")
        # שחזור אובייקט הבקשה
        request = ValidationRequestBody(**document["request"])
        db_log(environment_id, "407 - log every line")
        # הרצת הפעולות
        set_val_response(document["job_id"])
        db_log(environment_id, "410 - log every line")
        await perform_validation(document["job_id"], request)
        db_log(environment_id, "412 - Validation process started for job_id: " + document["job_id"])
        return {"message": f"Validation process started for job_id: " + document["job_id"]}
    except KeyError as e:
        db_log(environment_id, f"415 - Missing key in document: {e}")
        return {"error": f"416 - Missing key in document: {str(e)}"}
    except Exception as e:
        db_log(environment_id, f"418 - An error occurred: {e}")
        return {"error": f"419 - An error occurred: {str(e)}"}


# דוגמה לשימוש בפונקציה החדשה
@router.post("/process-validation/{job_id}")
async def api_process_validation(job_id: str, background_tasks: BackgroundTasks):
    return await process_validation_by_job_id(job_id, background_tasks)


# פונקציה שמחפשת job ב-DB לפי short_id
async def check_for_job():
    counter = 0  # מונה לריצות הלולאה
    while True:
        db_log(environment_id, f"{datetime.now()} - Checking for new jobs in DB")
        # שליפת המסמך מה-DB באמצעות await
        document = await get_from_db_by_short_id(
            project_id=PROJECT_ID,
            database=FIRESTORE_DB,
            collection_name="Requests",
            short_id=environment_id
        )
        db_log(environment_id, f"Job {json.dumps(document)} pulled.")
        if document:
            # בדיקה אם המסמך נמשך בעבר (pulled == True)
            if document.get('pulled'):
                db_log(environment_id, f"Job {document['job_id']} already pulled. Stopping timer.")
                break  # מפסיק את הלולאה אם המסמך כבר נמשך
            db_log(environment_id, f"Job found with job_id: {document['job_id']}")
            await process_validation_by_id()
            document['pulled'] = True  # מעדכן שהג'וב נמשך
            await update_job_in_db(document['job_id'], document)  # פונקציית עזר לעדכון ה-DB (async)
            db_log(environment_id, f"Job {document['job_id']} marked as pulled")
            break  # מפסיק את הלולאה לאחר הטיפול ב-Job
        # אם לא מצא ג'וב, מחכה דקה ובודק שוב
        await asyncio.sleep(60)
        counter += 1
        if counter >= 30:
            db_log(environment_id, f"Job check loop ran 30 times. Stopping.")
            break



# פונקציית עזר לעדכון ה-DB (כעת היא async)
async def update_job_in_db(job_id, document):
    db_log(environment_id, f"Updating document in DB for job_id: {job_id}")

    # שימוש ב-await לעדכון המסמך ב-DB
    store_on_db(
        project_id=PROJECT_ID,
        database=FIRESTORE_DB,
        collection_name="Requests",
        document_key=job_id,
        params=document
    )


# פונקציה ראשית שמתחילה את התהליך
async def start_job_checking():
    await check_for_job()

async def cleanup_inactive_nodepools():
    """
    Check and cleanup inactive nodepools
    """
    try:
        # Get list of active nodepools
        active_nodepools = await get_nodepools_from_db(PROJECT_ID, FIRESTORE_DB, status="active")
        
        for nodepool in active_nodepools:
            try:
                nodepool_id = nodepool.get('short_id')
                if not nodepool_id:
                    continue
                    
                db_log(environment_id, f"Checking activity for nodepool {nodepool_id}")
                
                # Check if nodepool is inactive
                is_inactive = await check_nodepool_last_activity(nodepool_id)
                
                if is_inactive:
                    db_log(environment_id, f"Nodepool {nodepool_id} is inactive. Initiating deletion...")
                    
                    # Delete the nodepool and deployment
                    await perform_nodepool_deployment_deletion(nodepool_id)
                    
                    # Update nodepool status using existing store_on_db function
                    nodepool_update = {
                        "status": "deleted",
                        "deleted_time": dt.datetime.now(pytz.timezone('Asia/Jerusalem')).strftime("%Y-%m-%d %H:%M:%S"),
                        "deletion_reason": "automatic cleanup - inactive for 6+ hours"
                    }
                    
                    store_on_db(
                        project_id=PROJECT_ID,
                        database=FIRESTORE_DB,
                        collection_name="NodePools",
                        document_key=nodepool['document_id'],
                        params=nodepool_update
                    )
                    
                    db_log(environment_id, f"Successfully deleted inactive nodepool {nodepool_id}")
                else:
                    db_log(environment_id, f"Nodepool {nodepool_id} is still active")
                    
            except Exception as e:
                db_log(environment_id, f"Error processing nodepool {nodepool_id}: {str(e)}")
                continue
                
    except Exception as e:
        db_log(environment_id, f"Error in cleanup_inactive_nodepools: {str(e)}")

async def start_cleanup_timer():
    """
    Start the periodic cleanup timer
    """
    while True:
        try:
            await cleanup_inactive_nodepools()
        except Exception as e:
            db_log(environment_id, f"Error in cleanup timer: {str(e)}")
            
        # Wait 30 minutes before next check
        await asyncio.sleep(1800)  # 30 minutes in seconds


async def initialize():
    await get_service_account_from_secret()
    init_process = run_terraform_init_background(terraform_directory)
    db_log(environment_id, "417 - Terraform init started in background.")
    if environment_id != "MainServer":
        db_log(environment_id, "419 - In the If (environment_id != MainServer)")
        await start_job_checking()
    else:
        db_log(environment_id, "Running cleanup timer in MainServer")
        await start_cleanup_timer()


@router.post("/post_validation_by_id/")
async def post_validation_by_id():
    return await process_validation_by_id()


def delete_terraform_file_core(short_id: str) -> bool:
    file_path = os.path.join(terraform_directory, f"new-gpu-server-{short_id}.tf")
    db_log(environment_id, f"Attempting to delete file: {file_path}")

    if os.path.exists(file_path):
        os.remove(file_path)
        db_log(environment_id, f"File {file_path} deleted successfully")

        if not os.path.exists(file_path):
            db_log(environment_id, f"File {file_path} confirmed deleted")
            return True
        else:
            db_log(environment_id, f"File {file_path} still exists after attempted deletion")
            return False
    else:
        db_log(environment_id, f"File {file_path} does not exist, nothing to delete")
        return True

@router.delete("/delete_terraform_file/{short_id}")
async def delete_terraform_file(short_id: str):
    try:
        success = delete_terraform_file_core(short_id)
        if success:
            return {"message": f"File for {short_id} deleted successfully if it existed"}
        else:
            raise HTTPException(status_code=500, detail="File deletion failed")

    except Exception as e:
        db_log(environment_id, f"Error deleting terraform file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")

@router.post("/run_terraform_plan/")
async def run_terraform_plan():
    try:
        result = subprocess.run(
            ["terraform", "plan"],
            cwd=terraform_directory,
            check=True,
            capture_output=True,
            text=True
        )
        db_log(environment_id, "430 - Terraform plan completed successfully.")
        db_log(environment_id, result.stdout)
        return result.stdout
    except subprocess.CalledProcessError as e:
        db_log(environment_id, f"434 - Terraform plan failed !!!!!!!! with error: {e}")
        db_log(environment_id, json.dumps({"error": str(e)}))
        db_log(environment_id, e.stderr)
        return None

# def check_command_exists(command):
#     return shutil.which(command) is not None

# def run_command(command):
#     try:
#         result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
#         return result.stdout.strip()
#     except subprocess.CalledProcessError as e:
#         return f"Error: {e.output.strip()}"

# def check_gcloud_config():
#     try:
#         project = run_command("gcloud config get-value project")
#         account = run_command("gcloud config get-value account")
#         return f"Project: {project}, Account: {account}"
#     except Exception as e:
#         return f"Error checking gcloud config: {str(e)}"
    
# def check_kubectl_config():
#     try:
#         context = run_command("kubectl config current-context")
#         clusters = run_command("kubectl config get-clusters")
#         users = run_command("kubectl config get-users")
#         return f"Current context: {context}\nClusters: {clusters}\nUsers: {users}"
#     except Exception as e:
#         return f"Error checking kubectl config: {str(e)}\n{run_command('kubectl config view')}"

# def run_the_command(command: str) -> dict:
#     """
#     Run a command and return detailed output.
#     """
#     try:
#         # Log the command being executed
#         db_log(environment_id, f"Executing command: {command}")
        
#         # Run the command and capture both stdout and stderr
#         result = subprocess.run(
#             command,
#             shell=True,
#             capture_output=True,
#             text=True
#         )
        
#         # Log the complete output
#         db_log(environment_id, f"Command stdout: {result.stdout}")
#         db_log(environment_id, f"Command stderr: {result.stderr}")
#         db_log(environment_id, f"Return code: {result.returncode}")
        
#         return {
#             "success": result.returncode == 0,
#             "stdout": result.stdout.strip(),
#             "stderr": result.stderr.strip(),
#             "return_code": result.returncode
#         }
        
#     except Exception as e:
#         error_msg = f"Error running command: {str(e)}"
#         db_log(environment_id, error_msg)
#         return {
#             "success": False,
#             "error": str(e),
#             "stdout": "",
#             "stderr": error_msg,
#             "return_code": -1
#         }

# @router.post("/run_command")
# async def run_command(command: str):
#     output = run_the_command(command)
#     db_log(environment_id, f"Command execution result: {output}")
#     return {"message": "Command executed", "output": output}

@router.post("/addGpuToDB/")
async def addGpuToDB():
    document2 = {"runTerraform": True,
                 "json": '''variable "gpu_num_XXXX" {
      description = "The ID for the duplicate gpu node"
      type        = string
      default     = "YYYY"
    }

    resource "kubernetes_deployment" "api_terraform_gpu_num_XXXX" {
      metadata {
        name = "app-deployment-gpu-${var.gpu_num_XXXX}"
      }

      spec {
        replicas = 1

        selector {
          match_labels = {
            app = "app-gpu-${var.gpu_num_XXXX}"
          }
        }

        template {
          metadata {
            labels = {
              app = "app-gpu-${var.gpu_num_XXXX}"
            }
          }

          spec {
            service_account_name = "k8s-service"

            node_selector = {
              "nodepool" = "sc-gpu-${var.gpu_num_XXXX}",
              "gpu"      = "true${var.gpu_num_XXXX}"
            }

            toleration {
              key      = "nvidia.com/gpu"
              operator = "Equal"
              value    = "present"
              effect   = "NoSchedule"
            }

            container {
              name  = "app-gpu-${var.gpu_num_XXXX}"
              image = "us-central1-docker.pkg.dev/autodefenseml/autodefenseml/app:latest"
              port {
                container_port = 8080
              }

              env {
                name  = "api_terraform_PORT"
                value = "8080"
              }

              env {
                name  = "api_terraform_HOST"
                value = "0.0.0.0"
              }

              env {
                name  = "USER_MNG_URL"
                value = "http://user-manager:80"
              }

              env {
                name  = "SYSTEM_SERVICE_URL"
                value = "http://system-service:80"
              }

              env {
                name  = "GOOGLE_CLOUD_PROJECT"
                value = "autodefenseml"
              }

              env {
                name  = "PROJECT_MNG_URL"
                value = "http://project-manager:80"
              }

              env {
                name  = "CONFIG_MNG_URL"
                value = "http://configuration-manager:80"
              }

              env {
                name  = "VALIDATION_MNG_URL"
                value = "http://validation-manager:80"
              }

              env {
                name  = "EVAL_DETECTION_MNG_URL"
                value = "http://eval-detection-manager:80"
              }

              env {
                name  = "FIREBASE_API_KEY"
                value = "AIzaSyDEOZpCAcrjpJI5MaKB4wECdkjw8V5_bxU"
              }

              env {
                name  = "ENVIRONMENT"
                value = "MainServer"
              }
            }
          }
        }
      }
    }

    resource "google_container_node_pool" "node-gpu_num_XXXX" {
      name               = "sc-gpu-${var.gpu_num_XXXX}"
      cluster            = "projects/autodefenseml/locations/us-central1/clusters/autodefenseml"
      initial_node_count = 1 # Ensure the pool starts with at least 1 node


      management {
        auto_repair  = true
        auto_upgrade = true
      }

      autoscaling {
        min_node_count = 1
        max_node_count = 5
      }

      node_config {
        preemptible  = false
        machine_type = "n1-standard-16" # Adjust the machine type as needed

        guest_accelerator {
          count = 1                 # Number of GPUs per node
          type  = "nvidia-tesla-t4" # Replace with the appropriate GPU type
        }

        # Optional taint configuration
        taint {
          key    = "nvidia.com/gpu"
          value  = "present"
          effect = "NO_SCHEDULE"
        }

        service_account = "kubernetes-test@autodefenseml.iam.gserviceaccount.com"
        oauth_scopes = [
          "https://www.googleapis.com/auth/cloud-platform"
        ]
        tags = ["gke-node"]

        labels = {
          "role"     = "compute"
          "nodepool" = "sc-gpu-${var.gpu_num_XXXX}"
          "gpu"      = "true${var.gpu_num_XXXX}"
        }

        # Add a startup script to install GPU drivers
        metadata = {
          "install-gpu-driver"       = "true"
          "disable-legacy-endpoints" = "true"
        }
      }
    }

    resource "kubernetes_service" "api_terraform_gpu_num_XXXX" {
      metadata {
        name = "api-terraform-service-gpu${var.gpu_num_XXXX}"
        annotations = {
          "cloud.google.com/neg" = jsonencode({
            ingress = true
          })
        }
      }

      spec {
        selector = {
          app = "app-gpu-${var.gpu_num_XXXX}"
        }

        port {
          port        = 80
          target_port = 8080
        }

        type = "LoadBalancer"
      }
    }
    '''}
    return store_on_db(project_id=PROJECT_ID,
                       database=FIRESTORE_DB,
                       collection_name="Stuff",
                       document_key="addGpuServer",
                       params=document2)


async def delete_node_pool_directly(short_id: str):
    """
    Delete the node pool using Container Client - for direct code calls.
    """
    try:
        db_log(environment_id, f"Starting node pool deletion with client for ID: {short_id}")
        
        # הגדרת משתנים
        project_id = "autodefenseml"
        zone = "us-central1"
        cluster_id = "autodefenseml"
        node_pool_id = f"sc-gpu-{short_id}"
        
        # יצירת הקליינט
        db_log(environment_id, "Creating cluster manager client")
        client = container_v1.ClusterManagerClient()
        
        # בניית שם המשאב
        name = f"projects/{project_id}/locations/{zone}/clusters/{cluster_id}/nodePools/{node_pool_id}"
        db_log(environment_id, f"Built resource name: {name}")
        
        # מחיקת ה-node pool
        db_log(environment_id, f"Initiating node pool deletion for: {node_pool_id}")
        operation = client.delete_node_pool(name=name)
        
        db_log(environment_id, "Node pool deletion initiated successfully")
        
        return {
            "status": "completed",
            "message": "Node pool deletion initiated successfully",
            "operation_name": operation.name
        }
        
    except Exception as e:
        error_message = f"Error initiating node pool deletion: {str(e)}"
        db_log(environment_id, error_message)
        return {
            "status": "error",
            "message": "Node pool deletion failed",
            "error": str(e)
        }

async def delete_deployment_directly(short_id: str):
    """
    Delete the deployment using Kubernetes Client - for direct code calls.
    """
    try:
        db_log(environment_id, f"Starting deployment deletion with client for ID: {short_id}")
        
        deployment_name = f"app-deployment-gpu-{short_id}"
        
        # טעינת קונפיגורציית Kubernetes מתוך הקלאסטר
        db_log(environment_id, "Loading in-cluster kubernetes config")
        kubernetes.config.load_incluster_config()
        
        # יצירת Kubernetes API client
        db_log(environment_id, "Creating kubernetes client")
        v1 = kubernetes.client.AppsV1Api()
        
        # מחיקת ה-deployment
        db_log(environment_id, f"Initiating deployment deletion for: {deployment_name}")
        delete_options = kubernetes.client.V1DeleteOptions(
            propagation_policy='Foreground',
            grace_period_seconds=0
        )
        
        api_response = v1.delete_namespaced_deployment(
            name=deployment_name,
            namespace="default",
            body=delete_options
        )
        
        db_log(environment_id, f"Deployment deletion completed successfully")
        
        return {
            "status": "completed",
            "message": "Deployment deletion completed successfully",
            "details": str(api_response.status)
        }
        
    except kubernetes.client.rest.ApiException as api_error:
        if api_error.status == 404:
            return {
                "status": "completed",
                "message": f"Deployment {deployment_name} not found (already deleted)",
                "details": str(api_error)
            }
        else:
            error_message = f"Kubernetes API Error: {str(api_error)}"
            db_log(environment_id, error_message)
            return {
                "status": "error",
                "message": "Failed to delete deployment",
                "error": error_message
            }
            
    except Exception as e:
        error_message = f"Error in deployment deletion: {str(e)}"
        db_log(environment_id, error_message)
        return {
            "status": "error",
            "message": "Failed to delete deployment",
            "error": str(e)
        }

@router.delete("/delete_node_pool_client/{short_id}")
async def delete_node_pool_client(short_id: str):
    """
    Delete the node pool using Container Client - API endpoint version.
    """
    return await delete_node_pool_directly(short_id)

@router.delete("/delete_deployment_client/{short_id}")
async def delete_deployment_client(short_id: str):
    """
    Delete the deployment using Kubernetes Client - API endpoint version.
    """
    return await delete_deployment_directly(short_id)

async def perform_nodepool_deployment_deletion(short_id: str):
    """
    Internal function to perform the actual deletion process
    """
    try:
        db_log(environment_id, f"Starting combined deletion process for ID: {short_id}")
        
        # First delete the deployment
        deployment_result = await delete_deployment_directly(short_id)
        db_log(environment_id, f"Deployment deletion result: {deployment_result}")
        
        # Then delete the node pool
        nodepool_result = await delete_node_pool_directly(short_id)
        db_log(environment_id, f"Node pool deletion result: {nodepool_result}")

        # delete terraform file if exists
        delete_terraform_file_core(short_id)

        # After successful deletion, update the nodepool status in DB
        nodepool_update = {
            "status": "deleted",
            "deleted_time": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        store_on_db(project_id=PROJECT_ID,
                    database=FIRESTORE_DB,
                    collection_name="NodePools",
                    document_key=f"nodepool-{short_id}",
                    params=nodepool_update)
        
        db_log(environment_id, f"Updated NodePool sc-gpu-{short_id} status to deleted in DB")
        
        return {
            "status": "completed",
            "deployment_result": deployment_result,
            "nodepool_result": nodepool_result
        }
        
    except Exception as e:
        error_message = f"Error in deletion process: {str(e)}"
        db_log(environment_id, error_message)
        raise Exception(error_message)

@router.delete("/delete_nodepool_and_deployment/{short_id}")
async def delete_nodepool_and_deployment(short_id: str):
    """
    API endpoint to delete both the node pool and deployment
    """
    try:
        result = await perform_nodepool_deployment_deletion(short_id)
        return result
        
    except Exception as e:
        error_message = f"Error in combined deletion process: {str(e)}"
        db_log(environment_id, error_message)
        raise HTTPException(status_code=500, detail=error_message)
    

async def get_nodepools_list(status: str = None):
    try:
        nodepools = await get_nodepools_from_db(
            project_id=PROJECT_ID,
            database=FIRESTORE_DB,
            status=status
        )
        return nodepools
    except Exception as e:
        error_message = f"Error fetching nodepools from DB: {str(e)}"
        db_log(environment_id, error_message)
        raise Exception(error_message)

# The API endpoint
@router.get("/list_db_nodepools")
async def list_db_nodepools(status: str = None):
    try:
        nodepools = await get_nodepools_list(status)
        return {
            "nodepools": nodepools,
            "count": len(nodepools)
        }
    except Exception as e:
        error_message = f"Error in list_db_nodepools: {str(e)}"
        db_log(environment_id, error_message)
        raise HTTPException(status_code=500, detail=error_message)

@router.delete("/delete_all_nodepools")
async def delete_all_nodepools():
    try:
        nodepools = await get_nodepools_list(status="active")
        
        deletion_results = []
        errors = []
        
        for nodepool in nodepools:
            try:
                short_id = nodepool.get('short_id')
                await perform_nodepool_deployment_deletion(short_id)
                
                deletion_results.append({
                    'nodepool_name': nodepool.get('nodepool_name'),
                    'short_id': short_id,
                    'status': 'deleted'
                })
                
                db_log(environment_id, f"Successfully deleted nodepool: {nodepool.get('nodepool_name')}")
            
            except Exception as e:
                error_msg = f"Failed to delete nodepool {nodepool.get('nodepool_name')}: {str(e)}"
                errors.append(error_msg)
                db_log(environment_id, error_msg)
        
        return {
            "message": f"Deletion process completed. Deleted {len(deletion_results)} nodepools",
            "deleted_nodepools": deletion_results,
            "errors": errors
        }
        
    except Exception as e:
        error_message = f"Error in delete_all_nodepools: {str(e)}"
        db_log(environment_id, error_message)
        raise HTTPException(status_code=500, detail=error_message)
    



async def get_service_account_from_secret():
    """
    Get service account key from Secret Manager and set it as environment variable
    Returns JSON string of service account credentials
    """
    try:
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{PROJECT_ID}/secrets/autodefenseml-service-key/versions/latest"
        response = client.access_secret_version(request={"name": name})
        secret_creds = response.payload.data.decode("UTF-8")
        
        # Verify it's valid JSON
        secret_json = json.loads(secret_creds)
        
        # Set environment variable
        os.environ["ACCOUNT_SERVICE_KEY"] = json.dumps(secret_json)
        db_log(environment_id, "Successfully set ACCOUNT_SERVICE_KEY in environment")
        
        return secret_creds
    except Exception as e:
        db_log(environment_id, f"Error getting service account from secret manager: {str(e)}")
        return None


@router.get("/test_service_account/")
async def test_service_account():
    try:
        # נסה את Secret Manager
        secret_creds = await get_service_account_from_secret()
        if secret_creds:
            db_log(environment_id, "Successfully retrieved credentials from Secret Manager")
            # הדפס את התוכן של הסוד
            db_log(environment_id, f"Secret content: {secret_creds[:50]}...") # מציג רק את 50 התווים הראשונים
            
            try:
                # הדפס מידע בסיסי על המפתח (ללא ערכים רגישים)
                safe_info = {
                    "type": secret_creds.get("type"),
                    "project_id": secret_creds.get("project_id"),
                    "client_email": secret_creds.get("client_email")
                }
                db_log(environment_id, f"Parsed credentials info: {safe_info}")
            except json.JSONDecodeError:
                db_log(environment_id, "Failed to parse credentials as JSON")
        
        return {
            "secret_manager_success": bool(secret_creds),
            "secret_preview": secret_creds[:100] if secret_creds else None  # מציג תצוגה מקדימה של הסוד
        }
        
    except Exception as e:
        error_message = f"Error testing service account: {str(e)}"
        db_log(environment_id, error_message)
        raise HTTPException(status_code=500, detail=error_message)