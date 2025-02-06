from fastapi import HTTPException, APIRouter
from fastapi.encoders import jsonable_encoder
from .request_classes import ManualStopRequestBody
import os
from dotenv import load_dotenv
from google.cloud import pubsub_v1
from .helpers import get_from_db, store_on_db
import logging
import traceback
import time
load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")
TOPIC = os.getenv("EVAL_TOPIC")
TOPIC_PING = os.getenv("TOPIC_PING")
TOPIC_EVAL = os.getenv("TOPIC_EVAL")
FIRESTORE_DB = os.getenv("FIRESTORE_DB")
FIRESTORE_REPORTS_COLLECTION = os.getenv("FIRESTORE_REPORTS_COLLECTION")
FIRESTORE_EVAL_STATUS_COLLECTION =  os.getenv("FIRESTORE_EVAL_STATUS_COLLECTION")
router = APIRouter()


eval_topic_path = 'projects/{project_id}/topics/{topic}'.format(
    project_id=PROJECT_ID,
    topic=TOPIC_EVAL,
)

publisher = pubsub_v1.PublisherClient()
@router.post("/stop/")
async def stop(request: ManualStopRequestBody):
    request = request.model_dump()
    job_id = request["job_id"]

    # Add by Eran Simtob for server use - start
    document = {
                #"user_id": request.user_id,
                "job_id": job_id,
                "request": jsonable_encoder(request),
                "gpu_num": '',
                "request_type": 'stop'}
    store_on_db(project_id=PROJECT_ID,
                database=FIRESTORE_DB,
                collection_name="Requests",
                document_key=job_id,
                params=document)
    # Add by Eran Simtob for server use - end

    evaluation_status = get_from_db(project_id=PROJECT_ID,
                                    database=FIRESTORE_DB,
                                    collection_name=FIRESTORE_EVAL_STATUS_COLLECTION,
                                    document_id=job_id)
    if evaluation_status is None:
        raise HTTPException(status_code=404, detail="Job ID not found")

    try:
        publisher.publish(eval_topic_path, "stop".encode("utf-8"))
        report = get_from_db(project_id=PROJECT_ID,database=FIRESTORE_DB,
                   collection_name=FIRESTORE_REPORTS_COLLECTION,document_id=job_id)
        while report is None:
            time.sleep(3)
            publisher.publish(eval_topic_path, "stop".encode("utf-8"))
            report = get_from_db(project_id=PROJECT_ID, database=FIRESTORE_DB,
                                collection_name=FIRESTORE_REPORTS_COLLECTION, document_id=job_id)
        tries = 1
        while (not all([isinstance(r, str) for r in report.values()])) and tries < 5:
            publisher.publish(eval_topic_path, "stop".encode("utf-8"))
            time.sleep(3)
            report = get_from_db(project_id=PROJECT_ID, database=FIRESTORE_DB,
                                collection_name=FIRESTORE_REPORTS_COLLECTION, document_id=job_id)
            tries += 1
        if tries < 5 and report is not None:
            return {"stoppage_status": "successful"}
        return {"stoppage_status": "failed"}
    except Exception as err:
        logging.error(err)
        traceback.format_exc()
        raise HTTPException(status_code=404, detail="Job ID not found in DB")


