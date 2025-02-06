import base64
import json
import logging
import os
import traceback
from fastapi import HTTPException, APIRouter
from fastapi.responses import JSONResponse
from .helpers import get_from_db, store_on_db
from dotenv import load_dotenv
import  time
from ...eval.evaluation.pdf_generator.report import *
from datetime import datetime

load_dotenv()
PROJECT_ID = os.getenv("PROJECT_ID")
FIRESTORE_DB = os.getenv("FIRESTORE_DB")
FIRESTORE_REPORTS_COLLECTION = os.getenv("FIRESTORE_REPORTS_COLLECTION")
FIRESTORE_EVAL_STATUS_COLLECTION =  os.getenv("FIRESTORE_EVAL_STATUS_COLLECTION")
router = APIRouter()

def calc_elapsed(start_time_str, end_time_str):
    # Parse start and end times
    start_time = datetime.strptime(start_time_str, "%d/%m/%Y %H:%M:%S")
    end_time = datetime.strptime(end_time_str, "%d/%m/%Y %H:%M:%S")

    # Calculate elapsed time
    elapsed_time = end_time - start_time
    seconds = elapsed_time.total_seconds()

    # Format elapsed time
    if seconds < 60:
        elapsed_str = f"{seconds:.0f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        elapsed_str = f"{minutes:.0f} minutes"
    elif seconds < 86400:
        hours = seconds / 3600
        elapsed_str = f"{hours:.0f} hours"
    else:
        days = seconds / 86400
        elapsed_str = f"{days:.0f} days"
    return elapsed_str
@router.get("/evaluation_status/{job_id}")
async def get_evaluation_status(job_id: str):
    evaluation_status = get_from_db(project_id=PROJECT_ID,
                                    database=FIRESTORE_DB,
                                    collection_name=FIRESTORE_EVAL_STATUS_COLLECTION,
                                    document_id=job_id)
    # check if job_id is recorded
    if evaluation_status is None:
        raise HTTPException(status_code=404, detail="Job ID not found")
    # check if job's status is failed
    elif not evaluation_status['process_status'] == "Failed":
        try:
            report = get_from_db(project_id=PROJECT_ID, database=FIRESTORE_DB,
                                 collection_name=FIRESTORE_REPORTS_COLLECTION, document_id=job_id)
        except Exception as err:
            logging.error(err)
            traceback.format_exc()
            raise HTTPException(status_code=404, detail="Job ID not found in DB")
        # check if there is a report and if all tasks has finished
        if report is not None \
                and len(report.keys()) == evaluation_status["num_of_defenses"] + 1:
            # check for total fail, meaning all tasks are failed
            if all([isinstance(r, str) for k, r in report.items() if k != "clean_model_evaluation"]):
                evaluation_status["process_stage"] = None
                evaluation_status["process_status"] = "Failed"
                evaluation_status["error"] = json.dumps(report)
                evaluation_status["stack trace"] = json.dumps(report)

            # check if some tasks failed
            elif any([isinstance(r, str) for r in report.values()]):
                evaluation_status["process_stage"] = None
                evaluation_status["process_status"] = "Done with failures"

            # else means all tasks has been successful
            else:
                evaluation_status["process_status"] = "Done"

            evaluation_status["process_stage"] = None
            evaluation_status["report"] = report
            # if evaluation_status["process_status"] == "Done":
            #     pdf = generate_pdf(report)
            #     evaluation_status["pdf"] = pdf


        else:
            evaluation_status["process_status"] = "Running"
            evaluation_status["report"] = report
            evaluation_status["pdf"] = None
    current_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    start_time_str = evaluation_status.get("start_time")
    if start_time_str:
        elapsed_time_formatted = calc_elapsed(start_time_str, current_time)
        evaluation_status["elapsed_time"] = elapsed_time_formatted
    if evaluation_status["process_status"] == "Done":
        try:
            report = evaluation_status["report"]
            print("generate pdf...")
            pdf = generate_pdf(report, job_id)
          # Convert to base64 string
            evaluation_status["pdf"] = pdf

        except Exception as err:
            #print traceback
            print(traceback.format_exc())
            print(err)

            evaluation_status["pdf"] = traceback.format_exc() #change to null, this is for debugging
    store_on_db(project_id=PROJECT_ID,
                database=FIRESTORE_DB,
                collection_name=FIRESTORE_EVAL_STATUS_COLLECTION,
                document_key=job_id,
                params=evaluation_status)
    if "start_time" in evaluation_status:
        del evaluation_status["start_time"]

    return JSONResponse(content=evaluation_status)

