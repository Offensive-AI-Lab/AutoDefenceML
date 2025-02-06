
from fastapi import HTTPException, APIRouter
from fastapi.responses import JSONResponse
from .helpers import get_from_db
import os
from dotenv import load_dotenv

load_dotenv()
PROJECT_ID = os.getenv("PROJECT_ID")
FIRESTORE_DB = os.getenv("FIRESTORE_DB")
FIRESTORE_BIAS_MIT_STATUS_COLLECTION = os.getenv("FIRESTORE_BIAS_MIT_STATUS_COLLECTION")
router = APIRouter()

@router.get("/bias_mitigation_status/{job_id}")
async def bias_detection_status(job_id: str):
    job_id = job_id.strip("'")
    bias_detection_status = get_from_db(project_id=PROJECT_ID,
                                    database=FIRESTORE_DB,
                                    collection_name=FIRESTORE_BIAS_MIT_STATUS_COLLECTION,
                                    document_id=job_id)
    if bias_detection_status is None:
        raise HTTPException(status_code=404, detail="Job ID not found")
    return JSONResponse(content=bias_detection_status)