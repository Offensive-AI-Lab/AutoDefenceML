import uvicorn
import os
from fastapi import FastAPI
from fastapi import BackgroundTasks
from .routers import validatetest, validate, validatetion_statuse, evaluate as eval, evaluation_status, stop, bias_validate, bias_validate_status, bias_detection, bias_detection_status, bias_mitigation, bias_mitigation_status, dataset_validate_status, dataset_evaluate_status, dataset_validate, dataset_evaluate
from dotenv import load_dotenv
from ..eval.main import start_pubsub_listener
import threading
import asyncio
load_dotenv()

app = FastAPI(
    version="1.3.45"
)

app.include_router(stop.router)
app.include_router(eval.router)
app.include_router(validate.router)
app.include_router(validatetest.router)
app.include_router(validatetion_statuse.router)
app.include_router(evaluation_status.router)
app.include_router(bias_validate.router)
app.include_router(bias_validate_status.router)
app.include_router(bias_detection.router)
app.include_router(bias_detection_status.router)
app.include_router(bias_mitigation.router)
app.include_router(bias_mitigation_status.router)
app.include_router(dataset_validate.router)
app.include_router(dataset_validate_status.router)
app.include_router(dataset_evaluate.router)
app.include_router(dataset_evaluate_status.router)

environment_id = os.getenv("ENVIRONMENT") or "MainServer"
if environment_id == "cloud_dev":
    environment_id = "MainServer"

@app.get("/")
async def root():
    return {"message": "Validation"}
def start_pubsub_listener_in_thread():
    listener_thread = threading.Thread(target=start_pubsub_listener)
    listener_thread.start()

@app.on_event("startup")
async def startup_event():
    if environment_id != "MainServer":
        print("Starting PubSub Listener in a separate thread")
        start_pubsub_listener_in_thread()
        print("PubSub Listener thread started")
    # Run the initialize function in a background task
    asyncio.create_task(validatetest.initialize())
    print("=== Node Initialization Complete ===")



if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8080)
