import os
import traceback
from google.cloud import pubsub_v1
from google.cloud import firestore
import pytz
from datetime import datetime, timedelta
import time
import uuid
from dotenv import load_dotenv
from ...user_files.helpers import get_files_package_root, get_model_package_root, get_dataset_package_root, \
    get_dataloader_package_root, get_loss_package_root, get_req_files_package_root

load_dotenv()
PROJECT_ID = os.getenv("PROJECT_ID")
TOPIC_EVAL = os.getenv("TOPIC_EVAL")
TOPIC_PING = os.getenv("TOPIC_PING")
FIRESTORE_DB = os.getenv("FIRESTORE_DB")
FIRESTORE_ESTIMATOR_COLLECTION = os.getenv("FIRESTORE_ESTIMATOR_COLLECTION")
FIRESTORE_REPORTS_COLLECTION = os.getenv("FIRESTORE_REPORTS_COLLECTION")
environment_id = os.getenv("ENVIRONMENT") or "MainServer"
if environment_id == "cloud_dev":
    environment_id = "MainServer"


def get_subscriber(topic):
    """
    Create a subscription to a topic
    """
    project_id = PROJECT_ID
    topic_id = topic
    uuid_str = str(uuid.uuid4())

    # Create a Pub/Sub client
    publisher = pubsub_v1.PublisherClient()
    subscriber = pubsub_v1.SubscriberClient()

    # Create paths
    topic_path = publisher.topic_path(project_id, topic_id)
    subscription_id = '{topic_id}-subscription-{uuid_str}'.format(topic_id=topic_id, uuid_str=uuid_str)
    subscription_path = subscriber.subscription_path(project_id, subscription_id)

    # Create the subscription (בלי פילטר בינתיים)
    subscription = subscriber.create_subscription(
        name=subscription_path,
        topic=topic_path
    )

    print(f"Created subscription: {subscription.name}")
    return subscription.name.rsplit('/')[-1]


# def get_subscriber(topic):
#     """
#     Create a subscription to a topic with filter
#     """
#     project_id = PROJECT_ID
#     topic_id = topic
#     uuid_str = str(uuid.uuid4())

#     # Create Pub/Sub clients
#     publisher = pubsub_v1.PublisherClient()
#     subscriber = pubsub_v1.SubscriberClient()

#     # Create paths
#     topic_path = publisher.topic_path(project_id, topic_id)
#     subscription_id = f'{topic_id}-subscription-{uuid_str}'
#     subscription_path = subscriber.subscription_path(project_id, subscription_id)

#     # Get environment ID for filtering
#     environment_id = os.getenv("ENVIRONMENT") or "MainServer"
#     if environment_id == "cloud_dev":
#         environment_id = "MainServer"

#     print(f"Creating subscription with filter for environment: {environment_id}")

#     try:
#         # Create subscription with filter
#         subscription = subscriber.create_subscription(
#             request={
#                 "name": subscription_path,
#                 "topic": topic_path,
#                 "filter": f'attributes.filter="{environment_id}"',
#                 "labels": {
#                     "environment": environment_id
#                 }
#             }
#         )

#         print(f"Created subscription: {subscription.name}")
#         print(f"With filter: attributes.filter='{environment_id}'")

#         return subscription.name.rsplit('/')[-1]

#     except Exception as e:
#         print(f"Error creating subscription: {e}")
#         raise e
#     finally:
#         subscriber.close()

def get_listening_subs(subs_list):
    print("Starting get_listening_subs")
    publisher = pubsub_v1.PublisherClient()
    subscriber = pubsub_v1.SubscriberClient()

    try:
        ping_sub_name = get_subscriber(TOPIC_PING)
        print(f"Created ping subscription: {ping_sub_name}")

        subscription_path = 'projects/{project_id}/subscriptions/{sub}'.format(
            project_id=PROJECT_ID,
            sub=ping_sub_name,
        )

        def callback(message):
            try:
                print(f"Got response: {message.data.decode('utf8')}")
                subs_list.append(message.data.decode("utf8"))
                message.ack()
            except Exception as e:
                print(f"Error in callback: {str(e)}")

        streaming_pull = subscriber.subscribe(subscription_path, callback=callback)
        print("Listening for responses...")

        # Send ping
        publisher.publish(
            f'projects/{PROJECT_ID}/topics/{TOPIC_EVAL}',
            b"ping"
        )
        print("Ping sent")

        # Wait for responses
        time.sleep(15)
        print(f"Found {len(subs_list)} subscribers")
        return subs_list

    except Exception as e:
        print(f"Error: {str(e)}")
        return subs_list


# def get_listening_subs(subs_list):
#     print("Starting get_listening_subs")
#     print(f"Looking for subscribers in environment: {environment_id}")

#     try:
#         # Use existing subscription ID format
#         subscription_id = f'Evaluation-{environment_id}'
#         print(f"Checking for subscription: {subscription_id}")

#         subscriber = pubsub_v1.SubscriberClient()
#         subscription_path = f'projects/{PROJECT_ID}/subscriptions/{subscription_id}'

#         # Add subscription to list if it exists
#         try:
#             subscriber.get_subscription(request={"subscription": subscription_path})
#             subs_list.append(subscription_id)
#             print(f"Found active subscription: {subscription_id}")
#         except Exception as e:
#             print(f"Error finding subscription: {str(e)}")

#         print(f"Found {len(subs_list)} subscribers: {subs_list}")
#         return subs_list

#     except Exception as e:
#         print(f"Error in get_listening_subs: {str(e)}")
#         print(traceback.format_exc())
#         return subs_list


def get_from_db(project_id, database, collection_name, document_id):
    # Initialize the Firestore client
    db = firestore.Client(project=project_id, database=database)

    # Reference to the document
    document_ref = db.collection(collection_name).document(document_id)

    # Retrieve the document data
    document_data = document_ref.get().to_dict()

    if document_data:
        print(f'Document ID: {document_id}, Data: {document_data}', type(document_data))
        return document_data
    else:
        print(f'Document ID {document_id} not found.')
        return None


async def get_from_db_by_short_id(project_id, database, collection_name, short_id):
    db = firestore.AsyncClient(project=project_id, database=database)
    collection_ref = db.collection(collection_name)

    # יצירת השאילתה לחיפוש המסמך לפי short_id
    query = collection_ref.where("short_id", "==", short_id)
    docs = query.stream()

    # שמירת המסמך הראשון שימצא
    async for doc in docs:
        doc_data = doc.to_dict()
        doc_data['id'] = doc.id
        print(f'Found document with short_id: {short_id}, Data: {doc_data}')
        return doc_data  # החזרת המסמך הראשון שנמצא

    # אם לא נמצא אף מסמך
    print(f'No documents found with short_id: {short_id}')
    return None


def store_on_db(project_id, database, collection_name, document_key, params):
    # Initialize the Firestore client with a specified namespace
    db = firestore.Client(project=project_id, database=database)

    # Set timezone to 'Asia/Jerusalem'
    israel_tz = pytz.timezone('Asia/Jerusalem')
    # Add a timestamp to the params
    params['timestamp'] = datetime.now(israel_tz).strftime("%Y-%m-%d %H:%M:%S")

    # Reference to the collection and document
    collection_ref = db.collection(collection_name)
    document_ref = collection_ref.document(document_key)
    # Set the JSON data to the document
    document_ref.set(params)

    print(f"file has been stored in Firestore under key '{document_key}' in database '{database}'.")


def db_log(name, text):
    job_idUuid = str(uuid.uuid4())
    document = {"name": name,
                "text": text}
    store_on_db(project_id=PROJECT_ID,
                database=FIRESTORE_DB,
                collection_name="Logs",
                document_key=job_idUuid,
                params=document)


def clean_env():
    def clean_dir(directory_path, top_level_directory):
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
        user_file_dir_path = get_files_package_root()
        os.remove(user_file_dir_path + r"/Estimator_params.json")
    except FileNotFoundError:
        pass


# import json
# with open(get_files_package_root() + "/Estimator_params.json", 'r') as f:
#     estimator_params = json.load(f)
# job_id = "123"
# set_estimator_params(project_id=os.getenv("PROJECT_ID"), database=os.getenv("FIRESTORE_DB"),
#                      collection_name=os.getenv("FIRESTORE_ESTIMATOR_COLLECTION"), document_key=job_id,
#                      params=estimator_params)
# project_id,database ,collection_name, document_id
# doc  = get_from_db(project_id=os.getenv("PROJECT_ID"), database=os.getenv("FIRESTORE_DB"),
#                      collection_name=os.getenv("FIRESTORE_ESTIMATOR_COLLECTION"), document_id=job_id)
#
# print(doc, type(doc))


def get_validate_id_by_eval_id(project_id: str, database: str, collection_name: str, eval_job_id: str) -> str:
    try:
        # Get the evaluation request document
        eval_doc = get_from_db(
            project_id=project_id,
            database=database,
            collection_name=collection_name,
            document_id=eval_job_id
        )
        if eval_doc and 'request' in eval_doc:
            # Extract validation_id from the request
            validation_id = eval_doc['request'].get('validation_id')
            if validation_id:
                return validation_id

        return None
    except Exception as e:
        print(f"Error getting validation ID: {str(e)}")
        return None


async def get_nodepools_from_db(project_id, database, status: str = None):
    """
    Get nodepools from Firestore database with optional status filter

    Args:
        project_id: Google Cloud project ID
        database: Firestore database name
        status: Optional status filter ('active', 'deleted', etc.)

    Returns:
        List of nodepool documents
    """
    try:
        db = firestore.Client(project=project_id, database=database)
        collection_ref = db.collection("NodePools")

        # Filter by status if provided
        if status:
            docs = collection_ref.where("status", "==", status).stream()
        else:
            docs = collection_ref.stream()

        nodepools = []
        for doc in docs:
            nodepool_data = doc.to_dict()
            nodepool_data['document_id'] = doc.id
            nodepools.append(nodepool_data)

        return nodepools

    except Exception as e:
        db_log(environment_id, f"Error in get_nodepools_from_db: {str(e)}")
        raise Exception(f"Failed to fetch nodepools from database: {str(e)}")
    
def get_validation_id_from_requests(project_id, database, document_id):
    """
    Get validation_id from Requests collection for a specific document
    
    Args:
        project_id: The project id
        database: The database name
        document_id: The document id to fetch
    
    Returns:
        validation_id: The validation_id from the request object or None if not found
    """
    try:
        # Initialize Firestore client
        db = firestore.Client(project=project_id, database=database)
        
        # Get the document
        doc = db.collection("Requests").document(document_id).get()
        
        if not doc.exists:
            print(f"Document {document_id} not found in Requests collection")
            return None
            
        # Get the data
        doc_data = doc.to_dict()
        
        # Extract validation_id from the request field
        if 'request' in doc_data and isinstance(doc_data['request'], dict):
            validation_id = doc_data['request'].get('validation_id')
            return validation_id
            
        return None
        
    except Exception as e:
        print(f"Error getting validation_id: {str(e)}")
        return None
    
async def check_nodepool_last_activity(nodepool_id: str, hours: int = 6) -> bool:
    """
    Check if a nodepool has had any activity in the logs within the specified hours
    Returns True if the nodepool is inactive (no logs in specified period)
    """
    try:
        db = firestore.Client(project=PROJECT_ID, database=FIRESTORE_DB)
        logs_ref = db.collection("Logs")

        # שלב 1 - יצירת אובייקט timezone
        israel_tz = pytz.timezone('Asia/Jerusalem')
        db_log(environment_id, f"1. Timezone created: {israel_tz}")

        # שלב 2 - קבלת הזמן הנוכחי עם timezone
        current_time = datetime.now(israel_tz)
        db_log(environment_id, f"2. Current time: {current_time}")

        # שלב 3 - חישוב הזמן להפחתה
        time_delta = timedelta(hours=hours)
        db_log(environment_id, f"3. Time delta: {time_delta}")

        # שלב 4 - חישוב זמן הסף
        cutoff_time = current_time - time_delta
        db_log(environment_id, f"4. Cutoff time: {cutoff_time}")

        cutoff_str = cutoff_time.strftime("%Y-%m-%d %H:%M:%S")
        db_log(environment_id, f"5. Cutoff string: {cutoff_str}")
        
        # Query for any logs from this nodepool after cutoff time
        query = logs_ref.where("name", "==", nodepool_id) \
                       .where("timestamp", ">=", cutoff_str) \
                       .limit(1)
        
        logs = list(query.stream())
        
        return len(logs) == 0
        
    except Exception as e:
        db_log(environment_id, f"Error checking activity for nodepool {nodepool_id}: {str(e)}")
        raise