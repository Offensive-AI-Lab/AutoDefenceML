import os
from google.cloud import pubsub_v1
import uuid
from dotenv import load_dotenv
load_dotenv()
PROJECT_ID = os.getenv("PROJECT_ID")

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
    subscription_id = f'{topic_id}-subscription-{uuid_str}'
    subscription_path = subscriber.subscription_path(project_id, subscription_id)

    # Create the subscription (בלי פילטר)
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

#     # Get environment ID for filtering
#     environment_id = os.getenv("ENVIRONMENT") or "MainServer"
#     if environment_id == "cloud_dev":
#         environment_id = "MainServer"
    
#     publisher = pubsub_v1.PublisherClient()
#     subscriber = pubsub_v1.SubscriberClient()
    
#     # Use a consistent subscription ID for each environment
#     subscription_id = f'{topic_id}-{environment_id}'
#     subscription_path = subscriber.subscription_path(project_id, subscription_id)
#     topic_path = publisher.topic_path(project_id, topic_id)

#     try:
#         # Try to get existing subscription
#         subscription = subscriber.get_subscription(request={"subscription": subscription_path})
#         print(f"Found existing subscription: {subscription.name}")
#         return subscription_path.rsplit('/')[-1]
#     except Exception:
#         print(f"Creating new subscription for {environment_id}")
#         # Create new subscription with filter
#         subscription = subscriber.create_subscription(
#             request={
#                 "name": subscription_path,
#                 "topic": topic_path,
#                 "filter": f'attributes.filter = "{environment_id}"'
#             }
#         )
#         print(f"Created new subscription: {subscription.name}")
#         return subscription.name.rsplit('/')[-1]