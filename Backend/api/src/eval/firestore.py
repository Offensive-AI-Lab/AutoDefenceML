from google.cloud import firestore
from datetime import datetime

def set_report(project_id, database, collection_name, document_key, report):
    """
    This code collects reports on defences ready from other node pools and merges (updates) it with the provided report
    Set the report in Firestore under the specified key.
    :param project_id: The project id.
    :param database: The database name.
    :param collection_name: The collection name.
    :param document_key: The document key.
    :param report: The report to store.
    """
    # Initialize the Firestore client with a specified namespace
    db = firestore.Client(project=project_id, database=database)

    # Reference to the collection and document
    collection_ref = db.collection(collection_name)
    document_ref = collection_ref.document(document_key)

    # Check if the document already exists
    if document_ref.get().exists:
        # If the document exists, retrieve the existing data
        existing_data = document_ref.get().to_dict()

        # Append the new JSON data to the existing data
        existing_data.update(report)
        report = existing_data

    # Add a timestamp to the params
    # report['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Set the JSON data to the document
    document_ref.set(report)

    print(f"report has been stored in Firestore under key '{document_key}' in database '{database}'.")
