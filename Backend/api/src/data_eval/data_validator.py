import pandas as pd
from .ensemble_depoisoning.main import run_ensemble_depoisoning
import json
import logging
import os
import traceback
from importlib import import_module
import importlib.util
import pandas as pd
from gcloud import storage
class BucketLoaderDataset:
    def __init__(self, request,
                 path_to_files_dir,
                path_to_dataloader_files_dir,
                 path_to_dataset_files_dir,
                 bucket_name,
                 account_service_key_path):

        self.bucket_name = bucket_name
        self.account_service_key_path = account_service_key_path
        self.path_to_files_dir = path_to_files_dir
        self.path_to_dataloader_files_dir = path_to_dataloader_files_dir
        self.path_to_dataset_files_dir = path_to_dataset_files_dir
        self.request = request
        if isinstance(self.request, str):
            self.request = json.loads(self.request)
        if isinstance(self.request, dict):
            # Extracting the parts regrading the model it's self
            try:

                # Extracting the parts regrading the dataloader
                self.__dataloader_file_id = self.request['dataloader']['definition']['uid']
                self.__dataloader_file_URL = self.request['dataloader']['definition']['path']
                self.__dataloader_file_class = self.request['dataloader']['definition']['class_name']
                # Extracting the parts regrading the test set
                self.__dataset_id = self.request['dataset']['uid']
                self.__dataset_URL = self.request['dataset']['path']

                ACCOUNT_SERVICE_KEY = self.account_service_key_path
                os.environ["ACCOUNT_SERVICE_KEY"] = ACCOUNT_SERVICE_KEY
            except Exception as err:
                logging.error(f"Metadata is not valid!\nError:\n{err}")
                logging.error(traceback.format_exc())

        else:
            raise TypeError('meta data need to be type dict or str')

    def get_client(self):
        """
        Function to get the client of the GCP bucket
        :return: the client of the GCP bucket
        """
        ACCOUNT_SERVICE_KEY = os.environ.get('ACCOUNT_SERVICE_KEY')
        # Check if ACCOUNT_SERVICE_KEY is defined
        if ACCOUNT_SERVICE_KEY is None:
            raise ValueError("ACCOUNT_SERVICE_KEY environment variable is not defined")
        # Check if BUCKET_NAME is defined
        if self.bucket_name is None:
            raise ValueError("BUCKET_NAME environment variable is not defined")
        # os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ACCOUNT_SERVICE_KEY
        os.environ["DONT_PICKLE"] = 'False'
        storage_client = storage.Client()
        bucket = storage_client.bucket(self.bucket_name)
        return bucket

    def upload_to_gcp(self, dest_file_path, src_file_name):
        """
        Function to upload files to the GCP bucket
        :param dest_file_path: the path of the file in the bucket
        :param src_file_name: the path of the file in the local machine
        :return:
        """
        try:
            bucket = self.get_client()
            blob = bucket.blob(dest_file_path)
            blob.upload_from_filename(src_file_name)
        except Exception as e:
            # Handle the exception here, you can log the error or take appropriate actions
            logging.error(f"An error occurred while uploading to GCP: {e}")
            logging.error(traceback.format_exc())
            # You can also raise the exception again if you want to propagate it



    def download_from_gcp(self, src_file_name,dest_file_name=None,folder=False):
        """
              Function to download files from the GCP bucket
              :param src_file_name: the path of the file in the bucket
              :param dest_file_name: the path of the file in the local machine
              :return:
              """
        bucket = self.get_client()
        src_file_name = BucketLoaderDataset.reformat_path(src_file_name, self.bucket_name)
        if folder:
            blobs = bucket.list_blobs()
            path = os.path.join(self.path_to_dataset_files_dir, "test_set")
            os.mkdir(path)
            for blob in blobs:
                file_name = blob.name
                if file_name.find(src_file_name) != -1 and not file_name.endswith(src_file_name + "/"):
                    dir = os.path.dirname(file_name)
                    full_dir = os.path.join(path, dir[len(src_file_name) + 1:])
                    if not os.path.exists(full_dir):
                        os.makedirs(full_dir)
                    dest = os.path.join(path , file_name[len(src_file_name) + 1:])


                    blob.download_to_filename(dest)
            return
        blob = bucket.blob(src_file_name)
        if dest_file_name:
            blob.download_to_filename(dest_file_name)
        else:
            blob.download_to_filename(src_file_name)

    @staticmethod
    def reformat_path(path, bucket_name):
        start_index = path.find(bucket_name)
        if start_index == -1:
            return path
        return path[start_index + len(bucket_name) + 1:]


    def get_dataloader(self):
        """
                Function to get the dataloader from the bucket
                """

        dataloader_dest = self.path_to_dataloader_files_dir + f"/dataloader_def.py"
        self.download_from_gcp(src_file_name=self.__dataloader_file_URL,
                               dest_file_name=dataloader_dest)


    def get_dataset(self):
        def get_data_format():
            client = self.get_client()
            blobs = client.list_blobs()

            file_names = [blob.name for blob in blobs]

            true_file_name = self.__dataset_id
            for file_name in file_names:
                file_format_index = file_name.rfind(".")
                file_format = file_name[file_format_index:]
                if (true_file_name + "/") in file_name:
                    return None
                elif file_format_index == -1:
                    continue
                elif file_name.endswith(true_file_name + file_format):
                    return file_format
                else:
                    continue

            raise Exception("Dataset not found in Bucket")


        data_format = get_data_format()
        is_folder = False
        if data_format is None:
            data_format = ""
            is_folder = True

        dataset_dest = self.path_to_dataset_files_dir + "/dataset" + data_format
        self.download_from_gcp(src_file_name=self.__dataset_URL,
                               dest_file_name=dataset_dest,folder=is_folder)
        if data_format[1:] == "zip":
            import zipfile
            folder_to_extract_zip = self.path_to_dataset_files_dir + "/dataset"
            with zipfile.ZipFile(dataset_dest, "r") as zip_ref:
                zip_ref.extractall(folder_to_extract_zip)

class FileHandlerDataset:
    def __init__(self, request,
                 path_to_files_dir,
                 path_to_dataloader_files_dir,
                 path_to_dataset_files_dir,
                 from_bucket,
                 bucket_name=None,
                 account_service_key_name=None):
        """
                Initialize a FileLoader instance.

                Parameters:
                -----------
                metadata (dict or str): Metadata related to the files and objects being loaded.

                """
        self.path_to_files_dir = path_to_files_dir
        self.path_to_dataloader_files_dir = path_to_dataloader_files_dir
        self.path_to_dataset_files_dir = path_to_dataset_files_dir
        self.from_bucket = bool(from_bucket)
        self.bucket_name = bucket_name
        self.account_service_key_path = self.path_to_files_dir + "/" + account_service_key_name
        if self.from_bucket:
            self.bucket_loader = BucketLoaderDataset(request,
                                              self.path_to_files_dir,
                                              self.path_to_dataloader_files_dir,
                                              self.path_to_dataset_files_dir,
                                              self.bucket_name,
                                              self.account_service_key_path)
        # self.from_bucket = False
        self.request = request
        if isinstance(self.request, str):
            self.request = json.loads(self.request)
        if isinstance(self.request, dict):
            self.__dataset_file_id = self.request['dataset']['uid']
            self.__dataloader_file_id = self.request['dataloader']['definition']['uid']
            # self.input_validator = InputValidator(metadata)


    def check_file_in_local(self, dir_path, expected_file):
        """
                        Check if a file exists in the local directory.

                        Parameters:
                        -----------
                        dir_path (str): The directory path to check.
                        expected_file (str): The name of the file to check for.

                        Returns:
                        --------
                        bool: True if the file exists, False otherwise.

                        """

        def check_single(dir_path, expected_file):
            scaner = os.scandir(path=dir_path)
            for entry in scaner:
                if entry.is_dir() or entry.is_file():
                    if entry.name == expected_file:
                        return True
            return False

        if isinstance(expected_file, list):
            return all([check_single(dir_path, file) for file in expected_file])
        else:
            return (check_single(dir_path, expected_file))

    def load_module_from_file(self,module_name, file_path):
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def get_dataloader(self):
        def get_dataset_path():
            path_to_dataset = None
            files_in_dataset_folder = os.listdir(self.path_to_dataset_files_dir)
            for file in files_in_dataset_folder:
                if file.startswith("dataset") and not file.endswith("zip"):
                    path_to_dataset = file
            if path_to_dataset is None:
                return None
            return self.path_to_dataset_files_dir + "/" + path_to_dataset

        def get_dataloader_from_local():
            class_name = self.request['dataloader']['definition'][
                'class_name']
            if self.from_bucket == True:
                path_to_dataset = get_dataset_path()
            else:
                path_to_dataset = self.request['dataset']['path']
            try:
                if self.from_bucket == True:
                    dataloader_path = os.path.join(self.path_to_dataloader_files_dir, 'dataloader_def.py')

                    # Load the dataloader module
                    dataloader_module = self.load_module_from_file('dataloader_def', dataloader_path)
                    dataloader = getattr(dataloader_module, class_name)(path_to_dataset)

                    # dataloader = getattr(import_module(".".join([os.getenv("FILES_PATH_DATA"), "dataloader.dataloader_def"])),
                    #                   class_name)(
                    #     path_to_dataset)
                else:
                    dataloader = getattr(import_module(self.request['dataloader']['definition']['path']), class_name)(
                        path_to_dataset)
            except KeyError:
                raise ValueError(f"Class '{class_name}' not found or imported.")
            return dataloader

        if get_dataset_path() is None:
            if self.from_bucket:
                self.bucket_loader.get_dataset()
            else:
                raise Exception("Could not found dataset in local")
            if get_dataset_path() is None:
                raise Exception("Could not found dataset in cloud storge")
        # check if the ML model's file is in the folder
        dataloader_expected_file = "dataloader_def.py"
        file_in_local = self.check_file_in_local(dir_path=self.path_to_dataloader_files_dir,
                                                 expected_file=dataloader_expected_file)

        if file_in_local:
            dataloader = get_dataloader_from_local()
            return dataloader

        elif self.from_bucket:
            try:
                self.bucket_loader.get_dataloader()
            except Exception as err:
                raise Exception(f"Failed to get dataloader from bucket:\nError: {err}").with_traceback(
                    err.__traceback__)

            dataloader = get_dataloader_from_local()
            return dataloader

        else:
            raise FileNotFoundError("dataloader file not found")
class DataValidator:

    def __init__(self, fileloader_obj) -> None:

        self.dataframe = None
        self.target_name = None
        self.__file_loader = fileloader_obj
        self.metadata = fileloader_obj.request
        # self.__input = {"Datloader": , "Dataset": }

    def validate_dataset(self):
        dataloader = self.__file_loader.get_dataloader()
        x, y = dataloader.get_data()
        # asset x and y are not empty and pd dataframe
        if x is None or y is None:
            raise ValueError("x and y should not be None")
        if not isinstance(x, pd.DataFrame) or not isinstance(y, pd.DataFrame):
            raise TypeError("x and y should be pandas dataframe")

        # assert x.shape[0] == y.shape[0]
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y should have the same number of samples")

        # Save the column name of "Y" (target)
        self.target_name = y.columns[0]

        # vstack X and Y together
        self.dataframe = pd.concat([x, y], axis=1)

        return True

    def get_dataframe(self):
        return self.dataframe

    def get_target_name(self):
        return self.target_name

    def validate_dataloader(self):
        dataloader = self.__file_loader.get_dataloader()
        # check if method get_batch exists
        if not hasattr(dataloader, 'get_batch') or not hasattr(dataloader, 'get_data'):
            raise AttributeError("get_batch method or get_data not found in dataloader")
        return True



def evaluate(dataframe, target_name):
    report,pdf_string = run_ensemble_depoisoning(dataframe, target_name)
    return report,pdf_string