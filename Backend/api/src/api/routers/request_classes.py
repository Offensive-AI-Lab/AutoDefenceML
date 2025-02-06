from pydantic import BaseModel
from typing import Optional , Dict
from enum import Enum
from typing import Union


class Definition(BaseModel):
    uid: str
    path: str
    class_name: str


class Parameters(BaseModel):
    uid: str
    path: str


# Enum for the framework and ml_type fields and device
class FrameworkEnum(str, Enum):
    pytorch = "pytorch"
    tensorflow = "tensorflow"
    sklearn = "sklearn"
    xgboost = "xgboost"
    keras = "keras"
    catboost = "catboost"


class MLTypeEnum(str, Enum):
    classification = "classification"
    regression = "regression"


class Device(str, Enum):
    device = "gpu"


class Meta(BaseModel):
    definition: Definition
    parameters: Parameters
    framework: FrameworkEnum
    ml_type: MLTypeEnum


#  Dim #####################################
class Dim(BaseModel):
    input: Union[int, tuple[int, ...]]
    num_classes: int = None
    clip_values: tuple[int, ...] = None


#  Loss #####################################
class Loss(BaseModel):
    uid: Optional[str] = None
    path: Optional[str] = None
    type: Optional[str] = None


#  Optimizer #####################################
class Optimizer(BaseModel):
    type: Optional[str] = None
    learning_rate: Optional[float] = None


# Base model for "ml_model" field
class MLModel(BaseModel):
    meta: Meta
    dim: Dim
    loss: Loss
    optimizer: Optional[Optimizer] = None



# Dataset #####################################
class Dataset(BaseModel):
    uid: Optional[str] = None
    path: Optional[str] = None

class Req_file(BaseModel):
    uid: Optional[str] = None
    path: Optional[str] = None
# Base model for "dataloader" field
class DataLoader(BaseModel):
    definition: Definition


class TargetEnum(str, Enum):
    NA = "NA"
    targeted = "targeted"
    untargeted = "untargeted"

class Attacks(BaseModel):
    class_name: list[str]

class Defenses(BaseModel):
    class_name: list[str]

class HyperparametersOptimization(BaseModel):
    hyperparameters_optimization_defense: bool
    hyperparameters_optimization_attack: bool
    epsilon: Optional[float]
    max_attack_iterations: Optional[Dict[str, int]]


class Configuration(BaseModel):
    timeout: int
    attack_time_limit: int
    num_iter: int


# Base model for the entire request body
class ValidationRequestBody(BaseModel):
    user_id: str
    ml_model: MLModel
    dataloader: DataLoader
    # train_set: Optional[Dataset] = None
    test_set: Dataset
    req_file: Optional[Req_file] = None



class ModelEvalRequestBody(BaseModel):
    user_id: str
    ml_model: MLModel
    dataloader: DataLoader
    test_set: Dataset
    attacks: Attacks
    defense: Defenses
    HyperparametersOptimization: HyperparametersOptimization
    validation_id: str


class DatasetValidationRequestBody(BaseModel):
    user_id: str
    dataloader: DataLoader
    dataset: Dataset

class DatasetEvaluationRequestBody(BaseModel):
    user_id: str
    dataloader: DataLoader
    dataset: Dataset



class BiasvalidationRequestBody(BaseModel):
    user_id: str
    dataloader: DataLoader
    dataset: Dataset

class BiasDetectionRequestBody(BaseModel):
    user_id: str
    dataloader: DataLoader
    dataset: Dataset


class PrivFeatures(BaseModel):
    name: str
    value: str


class Mitigations(BaseModel):
    name: str
    description: str


class BiasMitigationRequestBody(BaseModel):
    user_id: str
    dataloader: DataLoader
    dataset: Dataset
    priv_features: list[PrivFeatures]
    mitigations: list[Mitigations]
    download_url: Optional[str] = None

class ManualStopRequestBody(BaseModel):
    job_id: str

