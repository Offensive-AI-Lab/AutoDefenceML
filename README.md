# AutoDefenceML

## Overview

**AutoDefenceML (ADML)** is an open-source initiative designed to enhance the security of machine learning (ML) models. In the age of AI-driven technologies, ensuring the security and integrity of ML systems is critical. From data poisoning to adversarial attacks, ML models face numerous vulnerabilities. AutoDefenceML addresses these challenges by automating the security evalaution process, helping ML engineers identify potential security threats and providing actionable recommendations to bolster defenses.

This platform aims to bridge the gap between ML development and security expertise, making robust ML security accessible to all engineers, regardless of their background in cybersecurity. By automating complex vulnerability assessments and defense recommendations, it streamlines the path to safer, more reliable AI deployments.


## Features

AutoDefenceML is packed with powerful features to ensure the robustness and fairness of your ML models and datasets. Here's what it offers:

### 1. `Model Evaluation` 
- Comprehensive evaluation of model robustness to adversarial examples.
- Defence Evalautions are automatically scaled over a GPU node-pool over GCP.
- Assesment of the latest attacks and defences; some developed for this platform exclusivly and the rest sourced from the [ART](https://github.com/Trusted-AI/adversarial-robustness-toolbox) community.
- Evaluation of adaptive adversaries against potential defences.
- A JSON and PDF health report detailing vulnerabilities and recommended optimal defenses.
- Optional hyperparameter optimization for both the attacks and defences.
- Supports TensorFlow, PyTorch, Scikit-learn, XGBoost, and CatBoost.
- Supports various modalities, including image and tabular data.

### 2. `Data Evaluation` 
- Automatic detection of poisoning in training datasets using the method of [Cretu, Gabriela F., et al](https://ieeexplore.ieee.org/abstract/document/4531146).
- A JSON and PDF report identifying suspicious samples in the dataset.
  
### 3. `Bias Evaluation`
- Identifies biases in provided training datasets.
- Evalutates a suite of bias mitigation methods to improve fairness.
- Produces comprehensive JSON and PDF reports on bias detection and correction.

#### User Experience
*In terms of user experience, ADML offer the following:*
- A REST API for easy intergration and interaction
- The ability to provide a custom model and dataset python definitions or non-conventional formats.
- The ability to dynamically change python requiremetns in individual evaautuions (under developent)

#### Limitations
*AutoDefenceML is in its early stages of development so there are some limitations to be aware of:*
- **Model Support:** Limited to classification models and specific frameworks (TensorFlow, PyTorch, Scikit-learn, XGBoost, CatBoost).
- **Modality:** While the `Model Evlaution` can handle any non-sequential data (images, tabular, ...), the `Data Evalaution` and `Bias Evaluations` only support tabular data at this time.
- **Adversarial Examples** Currently, the `Model Evalaution` only evaluates *untargeted* adversarial examples.


## Tech Stack

- FastAPI
- Google Cloud Platform (GCP)
  - Cloud Firestore
  - Pub/Sub
  - Kubernetes Engine
  - Container Registry
- Terraform
- Python ML Frameworks Support:
  - PyTorch
  - TensorFlow
  - scikit-learn
  - XGBoost
  - Keras
  - CatBoost

## Prerequisites

- Python 3.10
- Google Cloud SDK
- Terraform
- kubectl
- Docker

## Environment Setup

1. Clone the repository
2. Create a `.env` file with the following variables:

```env
PROJECT_ID=your_project_id
FIRESTORE_DB=your_database
FIRESTORE_REPORTS_COLLECTION=your_reports_collection
FIRESTORE_ESTIMATOR_COLLECTION= your_estimator_params_collection
FIRESTORE_EVAL_STATUS_COLLECTION=your_eval_status_collection
FIRESTORE_VAL_STATUS_COLLECTION=your_validate_status_collection
FIRESTORE_BIAS_VAL_STATUS_COLLECTION= your_bias_validate_status_collection
FIRESTORE_BIAS_DET_STATUS_COLLECTION = your_bias_detection_status_collection
FIRESTORE_BIAS_MIT_STATUS_COLLECTION = your_bias_mitigation_status_collection
FIRESTORE_DATA_VAL_STATUS_COLLECTION= your_data_validate_status_collection
FIRESTORE_DATA_EVAL_STATUS_COLLECTION = your_data_evaluate_status_collection
TOPIC_EVAL=your_eval_topic
FROM_BUCKET=your_bucket_status (TRUE or FALSE for local)
BUCKET_NAME=your_bucket_name
ACCOUNT_SERVICE_KEY=your_service_key

FILES_PATH_VAL=src.user_files
FILES_PATH_EVAL=src.eval.user_files_eval
FILES_PATH_BIAS=src.bias_eval.user_files_bias
FILES_PATH_DATA=src.data_eval.user_files_data
```

## LocalHost Installation

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Install additional packages:

```bash
pip install --index-url https://us-central1-python.pkg.dev/autodefenseml/filehandler/simple/ file-loader
pip install --index-url https://us-central1-python.pkg.dev/autodefenseml/art-handler/simple/ art-handler
pip install --index-url https://me-west1-python.pkg.dev/autodefenseml/art-attacks-plugin/simple/ art-attacks-plugin

```

## GCP Installation

### 1. CI/CD Setup

First, configure the CI/CD pipeline by setting up the Terraform configuration for the build triggers.

#### Configure terraform.tfvars

Navigate to the CI/CD Terraform directory and create/modify `terraform.tfvars`:

```hcl
project_id     = "autodefenseml"
branch         = "master"
region         = "us-central1"
repo_name      = "api"
git_owner      = "MABADATA"
repository_id  = "autodefenseml"

services_names = {
  "terraform" = {
    trigger_name  = "api-terraform-trigger"
    manager_dir   = "api"
    image_name    = "app"
  }
}
```

#### Initialize and Apply CI/CD Infrastructure

```bash
# Initialize Terraform
terraform init

# Review planned changes
terraform plan

# Apply CI/CD configuration
terraform apply
```

### 2. Infrastructure Deployment

After setting up CI/CD, deploy the main infrastructure components.

#### Navigate to Infrastructure Directory

```bash
cd Backend/api/src/Infrastructure
```

#### Deploy Infrastructure Components

```bash
# Initialize Terraform for infrastructure
terraform init

# Review infrastructure changes
terraform plan

# Deploy infrastructure
terraform apply
```

This will create:

- Kubernetes cluster configuration
- GPU-enabled node pools
- Load balancers
- Service accounts and IAM roles
- Network configuration
- Storage buckets
- Pub/Sub topics and subscriptions

### 3. Post-Installation Verification

After both CI/CD and infrastructure deployment are complete:

1. Verify build triggers:

```bash
gcloud builds triggers list
```

2. Configure kubectl:

```bash
gcloud container clusters get-credentials autodefenseml --region us-central1
```

3. Verify kubernetes deployment:

```bash
kubectl get nodes
kubectl get pods
```

4. Check GPU availability:

```bash
kubectl get nodes -l gpu=true
```

### Infrastructure Cleanup

To remove all created resources:

```bash
# Clean up infrastructure
cd Backend/api/src/Infrastructure
terraform destroy

# Clean up CI/CD
cd <ci-cd-terraform-directory>
terraform destroy
```

Installation

Install dependencies:

bashCopypip install -r requirements.txt

Install additional packages:

bashCopypip install --index-url https://us-central1-python.pkg.dev/autodefenseml/filehandler/simple/ file-loader
pip install --index-url https://us-central1-python.pkg.dev/autodefenseml/art-handler/simple/ art-handler

# API Endpoints

### Model Evaluation Feature

- `POST /validate/` - Validate model configuration and compatibility
- `GET /validation_status/{job_id}` - Get validation status
-  `POST /evaluate/` -Evaluate the model with specific attacks and defenses
- `GET /evaluation_status/{job_id}` - Get evaluation status

### Bias Validation

- `POST /bias_validate/` - validate the dataset and dataloadedr the user uploaded
- `GET /bias_validate_status/{job_id}` - Get bias validation status

### Bias Detection

- `POST /bias_detection/` - Perform bias detection analysis
- `GET /bias_detection_status/{job_id}` - Get bias detection status

### Bias Mitigation

- `POST /bias_mitigation/` - Apply bias mitigation strategies
- `GET /bias_mitigation_status/{job_id}` - Get bias mitigation status

### Dataset Operations

- `POST /dataset_validate/` - Validate dataset structure
- `POST /dataset_evaluate/` - Evaluate dataset characteristics
- `GET /dataset_validate_status/{job_id}` - Get dataset validation status
- `GET /dataset_evaluate_status/{job_id}` - Get dataset evaluation status

### GPU Management

- `GET /listofgpu/` - List available GPU nodes
- `DELETE /removegpu/{gpu_id}` - Remove specific GPU node
- `DELETE /delete_node_pool_and_deployment/{short_id}` - Delete node pool and deployment

## Infrastructure Management

The service uses Terraform for infrastructure management and includes:

- Kubernetes cluster configuration
- GPU node pool management
- Load balancer setup
- Service account configuration

## Docker Support

Build the Docker image using the provided Dockerfile:

```bash
docker build -t ml-validation-service .
docker run -p 8080:8080 ml-validation-service
```

## Development

### Adding New Features

1. Create new router in `src/api/routers/`
2. Register router in `main.py`
3. Update infrastructure code if needed
4. Add tests and documentation

### Testing

Run tests using:

```bash
pytest tests/
```

## Troubleshooting

Common issues and solutions:

1. GPU Node Pool Issues

   - Check permissions
   - Verify GCP configuration
   - Ensure correct service account setup

2. Firestore Connection
   - Verify credentials
   - Check project configuration
   - Ensure correct collection names

## Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Create pull request

## License

MIT License

## Acknowledgements 
Thank you to all the amazing contributors! (listed in no particular order)
- Eran Simtob
- Michal Alhindi
- Roey Bokobza
- Tomer Meshulam

Project lead: Yisroel Mirsky
