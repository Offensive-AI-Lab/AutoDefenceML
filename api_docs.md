# AutoDefenseML API Documentation

## Overview

This document describes the API endpoints available in the AutoDefenseML system. The API is built with FastAPI and provides functionality for:

1. Model validation: Checks if the provided model, test data, and class files are compatible and returns a list of attacks+defences that are compatible for evaluation (supports tabular and image datasets)
2. Model evaluation: Evalautes the securtity of the provided model against adversarial example attacks and suggests the best pre/postprocessors defence. Both the attacks and defences can be optimized.
3. Dataset validation: Checks if the provided dataset and data loader are compatible (tabular datasets only)
4. Dataset evaluation: Searches the dataset for potential corruption based poisoning attacks, and returns the intedex to these samples.
5. Bias validation: Checks if the provided dataset and data loader are compatible (tabular datasets only)
6. Bias detection: Measures bias in each of the features
7. Bias mitigation: Mitigates bias in each of the indicates featurs using the provided list of algorithms.
8. Job control (stopping execution)

All endpoints return appropriate HTTP status codes:
- 200: Successful operation
- 404: Resource not found
- 422: Validation error
- 500: Server error

## Common Response Patterns

Most POST endpoints return a job ID that can be used to check the status of asynchronous operations:

```json
{
  "job_id": "user_id-uuid"
}
```

Status endpoints typically return information in the following format:

```json
{
  "job_id": "string",
  "process_stage": "string",
  "process_status": "string",
  "error": "string",
  "stack trace": "string"
}
```

Where `process_status` can be:
- `Running`: Operation is still in progress
- `Done`: Operation completed successfully
- `Failed`: Operation failed
- `Done with failures`: Operation completed with some failures

## Authentication

All endpoints require a `user_id` in the request body for POST requests.

## Model Validation

### Validate Model

Validates a machine learning model and its compatibility with the provided dataset.

**Endpoint:** `POST /validate/`

**Request Body:**

```json
{
  "user_id": "string",
  "ml_model": {
    "meta": {
      "definition": {
        "uid": "string",
        "path": "url",
        "class_name": "string"
      },
      "parameters": {
        "uid": "string",
        "path": "url"
      },
      "framework": "string",
      "ml_type": "string"
    },
    "dim": {
      "input": [int],
      "num_classes": int,
      "clip_values": [int]
    },
    "loss": {
      "uid": "string",
      "path": "url",
      "type": "string"
    },
    "optimizer": {
      "type": "string",
      "learning_rate": float
    }
  },
  "dataloader": {
    "definition": {
      "uid": "string",
      "path": "url",
      "class_name": "string"
    }
  },
  "test_set": {
    "uid": "string",
    "path": "url"
  },
  "req_file": {
    "uid": "string",
    "path": "url"
  }
}
```

**Response:**

```json
{
  "job_id": "string"
}
```

### Get Validation Status

Checks the status of a validation job.

**Endpoint:** `GET /validation_status/{job_id}`

**Parameters:**
- `job_id`: Job ID returned from the validation request

**Response:**

```json
{
  "job_id": "string",
  "process_status": "string",
  "process_stage": "string",
  "error": "string",
  "stack trace": "string",
  "compatible_attacks": [
    {
      "influence": "string",
      "assumption": "string",
      "class_name": "string",
      "has_max_iter": boolean,
      "default_max_iter": int,
      "p-norm": "string",
      "run_time": "string",
      "name": "string",
      "type": "string",
      "violation": "string",
      "description": "string"
    }
  ],
  "compatible_defenses": [
    {
      "class_name": "string",
      "name": "string",
      "type": "string",
      "description": "string"
    }
  ]
}
```

## Model Evaluation

### Evaluate Model

Evaluates a model against various attacks and defenses.

**Endpoint:** `POST /evaluate/`

**Request Body:**

```json
{
  "user_id": "string",
  "ml_model": {
    "meta": {
      "definition": {
        "uid": "string",
        "path": "url",
        "class_name": "string"
      },
      "parameters": {
        "uid": "string",
        "path": "url"
      },
      "framework": "string",
      "ml_type": "string"
    },
    "dim": {
      "input": [int],
      "num_classes": int,
      "clip_values": [int]
    },
    "loss": {
      "uid": "string",
      "path": "url",
      "type": "string"
    },
    "optimizer": {
      "type": "string",
      "learning_rate": float
    }
  },
  "dataloader": {
    "definition": {
      "uid": "string",
      "path": "url",
      "class_name": "string"
    }
  },
  "test_set": {
    "uid": "string",
    "path": "url"
  },
  "attacks": {
    "class_name": ["string"]
  },
  "defense": {
    "class_name": ["string"]
  },
  "HyperparametersOptimization": {
    "hyperparameters_optimization_defense": boolean,
    "hyperparameters_optimization_attack": boolean,
    "epsilon": float,
    "max_attack_iterations": {"string": int}
  },
  "validation_id": "string"
}
```

**Response:**

```json
{
  "job_id": "string"
}
```

### Get Evaluation Status

Checks the status of an evaluation job and retrieves results.

**Endpoint:** `GET /evaluation_status/{job_id}`

**Parameters:**
- `job_id`: Job ID returned from the evaluation request

**Response:**

```json
{
  "job_id": "string",
  "process_status": "string",
  "process_stage": "string",
  "num_of_defenses": int,
  "report": {
    "clean_model_evaluation": {},
    "defense1": {},
    "defense2": {}
  },
  "pdf": "base64_encoded_string",
  "error": "string",
  "stack trace": "string",
  "elapsed_time": "string"
}
```

## Dataset Operations

### Validate Dataset

Validates a dataset for compatibility with machine learning models.

**Endpoint:** `POST /dataset_validate/`

**Request Body:**

```json
{
  "user_id": "string",
  "dataloader": {
    "definition": {
      "uid": "string",
      "path": "url",
      "class_name": "string"
    }
  },
  "dataset": {
    "uid": "string",
    "path": "url"
  }
}
```

**Response:**

```json
{
  "job_id": "string"
}
```

### Get Dataset Validation Status

Checks the status of a dataset validation job.

**Endpoint:** `GET /dataset_validate_status/{job_id}`

**Parameters:**
- `job_id`: Job ID returned from the dataset validation request

**Response:**

```json
{
  "job_id": "string",
  "process_status": "string",
  "process_stage": "string",
  "error": "string",
  "stack trace": "string"
}
```

### Evaluate Dataset

Evaluates a dataset for potential issues like data poisoning.

**Endpoint:** `POST /dataset_evaluate/`

**Request Body:**

```json
{
  "user_id": "string",
  "dataloader": {
    "definition": {
      "uid": "string",
      "path": "url",
      "class_name": "string"
    }
  },
  "dataset": {
    "uid": "string",
    "path": "url"
  }
}
```

**Response:**

```json
{
  "job_id": "string"
}
```

### Get Dataset Evaluation Status

Checks the status of a dataset evaluation job and retrieves results.

**Endpoint:** `GET /dataset_evaluate_status/{job_id}`

**Parameters:**
- `job_id`: Job ID returned from the dataset evaluation request

**Response:**

```json
{
  "job_id": "string",
  "process_status": "string",
  "process_stage": "string",
  "error": "string",
  "stack trace": "string",
  "report": {},
  "pdf": "base64_encoded_string"
}
```

## Bias Operations

### Validate for Bias

Validates a dataset for potential bias analysis.

**Endpoint:** `POST /bias_validate/`

**Request Body:**

```json
{
  "user_id": "string",
  "dataloader": {
    "definition": {
      "uid": "string",
      "path": "url",
      "class_name": "string"
    }
  },
  "dataset": {
    "uid": "string",
    "path": "url"
  }
}
```

**Response:**

```json
{
  "job_id": "string"
}
```

### Get Bias Validation Status

Checks the status of a bias validation job.

**Endpoint:** `GET /bias_validate_status/{job_id}`

**Parameters:**
- `job_id`: Job ID returned from the bias validation request

**Response:**

```json
{
  "job_id": "string",
  "process_status": "string",
  "process_stage": "string",
  "error": "string",
  "stack trace": "string",
  "compatible_metrics": [
    {
      "name": "string",
      "description": "string"
    }
  ],
  "compatible_mitigations": [
    {
      "name": "string",
      "description": "string"
    }
  ],
  "features": [
    {
      "name": "string",
      "is_categorical": boolean,
      "values": []
    }
  ]
}
```

### Detect Bias

Detects bias in a dataset.

**Endpoint:** `POST /bias_detection/`

**Request Body:**

```json
{
  "user_id": "string",
  "dataloader": {
    "definition": {
      "uid": "string",
      "path": "url",
      "class_name": "string"
    }
  },
  "dataset": {
    "uid": "string",
    "path": "url"
  }
}
```

**Response:**

```json
{
  "job_id": "string"
}
```

### Get Bias Detection Status

Checks the status of a bias detection job and retrieves results.

**Endpoint:** `GET /bias_detection_status/{job_id}`

**Parameters:**
- `job_id`: Job ID returned from the bias detection request

**Response:**

```json
{
  "job_id": "string",
  "process_status": "string",
  "process_stage": "string",
  "error": "string",
  "stack trace": "string",
  "report": {
    "features": [
      {
        "name": "string",
        "metrics": [
          {
            "name": "string",
            "value": float
          }
        ]
      }
    ]
  }
}
```

### Mitigate Bias

Applies bias mitigation techniques to a dataset.

**Endpoint:** `POST /bias_mitigation/`

**Request Body:**

```json
{
  "user_id": "string",
  "dataloader": {
    "definition": {
      "uid": "string",
      "path": "url",
      "class_name": "string"
    }
  },
  "dataset": {
    "uid": "string",
    "path": "url"
  },
  "priv_features": [
    {
      "name": "string",
      "value": "string"
    }
  ],
  "mitigations": [
    {
      "name": "string",
      "description": "string"
    }
  ],
  "download_url": "string"
}
```

**Response:**

```json
{
  "job_id": "string"
}
```

### Get Bias Mitigation Status

Checks the status of a bias mitigation job and retrieves results.

**Endpoint:** `GET /bias_mitigation_status/{job_id}`

**Parameters:**
- `job_id`: Job ID returned from the bias mitigation request

**Response:**

```json
{
  "job_id": "string",
  "process_status": "string",
  "process_stage": "string",
  "error": "string",
  "stack trace": "string",
  "report": {},
  "pdf": "base64_encoded_string"
}
```

## Job Control

### Stop Job

Stops a running job.

**Endpoint:** `POST /stop/`

**Request Body:**

```json
{
  "job_id": "string"
}
```

**Response:**

```json
{
  "stoppage_status": "string"
}
```

## Error Responses

All endpoints may return error responses in case of validation failures or server errors:

### Validation Error (422)

```json
{
  "detail": [
    {
      "loc": ["string", 0],
      "msg": "string",
      "type": "string"
    }
  ]
}
```

### Not Found Error (404)

```json
{
  "detail": "Job ID not found"
}
```

## Implementation Details

The AutoDefenseML API is implemented using FastAPI, which provides automatic validation of request and response schemas. The backend uses multiple worker processes to handle long-running tasks asynchronously, with status endpoints to check progress.

Key technologies used:
- FastAPI for API framework
- Google Cloud Firestore for job status and results storage
- Google Cloud PubSub for distributed task processing
- Adversarial Robustness Toolbox (ART) for attack and defense implementations
- PyTorch, TensorFlow, and scikit-learn for ML model support
