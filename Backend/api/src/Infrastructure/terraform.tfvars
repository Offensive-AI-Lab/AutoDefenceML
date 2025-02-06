project_id                    = "autodefenseml"
region                        = "us-central1"
zone                          = "us-central1-a"
vpc_name                      = "bgu-vpc"
subnet_cidr                   = "10.0.0.0/21"
secondary_subnet_cidr         = "10.4.0.0/14"
secondary_service_subnet_cidr = "10.8.0.0/20"
cidr_block                    = "172.16.0.0/28"
repo_name                     = "autodefenseml"
workload_identity_pool_id     = "bgu-wli-pool"

services_names = {
  "api_terraform": {
    "deployment"    = "app-deployment"
    "service"       = "api-terraform-service"
    "app"           = "app"
    "trigger"       = "api-terraform-trigger"
    "name"          = "app"
    "env" = [
      {"name": "api_terraform_PORT", "value": "8080"},
      {"name": "api_terraform_HOST", "value": "0.0.0.0"},
      {"name": "USER_MNG_URL", "value": "http://user-manager:80"},
      {"name": "SYSTEM_SERVICE_URL", "value": "http://system-service:80"},
      {"name": "GOOGLE_CLOUD_PROJECT", "value": "autodefenseml"},
      {"name": "PROJECT_MNG_URL", "value": "http://project-manager:80"},
      {"name": "CONFIG_MNG_URL", "value": "http://configuration-manager:80"},
      {"name": "VALIDATION_MNG_URL", "value": "http://validation-manager:80"},
      {"name": "EVAL_DETECTION_MNG_URL", "value": "http://eval-detection-manager:80"},
      {"name": "FIREBASE_API_KEY", "value": "AIzaSyDEOZpCAcrjpJI5MaKB4wECdkjw8V5_bxU"},
      {"name": "ENVIRONMENT", "value": "MainServer"}
    ]
  }
  "api_terraform_gpu": {
    "deployment"    = "app-deployment-gpu"
    "service"       = "api-terraform-service-gpu"
    "app"           = "app-gpu"
    "trigger"       = "api-terraform-trigger-gpu"
    "name"          = "app-gpu"
    "env" = []
  }
}