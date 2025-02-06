# This file is used to configure the backend for terraform state
# The terraform state will be stored in the storage bucket
terraform {
 backend "gcs" {
    bucket  = "cbg-api-bucket"
    prefix  = "Infrastructure/state"
 }
}