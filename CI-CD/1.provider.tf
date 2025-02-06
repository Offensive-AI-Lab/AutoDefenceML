# This block specifies the required providers for your Terraform configuration. In this case:
# google: Specifies the Google Cloud provider.
# source: Indicates where the provider can be sourced from (hashicorp/google).
# version: Specifies the version of the provider to use (5.14.0).
terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "5.14.0"
    }
  }
}

# Main Google Provider
# This block configures the google provider with the following variables:
# project: The Google Cloud project ID (var.project_id).
# region: The region to operate in (var.region).
# zone: The zone to operate in (var.zone).
provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

# Google Beta Provider with User Project Override
# This block configures the google-beta provider. 
# The google-beta provider is used to access Google Cloud features that are in beta. 
# Here, the user_project_override is set to true, 
# which can be used to override the default behavior related to project and billing configurations in some advanced use cases.
provider "google-beta" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
  user_project_override = true
} 

# Google Beta Provider without User Project Override
# This block configures another instance of the google-beta provider but with an alias (no_user_project_override). 
# The alias allows you to reference this specific provider configuration in your Terraform resources. 
# Here, the user_project_override is set to false.
provider "google-beta" {
  alias = "no_user_project_override"
  user_project_override = false
}