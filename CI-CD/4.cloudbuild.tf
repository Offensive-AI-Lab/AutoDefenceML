# google_cloudbuild_trigger Resource
# for_each = var.services_names
# for_each: Iterates over the var.services_names map, creating a google_cloudbuild_trigger resource for each item.
# service_account
# service_account = google_service_account.cloud_build.id: Specifies the service account to be used by Cloud Build triggers.
# name, description, project, filename
# name: The name of the trigger, derived from each.value.trigger_name.
# description: A description of the trigger, including the image name.
# project: The project ID in which the trigger will be created.
# filename: The path to the cloudbuild.yaml file, specific to each service.
# github Block
# Configures the GitHub repository and branch to monitor for changes:

# owner: The GitHub repository owner.
# name: The GitHub repository name.
# push: The branch to trigger the build on push events, specified by var.branch.
# substitutions
# Defines substitution variables for the Cloud Build configuration:

# _REGION: The region where resources will be deployed.
# _PROJECT_ID: The project ID.
# _REPOSITORY: The repository ID (same as project ID in this case).
# _IMAGE_NAME: The image name for the Docker image.
# _SERVICE_DIR: The directory for the service within the backend.
# provisioner "local-exec"
# Executes a local command after the trigger is created:

# command: Runs a gcloud command to trigger the Cloud Build using the specified branch.
# depends_on
# Specifies dependencies to ensure these resources are created before the trigger:

# google_artifact_registry_repository.repo: The Artifact Registry repository.
# google_service_account.cloud_build: The service account for Cloud Build.
# Summary
# This Terraform configuration creates Cloud Build triggers for multiple services defined in var.services_names. 
# Each trigger monitors a specific branch in a GitHub repository and runs a cloudbuild.yaml file located in a specific directory for each service. 
# It uses a designated service account and sets up necessary substitutions for the build process. 
# The local-exec provisioner runs a command to manually trigger the build after the trigger is created.
resource "google_cloudbuild_trigger" "backend_triggers" {
  for_each = var.services_names
  service_account = google_service_account.cloud_build.id
  name        = each.value.trigger_name
  description = "Trigger to build and deploy ${each.value.image_name} using cloudbuild.yaml"
  project     = var.project_id
  filename    = "Backend/${each.value.manager_dir}/cloudbuild.yaml"
  # included_files = ["${each.value.manager_dir}/**"]
  github {
    owner = var.git_owner
    name  = var.repo_name
    push {
      branch = "^${var.branch}$"
    }
  }
  substitutions = {
    _REGION            = var.region
    _PROJECT_ID        = var.project_id
    _REPOSITORY        = var.repository_id
    _IMAGE_NAME           = each.value.image_name
    _SERVICE_DIR       = "Backend/${each.value.manager_dir}"
  }
    provisioner "local-exec" {
    command = "gcloud beta builds triggers run ${each.value.trigger_name} --branch=${var.branch}"
  }
  depends_on = [google_artifact_registry_repository.repo, google_service_account.cloud_build]
}