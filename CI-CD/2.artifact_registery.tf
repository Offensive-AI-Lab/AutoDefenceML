##### Create new repository in ___ project for the services - for example #####
resource "google_artifact_registry_repository" "repo" {
  project = var.project_id # The ID of the project in which the repository will be created - DO NOT EDIT
  location      = var.region # The location for the repository
  repository_id = var.repository_id # The ID of the repository
  format        = "DOCKER" # Specify DOCKER for Docker images
  description = "Artifact Registry for the Docker images"
  labels = {
    env = "dev"
  }
}