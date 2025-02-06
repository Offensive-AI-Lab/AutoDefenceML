##### Configure all the service accounts #####

resource "google_service_account" "kubernetes" {
  account_id = "kubernetes-test"
}

##### Grant the necessary roles, edit as needed #####

resource "google_project_iam_binding" "gcr_viewer" {
  project = var.project_id
  role    = "roles/storage.objectViewer"

  members = [
    "serviceAccount:${google_service_account.kubernetes.email}",
  ]
}

resource "google_project_iam_binding" "gcr_bucket_create" {
  project = var.project_id
  role    = "roles/storage.admin"

  members = [
    "serviceAccount:${google_service_account.kubernetes.email}",
  ]
}

resource "google_project_iam_binding" "artifact_registry_reader" {
  project = var.project_id
  role    = "roles/artifactregistry.reader"

  members = [
    "serviceAccount:${google_service_account.kubernetes.email}",
  ]
}

resource "google_project_iam_member" "token_creator" {
  project = var.project_id
  role    = "roles/iam.serviceAccountTokenCreator"
  member  = "serviceAccount:${google_service_account.kubernetes.email}"
}