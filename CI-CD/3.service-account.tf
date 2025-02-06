
##### Creating a new service account for Cloud Build #####
# Enabling Cloud Build API
# service: Specifies the service to enable, in this case, the Cloud Build API.
# project: The project ID where the service will be enabled.
# disable_on_destroy: Keeps the service enabled even if the Terraform resource is destroyed.
resource "google_project_service" "cloudbuild" {
  service = "cloudbuild.googleapis.com"
  project = var.project_id
  disable_on_destroy = false
}

# Creating a Service Account
# account_id: The ID of the service account.
# display_name: A display name for the service account.
# project: The project ID where the service account will be created.
resource "google_service_account" "cloud_build" {
  account_id   = "cloud-build-account-test"
  display_name = "Cloud Build Account"
  project      = var.project_id
  
}

# Assigning IAM Roles
# Each google_project_iam_member and google_project_iam_binding block assigns an IAM role to the newly created service account.
# role: Assigns the roles/container.developer role, allowing the service account to manage GKE resources.
# member: The member to whom the role is assigned, in this case, the service account created earlier.
resource "google_project_iam_member" "cloud_build_gke" {
  project = var.project_id
  role    = "roles/container.developer"
  member  = "serviceAccount:${google_service_account.cloud_build.email}"  
}
# role: Assigns the roles/logging.logWriter role, allowing the service account to write logs.
resource "google_project_iam_member" "cloud_build_logs_writer" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.cloud_build.email}"  
}
# role: Assigns the roles/iam.serviceAccountUser role, allowing the service account to act as other service accounts.
resource "google_project_iam_member" "cloud_build_service_user" {
  project = var.project_id
  role    = "roles/iam.serviceAccountUser"
  member  = "serviceAccount:${google_service_account.cloud_build.email}"
}

# Roles for Artifact Registry
# role: Assigns the roles/artifactregistry.writer role, allowing the service account to write to the Artifact Registry.
resource "google_project_iam_binding" "artifact_registry_writer" {
  project = var.project_id
  role    = "roles/artifactregistry.writer"
  members  = ["serviceAccount:${google_service_account.cloud_build.email}"]
}
# role: Assigns the roles/artifactregistry.admin role, allowing the service account full control over the Artifact Registry.
resource "google_project_iam_binding" "artifact_registry_admin" {
  project = var.project_id
  role    = "roles/artifactregistry.admin"
  members = ["serviceAccount:${google_service_account.cloud_build.email}"]
}

# Roles for Cloud Storage
# role: Assigns the roles/storage.objectAdmin role, allowing the service account to manage Cloud Storage objects.
resource "google_project_iam_member" "cloud_build_storage_admin" {
  project = var.project_id
  role    = "roles/storage.objectAdmin"
  member  = "serviceAccount:${google_service_account.cloud_build.email}"
}
# role: Assigns the roles/storage.objectViewer role, allowing the service account to view Cloud Storage objects.
resource "google_project_iam_member" "cloud_build_storage_object_viewer" {
  project = var.project_id
  role    = "roles/storage.objectViewer"
  member  = "serviceAccount:${google_service_account.cloud_build.email}"
}

# Roles for Firebase
# role: Assigns the roles/firebasehosting.admin role, allowing the service account to manage Firebase Hosting.
resource "google_project_iam_member" "cloud_build_firebase_hosting_admin" {
  project = var.project_id
  role    = "roles/firebasehosting.admin"
  member  = "serviceAccount:${google_service_account.cloud_build.email}"
}
# role: Assigns the roles/firebase.admin role, allowing the service account to manage Firebase resources.
resource "google_project_iam_member" "firebase_admin" {
  project = var.project_id
  role    = "roles/firebase.admin"
  member  = "serviceAccount:${google_service_account.cloud_build.email}"
}