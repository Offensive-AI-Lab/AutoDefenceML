##### ONLY if needed (if its not already a service inside kubernetes) - Creation of cloud build that deploying the frontend app to firebase #####

# resource "google_project_iam_member" "cloud_build_firebase_hosting_admin" {
#   project = var.project_id
#   role    = "roles/firebasehosting.admin"
#   member  = "serviceAccount:${google_service_account.cloud_build.email}"
# }

# # Firebase Admin 
# resource "google_project_iam_member" "firebase_admin" {
#   project = var.project_id
#   role    = "roles/firebase.admin"
#   member  = "serviceAccount:${google_service_account.cloud_build.email}"
# }

# resource "google_cloudbuild_trigger" "deploy_to_firebase" {
#   service_account = google_service_account.cloud_build.id
#   provider = google-beta
#   name = "deploy-to-firebase"
#   description = "Deploy React app to Firebase Hosting"
#   filename = "${var.react_cloudbuild_yaml_folder}/cloudbuild.yaml" 

#   github {
#     owner = "kaleidoo-ai"
#     name = "Kal-Sudio-Front"
#     push {
#       branch = "^${var.branch}"
#     }
#   }

#   substitutions = {
#     _SERVICE_NAME         = "frontend"
#     _API_GATEWAY_URL      = var.api_gateway_domain
#     _GCP_PROJECT_ID       = var.project_id
#     _APP_ID               = var.app_id
#     _API_KEY              = var.api_key
#     _AUTH_DOMAIN          = var.auth_domain
#     _DATABASE_URL         = var.database_url
#     _STORAGE_BUCKET       = var.storage_bucket
#     _MESSAGING_SENDER_ID  = var.messaging_sender_id
#     _MEASUREMENT_ID       = var.measurement_id
#   }

#     provisioner "local-exec" {
#         command = "gcloud beta builds triggers run deploy-to-firebase --branch=release"
#     }
# }
