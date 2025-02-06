
# Enables Firebase services for the new project created above.
resource "google_firebase_project" "default" {
  provider = google-beta
  project  = var.project_id
}

resource "google_firebase_web_app" "default" {
  provider     = google-beta
  project      = var.project_id
  display_name = var.project_id
  depends_on = [
    google_firebase_project.default,
  ]
}

data "google_firebase_web_app_config" "basic" {
  provider   = google-beta
  web_app_id = google_firebase_web_app.default.app_id

  depends_on = [google_firebase_web_app.default]
}

resource "google_storage_bucket" "default" {
  provider = google-beta
  name     = "${var.project_id}-fb-webapp"
  location = "us-central1"

  depends_on = [google_firebase_web_app.default]
}
