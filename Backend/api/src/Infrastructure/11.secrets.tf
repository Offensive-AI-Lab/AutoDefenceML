resource "kubernetes_secret" "google_application_credentials" {
  metadata {
    name      = "google-application-credentials"
    namespace = "default"
  }

  data = {
    GOOGLE_APPLICATION_CREDENTIALS = base64encode(file("/Library/validation/src/user_files/autodefenseml-dbf24d33eec1.json"))
  }
}

resource "kubernetes_secret" "firebase_credentials" {
  metadata {
    name      = "firebase-credentials"
    namespace = "default"
  }

  data = {
    FIREBASE_CREDENTIALS = base64encode(file("/Library/validation/src/user_files/autodefenseml-dbf24d33eec1.json"))
  }
}
