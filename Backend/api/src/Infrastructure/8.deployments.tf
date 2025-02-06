resource "kubernetes_deployment" "api_terraform" {
  metadata {
    name = var.services_names["api_terraform"].deployment
  }

  spec {
    replicas = 1

    selector {
      match_labels = {
        app = var.services_names["api_terraform"].app
      }
    }

    template {
      metadata {
        labels = {
          app = var.services_names["api_terraform"].app
        }
      }

      spec {
        service_account_name = kubernetes_service_account.k8s-service.metadata[0].name
        container {
          name  = var.services_names["api_terraform"].name
          image = "${var.region}-docker.pkg.dev/${var.project_id}/${var.repo_name}/${var.services_names["api_terraform"].name}:latest"
          port {
            container_port = 8080
          }

          # env {
          #   name = "GOOGLE_APPLICATION_CREDENTIALS"
          #   value_from {
          #     secret_key_ref {
          #       name = kubernetes_secret.google_application_credentials.metadata[0].name
          #       key  = "GOOGLE_APPLICATION_CREDENTIALS"
          #     }
          #   }
          # }

          dynamic "env" {
            for_each = var.services_names["api_terraform"].env
            content {
              name  = env.value.name
              value = env.value.value
            }
          }
        }
      }
    }
  }

  depends_on = [
    kubernetes_secret.google_application_credentials,
    kubernetes_secret.firebase_credentials
  ]
}


# resource "kubernetes_deployment" "api_terraform_gpu" {
#   metadata {
#     name = var.services_names["api_terraform_gpu"].deployment
#   }

#   spec {
#     replicas = 1

#     selector {
#       match_labels = {
#         app = var.services_names["api_terraform_gpu"].app
#       }
#     }

#     template {
#       metadata {
#         labels = {
#           app = var.services_names["api_terraform_gpu"].app
#         }
#       }

#       spec {
#         service_account_name = kubernetes_service_account.k8s-service.metadata[0].name

#         node_selector = {
#           "nodepool" = "sc-gpu",
#           "gpu"      = "true"
#         }

#         toleration {
#           key      = "nvidia.com/gpu"
#           operator = "Equal"
#           value    = "present"
#           effect   = "NoSchedule"
#         }

#         container {
#           name = var.services_names["api_terraform_gpu"].name
#           # image = "${var.region}-docker.pkg.dev/${var.project_id}/${var.repo_name}/${var.services_names["api_terraform"].app}:latest"
#           image = "${var.region}-docker.pkg.dev/${var.project_id}/${var.repo_name}/${var.services_names["api_terraform"].app}:29e1190"
#           port {
#             container_port = 8080
#           }

#           dynamic "env" {
#             for_each = var.services_names["api_terraform"].env
#             content {
#               name  = env.value.name
#               value = env.value.value
#             }
#           }
#         }
#       }
#     }
#   }

#   depends_on = [
#     kubernetes_secret.google_application_credentials,
#     kubernetes_secret.firebase_credentials
#   ]
# }
