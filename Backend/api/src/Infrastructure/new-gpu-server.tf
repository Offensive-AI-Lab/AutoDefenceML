# variable "gpu_num" {
#   description = "The ID for the duplicate gpu"
#   type        = string
#   default     = "1"
# }

# resource "kubernetes_deployment" "api_terraform_gpu_num" {
#   metadata {
#     name = "app-deployment-gpu${var.gpu_num}"
#   }

#   spec {
#     replicas = 1

#     selector {
#       match_labels = {
#         app = "app-gpu${var.gpu_num}"
#       }
#     }

#     template {
#       metadata {
#         labels = {
#           app = "app-gpu${var.gpu_num}"
#         }
#       }

#       spec {
#         service_account_name = kubernetes_service_account.k8s-service.metadata[0].name

#         node_selector = {
#           "nodepool" = "sc-gpu${var.gpu_num}",
#           "gpu"      = "true${var.gpu_num}"
#         }

#         toleration {
#           key      = "nvidia.com/gpu"
#           operator = "Equal"
#           value    = "present"
#           effect   = "NoSchedule"
#         }

#         container {
#           name  = "app-gpu${var.gpu_num}"
#           image = "${var.region}-docker.pkg.dev/${var.project_id}/${var.repo_name}/${var.services_names["api_terraform"].app}:latest"
#           # image = "me-west1-docker.pkg.dev/autodefenseml/autodefenseml/app:29e1190"
#           # image = "me-west1-docker.pkg.dev/autodefenseml/autodefenseml/app:bd7205b"
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

# resource "google_container_node_pool" "node-gpu_num" {
#   name               = "sc-gpu${var.gpu_num}"
#   cluster            = google_container_cluster.cluster.id
#   initial_node_count = 1 # Ensure the pool starts with at least 1 node


#   management {
#     auto_repair  = true
#     auto_upgrade = true
#   }

#   autoscaling {
#     min_node_count = 1
#     max_node_count = 5
#   }

#   node_config {
#     preemptible  = false
#     machine_type = "n1-standard-4" # Adjust the machine type as needed

#     guest_accelerator {
#       count = 1                 # Number of GPUs per node
#       type  = "nvidia-tesla-t4" # Replace with the appropriate GPU type
#     }

#     # Optional taint configuration
#     taint {
#       key    = "nvidia.com/gpu"
#       value  = "present"
#       effect = "NO_SCHEDULE"
#     }

#     service_account = google_service_account.kubernetes.email
#     oauth_scopes = [
#       "https://www.googleapis.com/auth/cloud-platform"
#     ]
#     tags = ["gke-node"]

#     labels = {
#       "role"     = "compute"
#       "nodepool" = "sc-gpu${var.gpu_num}"
#       "gpu"      = "true${var.gpu_num}"
#     }

#     # Add a startup script to install GPU drivers
#     metadata = {
#       "install-gpu-driver"       = "true"
#       "disable-legacy-endpoints" = "true"
#     }
#   }
# }

# resource "kubernetes_service" "api_terraform_gpu_num" {
#   metadata {
#     name = "api-terraform-service-gpu${var.gpu_num}"
#     annotations = {
#       "cloud.google.com/neg" = jsonencode({
#         ingress = true
#       })
#     }
#   }

#   spec {
#     selector = {
#       app = "app-gpu${var.gpu_num}"
#     }

#     port {
#       port        = 80
#       target_port = 8080
#     }

#     type = "LoadBalancer"
#   }
# }

# # output service_ip {
# #     value = kubernetes_service.api_terraform_gpu_num.spec.load_balancer_ip
# # }
# # os.getenv("ENV_NAME")
