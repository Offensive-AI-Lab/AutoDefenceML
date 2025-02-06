resource "kubernetes_service" "api_terraform" {
  metadata {
    name = var.services_names["api_terraform"].service
  }

  spec {
    selector = {
      app = var.services_names["api_terraform"].app
    }

    port {
      port        = 80
      target_port = 8080
    }

    type = "LoadBalancer"
  }
}
# resource "kubernetes_service" "api_terraform_gpu" {
#   metadata {
#     name = var.services_names["api_terraform_gpu"].service
#   }

#   spec {
#     selector = {
#       app = var.services_names["api_terraform_gpu"].app
#     }

#     port {
#       port        = 80
#       target_port = 8080
#     }

#     type = "LoadBalancer"
#   }
# }
