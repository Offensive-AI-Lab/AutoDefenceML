resource "google_compute_global_address" "static_ip" {
  name         = "cbg-static-ip"
  address_type = "EXTERNAL"
  project      = var.project_id
}

resource "google_compute_global_address" "ingress_gateway" {
  name = "ingress-gateway"
}

resource "kubernetes_ingress_v1" "api_gateway_ingress" {
  metadata {
    name = "api-gateway-ingress"
    annotations = {
      "kubernetes.io/ingress.global-static-ip-name" = google_compute_global_address.static_ip.name
      "networking.gke.io/managed-certificates"      = kubernetes_manifest.managed_certificate.manifest.metadata.name # Uncomment for Step 2
    }
  }
  spec {
    rule {
      host = "autodefenseml.355.co.il"
      http {
        path {
          path      = "/"
          path_type = "Prefix"
          backend {
            service {
              name = var.services_names["api_terraform"].service
              port {
                number = 80
              }
            }
          }
        }
      }
    }
  }
}


resource "kubernetes_manifest" "managed_certificate" {
  provider = kubernetes

  manifest = {
    apiVersion = "networking.gke.io/v1beta2"
    kind       = "ManagedCertificate"
    metadata = {
      name      = "dev-certificate"
      namespace = "default"
      annotations = {
        force-refresh = "true"
      }
    }
    spec = {
      domains = ["autodefenseml.355.co.il"]
    }
  }
}
