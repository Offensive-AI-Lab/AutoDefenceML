##### Configure the node-pools for the cluster #####

# https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/container_node_pool
resource "google_container_node_pool" "sc-main" {
  name       = "sc-main"
  cluster    = google_container_cluster.cluster.id
  node_count = 1

  management {
    auto_repair  = true
    auto_upgrade = true
  }

  node_config {
    preemptible  = false
    machine_type = "n2-standard-4"

    labels = {
      role = "general"
    }

    service_account = google_service_account.kubernetes.email
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
    tags = ["gke-node"]
  }
}

resource "google_container_node_pool" "node-spot" {
  name    = "sc-nodes"
  cluster = google_container_cluster.cluster.id

  management {
    auto_repair  = true
    auto_upgrade = true
  }

  autoscaling {
    min_node_count = 0
    max_node_count = 10
  }

  node_config {
    preemptible  = true
    machine_type = "n2-standard-4"

    service_account = google_service_account.kubernetes.email
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
    tags = ["gke-node"]
  }
}

# resource "google_container_node_pool" "node-gpu" {
#   name    = "sc-gpu"
#   cluster = google_container_cluster.cluster.id
#   # node_count = 1
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
#       "nodepool" = "sc-gpu"
#       "gpu"      = "true"
#     }

#     # Add a startup script to install GPU drivers
#     metadata = {
#       "install-gpu-driver" = "true"
#     }
#   }
# }

