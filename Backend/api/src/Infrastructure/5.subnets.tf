resource "google_compute_subnetwork" "private" {
  name                     = "${var.vpc_name}-private"
  ip_cidr_range            = var.subnet_cidr
  region                   = var.region
  network                  = google_compute_network.main.id
  private_ip_google_access = true

  secondary_ip_range {
    range_name    = "pod-secondary-range"
    ip_cidr_range = var.secondary_subnet_cidr
  }

  secondary_ip_range {
    range_name    = "service-secondary-range"
    ip_cidr_range = var.secondary_service_subnet_cidr
  }
}
