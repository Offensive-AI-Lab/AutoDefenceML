resource "google_compute_network" "main" {
  name                            = "${var.vpc_name}-vpc-test"
  routing_mode                    = "REGIONAL"
  auto_create_subnetworks         = false
  mtu                             = 1460
  delete_default_routes_on_create = false

  depends_on = [
    google_project_service.compute,
    google_project_service.container
  ]
}

resource "google_compute_router" "router" {
  name    = "${var.vpc_name}-router"
  region  = "us-central1"
  network = google_compute_network.main.name

  depends_on = [
    google_compute_network.main
  ]
}

resource "google_compute_router_nat" "nat" {
  name                               = "${var.vpc_name}-nat"
  router                             = google_compute_router.router.name
  region                             = google_compute_router.router.region
  nat_ip_allocate_option             = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"

  depends_on = [
    google_compute_router.router
  ]
}
