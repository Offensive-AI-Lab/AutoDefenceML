##### Configure all the firewalls configuration - edit to your own requirments #####

resource "google_compute_firewall" "allow-ssh" {
  name    = "allow-ssh"
  network = google_compute_network.main.name

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  source_ranges = ["0.0.0.0/0"]

  log_config {
    metadata = "INCLUDE_ALL_METADATA"
  }
}

resource "google_compute_firewall" "allow-https" {
  name    = "allow-https"
  network = google_compute_network.main.name

  allow {
    protocol = "tcp"
    ports    = ["443"]
  }
  allow {
    protocol = "tcp"
    ports    = ["80"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["gke-node","goog-gke-node","gke-kal-studio-dev-7b6dc352-node"]



  log_config {
    metadata = "INCLUDE_ALL_METADATA"
  }
}
resource "google_compute_firewall" "allow-http" {
  name    = "allow-http"
  network = google_compute_network.main.name

  allow {
    protocol = "tcp"
    ports    = ["80"]
  }

  source_ranges = ["0.0.0.0/0"]

  log_config {
    metadata = "INCLUDE_ALL_METADATA"
  }
}

resource "google_compute_firewall" "allow-imap-ssl-egress" {
  name                    = "allow-imap-ssl-egress"
  network                 = google_compute_network.main.name
  direction               = "EGRESS"
  destination_ranges      = ["0.0.0.0/0"]  # Allows access to all external IPs
  priority                = 1000

  allow {
    protocol              = "tcp"
    ports                 = ["993"]
  }
}


