variable "services_names" {
  type = map(object({
    image_name   = string
    trigger_name = string
    manager_dir  = string
  }))
}

variable "repo_name" {
  type = string
}
variable "git_owner" {
  type = string
}

variable "project_id" {
  type = string
}

variable "repository_id" {
  type = string
}

variable "branch" {
  type = string
}

variable "region" {
  type    = string
  default = "us-central1"
}

variable "zone" {
  type    = string
  default = "us-central1-a"
}
