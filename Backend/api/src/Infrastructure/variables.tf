variable "project_id" {
  description = "The Google Cloud project ID"
  type        = string
}

variable "region" {
  description = "The region where the resources will be created"
  type        = string
}

variable "zone" {
  description = "The zone within the region where resources will be created"
  type        = string
}

variable "vpc_name" {
  description = "The name of the VPC"
  type        = string
}

variable "subnet_cidr" {
  description = "The CIDR block for the subnet"
  type        = string
}

variable "secondary_subnet_cidr" {
  description = "The secondary CIDR block for the subnet"
  type        = string
}

variable "secondary_service_subnet_cidr" {
  description = "The secondary CIDR block for the subnet"
  type        = string
}

variable "cidr_block" {
  description = "The CIDR block for the private cluster"
  type        = string
}

variable "repo_name" {
  description = "The name of the repository"
  type        = string
}

variable "services_names" {
  description = "A map of service names to their configurations"
  type = map(object({
    deployment = string
    service    = string
    trigger    = string
    app        = string
    name       = string
    env = list(object({
      name  = string
      value = string
    }))
  }))
}

variable "workload_identity_pool_id" {
  description = "The ID for the workload identity pool"
  type        = string
}
