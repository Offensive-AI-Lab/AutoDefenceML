project_id    = "autodefenseml"
branch        = "master"
region        = "us-central1"
repo_name     = "api"
git_owner     = "MABADATA"
repository_id = "autodefenseml"

services_names = {
  "terraform" = {
    trigger_name  = "api-terraform-trigger"
    manager_dir   = "api"
    image_name    = "app"
  }
}
