# Kubernetes Infrastructure Setup with Terraform
This README provides detailed instructions on setting up a Kubernetes infrastructure using Terraform within Google Cloud Platform (GCP). This setup includes creating a Kubernetes cluster, associated services, deployments, and optional front-end services not hosted on Firebase.

## Prerequisites

Before you begin, ensure you have the following installed:
- **Google Cloud SDK**: Follow the [Google Cloud SDK Installation Guide](https://cloud.google.com/sdk/docs/install).
- **Terraform**: Install Terraform by following the [official Terraform installation guide](https://learn.hashicorp.com/tutorials/terraform/install-cli).

## Configuration Files Overview

- `backend.tf`: Specifies where to store the state file.
- `provider.tf`: Defines the Terraform provider for Google Cloud.
- `enable-apis.tf`: Enables the necessary Google Cloud services for the project.
- `firebase.tf`: Configures Firebase integration if required.
- `service-account.tf`: Sets up and configures service accounts and their permissions.
- `vpc.tf`: Contains Virtual Private Cloud (VPC) configurations including custom networks and subnets.
- `firewalls.tf`: Defines firewall rules for network security.
- `subnets.tf`: Defines subnets within the VPC for the Kubernetes deployments.
- `kubernetes.tf`: Configures the Kubernetes cluster and related resources.
- `node-pools.tf`: Sets up node pools for the Kubernetes cluster.
- `deployments.tf`: Configures Kubernetes deployments as specified.
- `ingress.tf`: Manages ingress resources and related configurations.
- `services.tf`: Defines Kubernetes services for the deployments.

## Add Deployments and Services:
To add new deployments and services, update the services_names variable in terraform.tfvars with the appropriate service names, deployment configurations, and environment variables. See the example in `terraform.tfvars` file.

## Steps to Deploy

1. **Initialize The Google Cloud Project Environment**:

  To begin, you need to initialize the Google Cloud SDK environment. Follow these steps:

  - Open your terminal or command prompt.
  - Run the following command to initialize the Google Cloud SDK:
  ```bash
  gcloud init
  ```

  - This command will prompt you to log in to your Google account. Follow the instructions to authenticate with your desired Google account.
  - After successful authentication, you'll be prompted to choose a Google Cloud project. Select the project you want to work on from the list, or create a new project if needed.
  - Once the project is selected, the Google Cloud SDK will be configured to work with that project.

#### Important Notes:

  - Ensure that you have the necessary permissions to add a new Artifact Registry to PROJECT NAME project.
  - Additionally, verify that you have access to the Cloud Build service account associated with the PROJECT NAME project.

2. **Configure Terraform Variables and the Backend File**:
   - Edit `terraform.tfvars` to include your project-specific settings such as `project_id`, `region`, `zone`, and details about each service you want to deploy (e.g., service name, deployment name, image location).
   - Configure `backend.tf` to define the bucket name for storing the state file, as well as the directory within that bucket. Ensure you assign a unique directory name for each environment to prevent conflicts and maintain clear separation of state files.


3. **Initialize Terraform**:
- Open a terminal in the directory containing your Terraform files.
- Run the following command to initialize Terraform and download the necessary providers:
    ```bash
    terraform init
    ```
4. **Review the Terraform Plan**:
   - Execute the following command to see what Terraform plans to do:
     ```bash
     terraform plan
     ```
5. **Apply the Terraform Configuration**:
   - Apply the configurations to set up the CI/CD pipeline:
     ```bash
     terraform apply
     ```
   - Confirm the action when prompted to proceed with the changes.
6. **Manage Ingress Configuration**:
    - Initially, deploy the ingress resource with the basic setup by running terraform apply.
    - After confirming the ingress is correctly provisioned, uncomment STEP 2 sections in `ingress.tf` related to SSL certificates or advanced routing, and run terraform apply again to update the ingress settings.

## Troubleshooting
- **Build Failures**:
  - Check Cloud Build logs in the Google Cloud Console for details about build failures.
  - Ensure that the service account has sufficient permissions as configured in `service-account.tf`.

- **Terraform Errors**:
  - Ensure all variables in `terraform.tfvars` are correctly filled out.
  - Re-run `terraform apply` if there are transient errors or after fixing configuration mistakes.

## Additional Resources
- [Google Cloud Build Documentation](https://cloud.google.com/build/docs)
- [Terraform Documentation](https://www.terraform.io/docs)

This guide will help you set up a robust Kubernetes infrastructure tailored to your project's needs, ensuring smooth deployment and management of your services in Google Cloud. Adjustments can be made to fit specific configurations.