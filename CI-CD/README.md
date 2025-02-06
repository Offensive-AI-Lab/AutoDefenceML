# CI/CD Pipeline Setup using Terraform and Google Cloud Build

This guide provides instructions on how to set up a continuous integration and deployment (CI/CD) pipeline using Terraform and Google Cloud Build for deploying services in Google Cloud. The setup automates the building of Docker images and updating Kubernetes deployments.

## Prerequisites

Before you begin, ensure you have the following installed:
- **Google Cloud SDK**: Follow the [Google Cloud SDK Installation Guide](https://cloud.google.com/sdk/docs/install).
- **Terraform**: Install Terraform by following the [official Terraform installation guide](https://learn.hashicorp.com/tutorials/terraform/install-cli).

## Configuration Files Overview

- `backend.tf`: Specifies where to store the state file.
- `provider.tf`: Configures the Google Cloud provider settings.
- `service-account.tf`: Sets up the service account and IAM roles for Cloud Build.
- `cloudbuild.tf`: Defines the Cloud Build trigger for automating the build and deployment processes.
- `variables.tf`: Declares variables used across Terraform configurations.
- `terraform.tfvars`: Specifies the actual values for the declared variables, tailored to your project.

## Steps to Deploy

1. **Configure Terraform Variables and the Backend File**:
   - Edit `terraform.tfvars` to include your project-specific settings such as `project_id`, `region`, `zone`, and details about each service you want to deploy (e.g., service name, Dockerfile directory).
   - Configure `backend.tf` to define the bucket name for storing the state file, as well as the directory within that bucket. Ensure you assign a unique directory name for each environment to prevent conflicts and maintain clear separation of state files.

2. **Initialize Terraform**:
   - Open a terminal in the directory containing your Terraform files.
   - Run the following command to initialize Terraform and download the necessary providers:
     ```bash
     terraform init
     ```

3. **Review the Terraform Plan**:
   - Execute the following command to see what Terraform plans to do:
     ```bash
     terraform plan
     ```

4. **Apply the Terraform Configuration**:
   - Apply the configurations to set up the CI/CD pipeline:
     ```bash
     terraform apply
     ```
   - Confirm the action when prompted to proceed with the changes.

5. **Manage Cloud Build Triggers**:
   - Once Terraform successfully creates the resources, Cloud Build triggers are set to automate builds upon code pushes.
   - If needed, triggers can be manually invoked using Google Cloud SDK commands detailed in the `cloudbuild.tf`.

## Customizing the Pipeline

- **Modify Cloud Build Steps**:
  - Update the `cloudbuild.yaml` in each service directory to customize build steps, Docker tagging, or deployment strategies.
  
- **Adjust IAM Permissions**:
  - If different permissions are needed, modify the IAM roles and bindings in `service-account.tf`.

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

By following this guide, you can set up a fully automated CI/CD pipeline that builds Docker images and updates Kubernetes deployments based on your project needs. Adjust the provided Terraform configurations and `cloudbuild.yaml` to tailor the setup to your specific requirements.
