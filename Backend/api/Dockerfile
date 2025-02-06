# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory to /Library/validation
WORKDIR /Library/validation/

RUN mkdir "src"

# Copy the entire project into the container at /Library/validation
COPY src ./src
COPY .env ./src

# Create a temp key file with the service account key.
ENV GOOGLE_APPLICATION_CREDENTIALS=/Library/validation/src/user_files/autodefenseml-dbf24d33eec1.json

RUN pip install --no-cache-dir -U pip
RUN pip install --no-cache-dir -U keyrings.google-artifactregistry-auth
RUN pip install --index-url https://us-central1-python.pkg.dev/autodefenseml/filehandler/simple/ file-loader
RUN pip install --index-url https://us-central1-python.pkg.dev/autodefenseml/art-handler/simple/ art-handler
RUN pip install --index-url https://me-west1-python.pkg.dev/autodefenseml/art-attacks-plugin/simple/ art-attacks-plugin

# Install Rust (required by some dependencies)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Set environment variable to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Update package lists and install required tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    unzip \
    apt-transport-https \
    ca-certificates \
    gnupg

# Install Terraform
RUN curl -O https://releases.hashicorp.com/terraform/1.9.6/terraform_1.9.6_linux_amd64.zip && \
    unzip terraform_1.9.6_linux_amd64.zip && \
    mv terraform /usr/local/bin/ && \
    rm terraform_1.9.6_linux_amd64.zip

# Add Google Cloud SDK distribution URI as a package source
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -

# Install Google Cloud SDK (includes gsutil)
RUN apt-get update && apt-get install -y google-cloud-sdk

# Install kubectl
RUN curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" && \
    chmod +x kubectl && \
    mv kubectl /usr/local/bin/

# Clean up
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python requirements
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 80
EXPOSE 80

# Run the FastAPI application when the container launches
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8080"]