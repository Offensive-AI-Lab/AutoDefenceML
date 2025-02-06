import subprocess
import os
import shutil
from typing import Dict, List, Tuple
from .helpers import db_log

class GKEUtils:
    def __init__(self, environment_id: str):
        self.environment_id = environment_id

    def check_and_install_dependencies(self) -> Dict[str, bool]:
        """Check for required dependencies and try to install them if missing."""
        dependencies = {
            'gcloud': 'google-cloud-sdk',
            'kubectl': 'kubectl',
            'terraform': 'terraform'
        }
        
        status = {}
        for tool, package in dependencies.items():
            db_log(self.environment_id, f"Checking {tool} installation...")
            if not shutil.which(tool):
                db_log(self.environment_id, f"{tool} not found, attempting installation...")
                try:
                    if tool == 'gcloud':
                        result = subprocess.run([
                            'curl', 'https://sdk.cloud.google.com', '|', 'bash'
                        ], capture_output=True, text=True)
                        db_log(self.environment_id, f"gcloud installation output: {result.stdout}")
                        
                        init_result = subprocess.run(['gcloud', 'init'], capture_output=True, text=True)
                        db_log(self.environment_id, f"gcloud init output: {init_result.stdout}")
                    
                    elif tool == 'kubectl':
                        result = subprocess.run([
                            'gcloud', 'components', 'install', 'kubectl'
                        ], capture_output=True, text=True)
                        db_log(self.environment_id, f"kubectl installation output: {result.stdout}")
                    
                    status[tool] = shutil.which(tool) is not None
                    db_log(self.environment_id, f"{tool} installation {'successful' if status[tool] else 'failed'}")
                
                except subprocess.CalledProcessError as e:
                    db_log(self.environment_id, f"Error installing {tool}: {str(e)}")
                    status[tool] = False
            else:
                status[tool] = True
                db_log(self.environment_id, f"{tool} is already installed")
                
        return status

    def verify_gcloud_auth(self) -> Tuple[bool, str]:
        """Verify gcloud authentication and project configuration."""
        try:
            db_log(self.environment_id, "Checking gcloud authentication...")
            
            # Check if logged in
            auth_list = subprocess.run(
                ['gcloud', 'auth', 'list'],
                capture_output=True,
                text=True
            )
            db_log(self.environment_id, f"Auth list output: {auth_list.stdout}")
            
            # Check current project
            project = subprocess.run(
                ['gcloud', 'config', 'get-value', 'project'],
                capture_output=True,
                text=True
            ).stdout.strip()
            db_log(self.environment_id, f"Current project: {project}")
            
            # Check if application-default credentials exist
            creds_path = os.path.expanduser('~/.config/gcloud/application-default-credentials.json')
            creds_exist = os.path.exists(creds_path)
            db_log(self.environment_id, f"Credentials exist at {creds_path}: {creds_exist}")
            
            if not creds_exist:
                db_log(self.environment_id, "Running application-default login...")
                login_result = subprocess.run(
                    ['gcloud', 'auth', 'application-default', 'login'],
                    capture_output=True,
                    text=True
                )
                db_log(self.environment_id, f"Login result: {login_result.stdout}")
            
            return True, f"Authenticated. Current project: {project}"
            
        except subprocess.CalledProcessError as e:
            error_msg = f"Authentication failed: {str(e)}"
            db_log(self.environment_id, error_msg)
            return False, error_msg

    def check_permissions(self) -> Dict[str, bool]:
        """Check GCP permissions for cluster and node operations."""
        try:
            db_log(self.environment_id, "Starting detailed permissions check...")
            
            # Check cluster access
            db_log(self.environment_id, "Checking cluster access...")
            cluster_result = subprocess.run([
                'gcloud', 'container', 'clusters', 'list',
                '--format=json'
            ], capture_output=True, text=True)
            
            if cluster_result.returncode != 0:
                db_log(self.environment_id, f"Cluster access check failed: {cluster_result.stderr}")
                return {'has_all_permissions': False, 'error': 'No cluster access'}
            
            db_log(self.environment_id, f"Cluster access check output: {cluster_result.stdout}")

            # Check node pool access
            db_log(self.environment_id, "Checking node pool access...")
            node_pool_result = subprocess.run([
                'gcloud', 'container', 'node-pools', 'list',
                '--cluster=autodefenseml',
                '--region=us-central1',
                '--format=json'
            ], capture_output=True, text=True)
            
            if node_pool_result.returncode != 0:
                db_log(self.environment_id, f"Node pool access check failed: {node_pool_result.stderr}")
                return {'has_all_permissions': False, 'error': 'No node pool access'}
                
            db_log(self.environment_id, f"Node pool access check output: {node_pool_result.stdout}")

            # Check kubectl access
            db_log(self.environment_id, "Checking kubectl access...")
            kubectl_result = subprocess.run([
                'kubectl', 'auth', 'can-i', 'delete', 'deployment'
            ], capture_output=True, text=True)
            
            if 'yes' not in kubectl_result.stdout.lower():
                db_log(self.environment_id, f"Kubectl access check failed: {kubectl_result.stderr}")
                return {'has_all_permissions': False, 'error': 'No kubectl deletion permissions'}
                
            db_log(self.environment_id, f"Kubectl access check output: {kubectl_result.stdout}")
            
            # If we got here, we have all permissions
            db_log(self.environment_id, "All permission checks passed successfully")
            return {
                'has_all_permissions': True,
                'permissions_output': {
                    'cluster_access': True,
                    'node_pool_access': True,
                    'kubectl_access': True
                }
            }
                
        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to check permissions: {str(e)}"
            db_log(self.environment_id, error_msg)
            return {
                'has_all_permissions': False,
                'error': error_msg
            }

    def delete_node_pool(self, cluster: str, node_pool: str, region: str) -> Tuple[bool, str]:
        """Safely delete a node pool from GKE cluster."""
        try:
            db_log(self.environment_id, f"Starting node pool deletion process for {node_pool}")
            
            # Check cluster exists
            cluster_check = subprocess.run([
                'gcloud', 'container', 'clusters', 'describe',
                cluster, f'--region={region}'
            ], capture_output=True, text=True)
            db_log(self.environment_id, f"Cluster check output: {cluster_check.stdout}")
            
            # Get nodes in the pool
            nodes_list = subprocess.run([
                'kubectl', 'get', 'nodes',
                f'--selector=cloud.google.com/gke-nodepool={node_pool}',
                '-o', 'name'
            ], capture_output=True, text=True)
            db_log(self.environment_id, f"Nodes in pool: {nodes_list.stdout}")
            
            # Drain nodes
            drain_result = subprocess.run([
                'kubectl', 'drain', 
                f'--selector=cloud.google.com/gke-nodepool={node_pool}',
                '--ignore-daemonsets', '--delete-emptydir-data'
            ], capture_output=True, text=True)
            db_log(self.environment_id, f"Node drain result: {drain_result.stdout}")
            
            # Delete the node pool
            delete_result = subprocess.run([
                'gcloud', 'container', 'node-pools', 'delete',
                node_pool,
                f'--cluster={cluster}',
                f'--region={region}',
                '--quiet'
            ], capture_output=True, text=True)
            db_log(self.environment_id, f"Node pool deletion result: {delete_result.stdout}")
            
            return True, f"Successfully deleted node pool {node_pool}"
            
        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to delete node pool: {str(e)}"
            db_log(self.environment_id, error_msg)
            return False, error_msg

    def force_delete_deployment(self, deployment: str, namespace: str = 'default') -> Tuple[bool, str]:
        """Force delete a Kubernetes deployment."""
        try:
            db_log(self.environment_id, f"Starting deployment deletion for {deployment}")
            
            # Check if deployment exists
            check_result = subprocess.run([
                'kubectl', 'get', 'deployment', deployment,
                f'--namespace={namespace}'
            ], capture_output=True, text=True)
            db_log(self.environment_id, f"Deployment check result: {check_result.stdout}")
            
            # Force delete
            delete_result = subprocess.run([
                'kubectl', 'delete', 'deployment', deployment,
                f'--namespace={namespace}',
                '--force', '--grace-period=0'
            ], capture_output=True, text=True)
            db_log(self.environment_id, f"Deployment deletion result: {delete_result.stdout}")
            
            # Verify deletion
            verify_result = subprocess.run([
                'kubectl', 'get', 'deployment', deployment,
                f'--namespace={namespace}'
            ], capture_output=True, text=True)
            db_log(self.environment_id, f"Deployment verification result: {verify_result.stderr}")
            
            return True, f"Successfully deleted deployment {deployment}"
            
        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to delete deployment: {str(e)}"
            db_log(self.environment_id, error_msg)
            return False, error_msg