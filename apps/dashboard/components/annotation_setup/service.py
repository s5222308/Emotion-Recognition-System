"""
Annotation Setup Service
CRITICAL: This preserves EXACT business logic from monolithic system
All Label Studio integration and setup logic must match original implementation
"""

import subprocess
import requests
import os
from pathlib import Path


class AnnotationSetupService:
    """Service for Annotation Setup component
    
    CRITICAL: This handles Label Studio integration exactly as the original
    DO NOT create new patterns - preserve existing Label Studio interaction
    """
    
    def __init__(self):
        # Configuration for Label Studio and ML Backend
        self.label_studio_url = 'http://localhost:8200'
        self.ml_backend_url = 'http://localhost:9091'
        self.ml_service_url = 'http://localhost:5003'
        
    def get_setup_status(self):
        """Get complete annotation environment setup status
        
        CRITICAL: This MUST produce the EXACT same data structure as the original
        Original API endpoint: /api/labelstudio/check-setup
        """
        # Check all components
        label_studio_status = self._check_label_studio_status()
        ml_backend_status = self._check_ml_backend_status()
        ml_service_status = self._check_ml_service_status()
        
        # Check if projects exist (if Label Studio is running)
        projects_count = 0
        if label_studio_status['status'] == 'running':
            try:
                # This would need authentication in full implementation
                projects_count = 0  # Placeholder - requires auth to get real count
            except:
                pass
        
        # Determine overall readiness
        ready_to_annotate = (
            label_studio_status['status'] == 'running' and
            ml_backend_status['status'] == 'running' and
            ml_service_status['status'] == 'running'
        )
        
        # Match old API structure exactly
        return {
            'label_studio': {
                'installed': label_studio_status['status'] != 'not_installed',
                'version': label_studio_status.get('version', 'unknown'),
                'status': label_studio_status['status'],
                'url': label_studio_status.get('url', self.label_studio_url),
                'projects_count': projects_count
            },
            'ml_backend': {
                'connected': ml_backend_status['status'] == 'running',
                'status': 'running' if ml_backend_status['status'] == 'running' else 'not_running',
                'message': ml_backend_status.get('message', 'ML Backend status'),
                'url': ml_backend_status.get('url', self.ml_backend_url)
            },
            'ml_service': ml_service_status,
            'setup_complete': ready_to_annotate,
            'recommended_action': self._get_recommended_action(
                label_studio_status['status'], 
                label_studio_status['status'] != 'not_installed',
                ml_backend_status['status'] == 'running'
            ),
            'ready_to_annotate': {
                'status': 'ready' if ready_to_annotate else 'not_ready',
                'message': 'Ready to start annotation' if ready_to_annotate else 'Some services are not running'
            }
        }
    
    def _check_label_studio_status(self):
        """Check Label Studio installation and running status"""
        try:
            # First check if Label Studio is installed
            result = subprocess.run(['which', 'label-studio'], 
                                  capture_output=True, text=True, timeout=3)
            if result.returncode != 0:
                return {
                    'status': 'not_installed',
                    'message': 'Label Studio not installed',
                    'installation_path': None
                }
            
            installation_path = result.stdout.strip()
            
            # Check if service is running
            try:
                response = requests.get(f'{self.label_studio_url}/api/version', timeout=3)
                if response.status_code == 200:
                    version_data = response.json()
                    return {
                        'status': 'running',
                        'message': 'Label Studio is running',
                        'installation_path': installation_path,
                        'version': version_data.get('version', 'unknown'),
                        'url': self.label_studio_url
                    }
                else:
                    return {
                        'status': 'installed_not_running',
                        'message': f'Label Studio installed but not running (HTTP {response.status_code})',
                        'installation_path': installation_path
                    }
            except requests.exceptions.RequestException as e:
                return {
                    'status': 'installed_not_running',
                    'message': f'Label Studio installed but not responding: {str(e)}',
                    'installation_path': installation_path
                }
                
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            return {
                'status': 'unknown',
                'message': f'Could not check Label Studio status: {str(e)}',
                'installation_path': None
            }
    
    def _check_ml_backend_status(self):
        """Check ML Backend (Label Studio Connector) status"""
        try:
            response = requests.get(f'{self.ml_backend_url}/health', timeout=3)
            if response.status_code == 200:
                return {
                    'status': 'running',
                    'message': 'ML Backend is running',
                    'url': self.ml_backend_url
                }
            else:
                return {
                    'status': 'not_running',
                    'message': f'ML Backend not responding (HTTP {response.status_code})',
                    'url': self.ml_backend_url
                }
        except requests.exceptions.RequestException as e:
            return {
                'status': 'not_running',
                'message': f'ML Backend not accessible: {str(e)}',
                'url': self.ml_backend_url
            }
    
    def _check_ml_service_status(self):
        """Check ML Service (Visual Inference) status"""
        try:
            response = requests.get(f'{self.ml_service_url}/health', timeout=3)
            if response.status_code == 200:
                return {
                    'status': 'running',
                    'message': 'ML Service is running',
                    'url': self.ml_service_url
                }
            else:
                return {
                    'status': 'not_running',
                    'message': f'ML Service not responding (HTTP {response.status_code})',
                    'url': self.ml_service_url
                }
        except requests.exceptions.RequestException as e:
            return {
                'status': 'not_running',
                'message': f'ML Service not accessible: {str(e)}',
                'url': self.ml_service_url
            }
    
    
    def create_label_studio_project(self, project_name, email, password, storage_path, auth_response):
        """Create Label Studio project for existing authenticated user
        
        CRITICAL: This replicates the exact logic from the old setup wizard
        """
        import time
        import re
        from pathlib import Path
        
        try:
            # Get token from auth response
            token_data = auth_response.json()
            access_token = token_data.get('access_token')
            
            if not access_token:
                return {'error': 'No access token received'}
            
            # Load label config
            config_path = Path(__file__).parent.parent.parent.parent.parent / 'config' / 'label_studio.xml'
            with open(config_path, 'r') as f:
                label_config = f.read()
            
            # Create project data
            project_data = {
                'title': project_name,
                'description': 'Automated emotion detection project via dashboard setup',
                'label_config': label_config,
                'is_published': True,
                'enable_empty_annotation': True,
                'show_skip_button': True,
                'maximum_annotations': 1,
                'show_annotation_history': True,
                'show_predictions_to_annotators': True,
                'auto_prediction': True
            }
            
            # Create project using session if available, otherwise bearer token
            if hasattr(auth_response, '_web_session') and auth_response._web_session:
                create_response = auth_response._web_session.post(
                    f'{self.label_studio_url}/api/projects/',
                    json=project_data,
                    timeout=10
                )
            else:
                headers = {
                    'Authorization': f'Bearer {access_token}',
                    'Content-Type': 'application/json'
                }
                create_response = requests.post(
                    f'{self.label_studio_url}/api/projects/',
                    json=project_data,
                    headers=headers,
                    timeout=10
                )
            
            if create_response.status_code not in [200, 201]:
                error_msg = create_response.text[:200] if create_response.text else 'Unknown error'
                return {'error': f'Project creation failed: {error_msg}'}
            
            project = create_response.json()
            project_id = project['id']
            
            # Configure storage and ML backend
            result = self._configure_project_full_setup(
                project_id, storage_path, access_token, auth_response
            )
            
            if result.get('success'):
                return {
                    'success': True,
                    'message': f'Project "{project_name}" created successfully',
                    'project_id': project_id,
                    'project_name': project_name,
                    'url': f'{self.label_studio_url}/projects/{project_id}',
                    'access_token': access_token,
                    'ml_backend_connected': result.get('ml_connected', False),
                    'storage_configured': result.get('storage_configured', False),
                    'video_files_found': result.get('video_files_found', 0)
                }
            else:
                return {
                    'success': True,
                    'message': f'Project "{project_name}" created but setup incomplete',
                    'project_id': project_id,
                    'project_name': project_name,
                    'url': f'{self.label_studio_url}/projects/{project_id}',
                    'access_token': access_token,
                    'error': result.get('error')
                }
                
        except Exception as e:
            return {'error': f'Project creation failed: {str(e)}'}
    
    def create_label_studio_user_and_project(self, project_name, email, password, storage_path):
        """Create new Label Studio user and project
        
        CRITICAL: This replicates the user creation and project setup from old wizard
        """
        import time
        import re
        
        try:
            # Get CSRF token from signup page
            session = requests.Session()
            csrf_response = session.get(f'{self.label_studio_url}/user/signup/')
            csrf_token = None
            if 'csrfmiddlewaretoken' in csrf_response.text:
                match = re.search(r'name="csrfmiddlewaretoken" value="([^"]*)"', csrf_response.text)
                if match:
                    csrf_token = match.group(1)
            
            if not csrf_token:
                return {'error': 'Could not get CSRF token for user creation'}
            
            # Create user account
            signup_data = {
                'email': email,
                'password': password,
                'how_find_us': 'Other',
                'allow_newsletters': 'false',
                'csrfmiddlewaretoken': csrf_token
            }
            
            signup_response = session.post(
                f'{self.label_studio_url}/user/signup/',
                data=signup_data,
                timeout=5,
                allow_redirects=False
            )
            
            if signup_response.status_code not in [200, 201, 302]:
                return {'error': f'Account creation failed (status: {signup_response.status_code})'}
            
            # Login with new account
            login_csrf_response = session.get(f'{self.label_studio_url}/user/login/')
            login_csrf_token = None
            if 'csrfmiddlewaretoken' in login_csrf_response.text:
                match = re.search(r'name="csrfmiddlewaretoken" value="([^"]*)"', login_csrf_response.text)
                if match:
                    login_csrf_token = match.group(1)
            
            login_data = {
                'email': email,
                'password': password,
                'csrfmiddlewaretoken': login_csrf_token
            }
            
            login_response = session.post(
                f'{self.label_studio_url}/user/login/',
                data=login_data,
                timeout=5,
                allow_redirects=False
            )
            
            if login_response.status_code not in [200, 302]:
                return {'error': f'Login failed after account creation (status: {login_response.status_code})'}
            
            # Create mock auth_response to reuse project creation logic
            from unittest.mock import Mock
            auth_response = Mock()
            auth_response.status_code = 200
            auth_response.json = lambda: {
                'access_token': 'web_session_auth',
                'user': {'email': email}
            }
            auth_response._web_session = session
            
            # Use existing project creation method
            return self.create_label_studio_project(
                project_name, email, password, storage_path, auth_response
            )
            
        except Exception as e:
            return {'error': f'User and project creation failed: {str(e)}'}
    
    def _configure_project_full_setup(self, project_id, storage_path, access_token, auth_response):
        """Configure storage and ML backend for project
        
        CRITICAL: This handles the complete project configuration from old wizard
        """
        import os
        from pathlib import Path
        
        try:
            result = {
                'success': True,
                'ml_connected': False,
                'storage_configured': False,
                'video_files_found': 0
            }
            
            # Validate and configure storage
            if not os.path.exists(storage_path):
                return {'success': False, 'error': f'Storage path does not exist: {storage_path}'}
            
            if not os.path.isdir(storage_path):
                return {'success': False, 'error': f'Storage path is not a directory: {storage_path}'}
            
            # Find video files
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v']
            video_files = []
            for root, dirs, files in os.walk(storage_path):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in video_extensions):
                        video_files.append(os.path.join(root, file))
            
            result['video_files_found'] = len(video_files)
            
            # Configure Label Studio storage
            storage_config = {
                "project": project_id,
                "type": "localfiles",
                "path": storage_path,
                "regex_filter": ".*\\.(mp4|avi|mov|mkv|webm|m4v)$",
                "use_blob_urls": True,
                "title": "Local Video Storage",
                "description": f"Videos from {storage_path}",
                "presign": True,
                "presign_ttl": 3600,
                "recursive_scan": True,
                "sync_enabled": True
            }
            
            # Add storage using session if available
            if hasattr(auth_response, '_web_session') and auth_response._web_session:
                storage_response = auth_response._web_session.post(
                    f'{self.label_studio_url}/api/storages/localfiles/',
                    json=storage_config,
                    timeout=10
                )
            else:
                headers = {'Authorization': f'Bearer {access_token}'}
                storage_response = requests.post(
                    f'{self.label_studio_url}/api/storages/localfiles/',
                    json=storage_config,
                    headers=headers,
                    timeout=10
                )
            
            if storage_response.status_code in [200, 201]:
                result['storage_configured'] = True
                storage_data = storage_response.json()
                storage_id = storage_data.get('id')
                
                # Trigger initial sync
                if storage_id:
                    try:
                        if hasattr(auth_response, '_web_session') and auth_response._web_session:
                            sync_response = auth_response._web_session.post(
                                f'{self.label_studio_url}/api/storages/localfiles/{storage_id}/sync',
                                timeout=30
                            )
                        else:
                            headers = {'Authorization': f'Bearer {access_token}'}
                            sync_response = requests.post(
                                f'{self.label_studio_url}/api/storages/localfiles/{storage_id}/sync',
                                headers=headers,
                                timeout=30
                            )
                    except Exception as sync_error:
                        print(f"[WARNING] Storage sync failed: {sync_error}")
            
            # Configure ML Backend
            ml_backend_config = {
                "project": project_id,
                "url": self.ml_backend_url,
                "title": "Emotion Recognition ML Backend",
                "description": "Multi-modal emotion recognition for clinical analysis",
                "use_ground_truth": False,
                "model_version": "latest",
                "is_interactive": False
            }
            
            if hasattr(auth_response, '_web_session') and auth_response._web_session:
                ml_response = auth_response._web_session.post(
                    f'{self.label_studio_url}/api/ml/',
                    json=ml_backend_config,
                    timeout=10
                )
            else:
                headers = {'Authorization': f'Bearer {access_token}'}
                ml_response = requests.post(
                    f'{self.label_studio_url}/api/ml/',
                    json=ml_backend_config,
                    headers=headers,
                    timeout=10
                )
            
            if ml_response.status_code in [200, 201]:
                result['ml_connected'] = True
                
                # Configure Data Manager view
                self._configure_data_manager_view(project_id, auth_response, access_token)
            
            return result
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _configure_data_manager_view(self, project_id, auth_response, access_token):
        """Configure Data Manager view with custom columns
        
        CRITICAL: Matches the view configuration from old wizard
        """
        try:
            view_config = {
                "data": {
                    "title": "Emotion Detection Workflow",
                    "type": "list", 
                    "target": "tasks",
                    "ordering": ["tasks:id"],
                    "selectedItems": {"all": False, "included": []},
                    "hiddenColumns": {
                        "explore": [
                            "tasks:id", "tasks:file_upload", "tasks:created_at", "tasks:updated_at", 
                            "tasks:updated_by", "tasks:avg_lead_time", "tasks:draft_exists", 
                            "tasks:annotations_ids"
                        ],
                        "labeling": [
                            "tasks:id", "tasks:file_upload", "tasks:created_at", "tasks:updated_at",
                            "tasks:updated_by", "tasks:avg_lead_time", "tasks:draft_exists",
                            "tasks:annotations_ids"
                        ]
                    },
                    "columnsWidth": {
                        "tasks:inner_id": 80,
                        "tasks:completed_at": 120,
                        "tasks:total_annotations": 150,
                        "tasks:cancelled_annotations": 120,
                        "tasks:total_predictions": 150,
                        "tasks:annotators": 150,
                        "tasks:annotations_results": 250,
                        "tasks:predictions_score": 120,
                        "tasks:predictions_model_versions": 180,
                        "tasks:predictions_results": 250,
                        "tasks:storage_filename": 200,
                        "video": 300
                    },
                    "columnsDisplayType": {},
                    "gridWidth": 4,
                    "gridFitImagesToWidth": False,
                    "semantic_search": [],
                    "filters": {"conjunction": "and", "items": []}
                },
                "project": project_id
            }
            
            if hasattr(auth_response, '_web_session') and auth_response._web_session:
                auth_response._web_session.post(
                    f'{self.label_studio_url}/api/dm/views',
                    json=view_config,
                    timeout=10
                )
            else:
                headers = {'Authorization': f'Bearer {access_token}'}
                requests.post(
                    f'{self.label_studio_url}/api/dm/views',
                    json=view_config,
                    headers=headers,
                    timeout=10
                )
        except Exception:
            # View configuration is optional, don't fail the whole setup
            pass
    
    def _get_recommended_action(self, ls_status, ls_installed, ml_connected):
        """Determine the recommended action based on system status
        
        CRITICAL: This matches the exact logic from old api_routes.py.bak
        """
        if ls_status == 'not_installed':
            return {
                'action': 'install',
                'message': 'Label Studio is not installed. Please install it first.',
                'button_text': 'Install Label Studio',
                'button_class': 'btn-warning'
            }
        elif ls_status == 'installed_not_running':
            return {
                'action': 'start',
                'message': 'Label Studio is installed but not running.',
                'button_text': 'Start Label Studio',
                'button_class': 'btn-primary'
            }
        elif ls_status == 'running' and not ml_connected:
            return {
                'action': 'connect_ml',
                'message': 'Label Studio is running but ML backend is disconnected.',
                'button_text': 'Connect ML Backend',
                'button_class': 'btn-info'
            }
        elif ls_status == 'running' and ml_connected:
            return {
                'action': 'configure_project',
                'message': 'System is ready. You can create a new project or configure existing ones.',
                'button_text': 'Configure Project',
                'button_class': 'btn-success'
            }
        else:
            return {
                'action': 'full_setup',
                'message': 'Full system setup is required.',
                'button_text': 'Run Full Setup',
                'button_class': 'btn-warning'
            }