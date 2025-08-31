"""
Annotation Setup API Routes
CRITICAL: This preserves EXACT API endpoints from monolithic system
All route paths and response formats must match original implementation
"""

from flask import Blueprint, jsonify, request
import os
import time
import subprocess
import requests
import re
from pathlib import Path
from .service import AnnotationSetupService
from core import service_metrics  # Access to monitoring if needed

# Create blueprint with templates and static files
annotation_setup_bp = Blueprint('annotation_setup', __name__, 
                               template_folder='templates',
                               static_folder='static', 
                               static_url_path='/annotation_setup/static')

# Service instance
service = AnnotationSetupService()

@annotation_setup_bp.route('/api/labelstudio/check-setup')
def api_check_labelstudio_setup():
    """Check Label Studio setup status
    
    CRITICAL: This MUST match the EXACT endpoint from original
    Original endpoint: /api/labelstudio/check-setup (dashboard.html line 2337)
    """
    try:
        result = service.get_setup_status()
        return jsonify(result)
    except Exception as e:
        print(f"[ERROR] Annotation Setup API error: {e}")
        return jsonify({'error': str(e)}), 500


@annotation_setup_bp.route('/api/labelstudio/setup', methods=['POST'])
def api_labelstudio_setup():
    """Set up Label Studio project with ML backend - with authentication
    
    CRITICAL: This is the main setup endpoint called by the setup wizard
    Handles project creation, authentication, and ML backend connection
    """
    try:
        print(f"[DEBUG] Setup request headers: {dict(request.headers)}")
        print(f"[DEBUG] Setup request data: {request.json}")
    except Exception as debug_error:
        print(f"[DEBUG] Error printing debug info: {debug_error}")
    
    print(f"[DEBUG] About to start main try block...")
    
    try:
        print(f"[DEBUG] Starting setup process...")
        data = request.json
        print(f"[DEBUG] Successfully parsed JSON data: {data}")
        
        if not data:
            print(f"[DEBUG] No JSON data received!")
            return jsonify({'error': 'No data received'}), 400
            
        project_name = data.get('project_name', 'Emotion Detection Project')
        email = data.get('email', '')
        password = data.get('password', '')
        
        # Use absolute path from current project root
        project_root = Path(__file__).parent.parent.parent.parent.parent
        default_storage_path = str(project_root / 'data' / 'datasets' / 'ravdess')
        storage_path = data.get('storage_path', default_storage_path)
        
        # Convert relative paths to absolute paths for Label Studio compatibility
        if not Path(storage_path).is_absolute():
            storage_path = str(project_root / storage_path)
        
        print(f"[DEBUG] Parsed data - project: {project_name}, email: {email}, storage: {storage_path}")
        
        if not email or not password:
            print(f"[DEBUG] Missing email or password")
            return jsonify({'error': 'Email and password required'}), 400
            
        # Start Label Studio if not running
        print(f"[DEBUG] Checking if Label Studio is running...")
        try:
            response = requests.get('http://localhost:8200/api/version', timeout=3)
            if response.status_code != 200:
                raise Exception("Label Studio not accessible")
            print(f"[DEBUG] Label Studio is running, version check OK")
        except Exception as e:
            print(f"[DEBUG] Label Studio not running: {e}")
            return jsonify({'error': 'Label Studio must be running. Please start it from the Services Control Panel.'}), 400
        
        # Try to authenticate with Label Studio
        print(f"[DEBUG] Attempting to authenticate with Label Studio...")
        try:
            auth_response = requests.post(
                'http://localhost:8200/api/users/login/',
                json={'email': email, 'password': password},
                timeout=5
            )
            print(f"[DEBUG] Login response status: {auth_response.status_code}")
            
            if auth_response.status_code == 200:
                # Authentication successful - create project
                result = service.create_label_studio_project(
                    project_name=project_name,
                    email=email,
                    password=password,
                    storage_path=storage_path,
                    auth_response=auth_response
                )
                return jsonify(result)
                
            elif auth_response.status_code in [400, 401]:
                # User doesn't exist, try to create account
                print(f"[DEBUG] User doesn't exist, attempting to create account...")
                result = service.create_label_studio_user_and_project(
                    project_name=project_name,
                    email=email,
                    password=password,
                    storage_path=storage_path
                )
                return jsonify(result)
            else:
                return jsonify({'error': f'Authentication failed with status {auth_response.status_code}'}), 400
                
        except Exception as auth_error:
            print(f"[DEBUG] Authentication error: {auth_error}")
            return jsonify({'error': f'Authentication failed: {str(auth_error)}'}), 500
            
    except Exception as e:
        print(f"[DEBUG] Main setup error: {e}")
        return jsonify({'error': str(e)}), 500

def init_annotation_setup(app):
    """Initialize annotation setup component with Flask app
    
    CRITICAL: This follows the EXACT same pattern as other successful components
    """
    print("[DEBUG] Initializing Annotation Setup component...")
    
    # Register the blueprint with the main app
    app.register_blueprint(annotation_setup_bp)
    
    print("[DEBUG] Annotation Setup component initialized successfully")
    return annotation_setup_bp