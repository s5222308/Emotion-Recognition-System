"""
Main page routes for dashboard
"""
from flask import Blueprint, render_template, session, redirect, send_from_directory
# No auth for localhost research
from config.settings import DashboardConfig
from datetime import datetime
import os

# Create main blueprint
main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def dashboard():
    """Main dashboard page"""
    
    # Label Studio project info
    label_studio_project = {
        'project_name': 'Workspace',
        'project_url': 'http://localhost:8200'
    }
    
    # Services configuration - use centralized config
    services = DashboardConfig.SERVICES
    
    return render_template('dashboard_modular.html', 
                         services=services,
                         label_studio_project=label_studio_project,
                         current_time=datetime.now())

@main_bp.route('/debug-images/<filename>')
def serve_debug_image(filename):
    """Serve debug images"""
    debug_images_dir = '/home/lleyt/WIL_project/emotion_labelstudio_final/services/ml_engine/debug_images'
    if os.path.exists(os.path.join(debug_images_dir, filename)):
        return send_from_directory(debug_images_dir, filename)
    else:
        return "Image not found", 404