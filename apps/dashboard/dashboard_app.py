"""
Modular Emotion Recognition Dashboard
Clean, maintainable Flask application
"""
import os
import sys

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Import modular components
from config.settings import DashboardConfig
from services.monitoring import SystemMonitor
from routes.main_routes import main_bp

# Import System Overview component
from components.system_overview import system_overview_bp, init_system_overview

# Import System Logs component  
from components.system_logs import system_logs_bp, init_system_logs

# Import Debug Images component
from components.debug_images import debug_images_bp, init_debug_images

# Import Annotation Setup component
from components.annotation_setup import annotation_setup_bp, init_annotation_setup

# Import Services Control component
from components.services_control import services_control_bp, init_services_control

# Import Processing Queue component
from components.processing_queue import processing_queue_bp, ProcessingQueueService, init_processing_queue

# Import System Hardware component
from components.system_hardware import system_hardware_bp, init_system_hardware
from components.model_configuration import model_configuration_bp, init_model_configuration

class DashboardApp:
    """Main dashboard application class"""
    
    def __init__(self):
        self.app = None
        self.monitor = None
        
    def create_app(self):
        """Create and configure Flask application"""
        self.app = Flask(__name__)
        
        # Load configuration
        self.app.config.from_object(DashboardConfig)
        
        # Initialize extensions
        # No CSRF protection for localhost research environment
        Limiter(
            key_func=get_remote_address,
            app=self.app,
            default_limits=["100 per minute"],
            storage_uri="memory://"
        )
        
        # Initialize monitoring
        self.monitor = SystemMonitor()
        
        # Initialize components
        init_system_overview(self.app)
        init_system_logs(self.app)
        init_debug_images(self.app)
        init_annotation_setup(self.app)
        init_services_control(self.app)
        init_system_hardware(self.app)
        init_model_configuration(self.app)
        init_processing_queue(self.app)
        
        # Register main blueprint
        self.app.register_blueprint(main_bp)
        # Component blueprints are registered via init_* functions
        
        # Set template and static folders
        # Templates are in the apps/dashboard/templates subdirectory
        template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
        static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
        
        self.app.template_folder = template_dir
        self.app.static_folder = static_dir
        
        return self.app
    
    def run(self):
        """Start the dashboard application"""
        # Create directories
        os.makedirs('../../web/templates', exist_ok=True)
        os.makedirs('../../web/static/css', exist_ok=True)
        os.makedirs('../../web/static/js', exist_ok=True)
        
        # Start monitoring
        self.monitor.start()
        self.monitor._add_log('INFO', 'Modular dashboard started')
        
        # Display startup info
        print("Emotion Recognition Dashboard")
        print("Starting on: http://localhost:8081")
        print("Links:")
        print("   - Dashboard:     http://localhost:8081")
        print("   - Label Studio:  http://localhost:8200")
        print("   - ML Service:    http://localhost:5003/health")
        print("   - ML Backend:    http://localhost:9091/health")
        print("Architecture: Component-based routes")
        
        # Run the application
        self.app.run(host='0.0.0.0', port=8081, debug=False)

def main():
    """Main entry point"""
    dashboard = DashboardApp()
    app = dashboard.create_app()
    dashboard.run()

if __name__ == '__main__':
    main()