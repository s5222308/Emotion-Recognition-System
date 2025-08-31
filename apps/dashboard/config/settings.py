"""
Dashboard configuration settings
"""
import os
from datetime import timedelta

class DashboardConfig:
    """Centralized configuration for dashboard"""
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY', 'your-secret-key-here-change-in-production')
    PERMANENT_SESSION_LIFETIME = timedelta(hours=8)
    SESSION_TIMEOUT_WARNING = timedelta(minutes=10)
    
    # Security settings
    WTF_CSRF_TIME_LIMIT = None
    
    # Rate limiting
    RATELIMIT_STORAGE_URL = "memory://"
    RATELIMIT_DEFAULT = "100 per minute"
    
    # Service configuration
    SERVICES = {
        'ml_service': {
            'name': 'ML Visual Inference Service',
            'description': 'Emotion Recognition',
            'port': 5003,
            'health_url': 'http://localhost:5003/health',
            'start_cmd': ['python', 'app.py'],
            'cwd': '/home/lleyt/WIL_project/emotion_labelstudio_final/services/ml_engine',
            # Additional fields for service control
            'url': 'http://localhost:5003/status',
            'base_url': 'http://localhost:5003', 
            'health_endpoint': '/health',
            'shutdown_url': 'http://localhost:5003/shutdown',
            'process_name': 'ml_engine'
        },
        'ml_backend': {
            'name': 'ML Backend',
            'description': 'Label Studio Connector',
            'port': 9091,
            'health_url': 'http://localhost:9091/health', 
            'start_cmd': ['python', 'services/label_studio_connector/app.py'],
            'cwd': '/home/lleyt/WIL_project/emotion_labelstudio_final',
            # Additional fields for service control
            'url': 'http://localhost:9091/status',
            'base_url': 'http://localhost:9091',
            'health_endpoint': '/health', 
            'shutdown_url': 'http://localhost:9091/shutdown',
            'process_name': 'label_studio_connector/app.py'
        },
        'label_studio': {
            'name': 'Label Studio',
            'description': 'Annotation Interface', 
            'port': 8200,
            'health_url': 'http://localhost:8200/',
            'start_cmd': ['label-studio', 'start', '--host', '0.0.0.0', '--port', '8200'],
            'cwd': '/home/lleyt/WIL_project/emotion_labelstudio_final',
            # Additional fields for service control
            'url': 'http://localhost:8200',
            'base_url': 'http://localhost:8200',
            'health_endpoint': '/',
            'shutdown_url': 'http://localhost:8200',
            'process_name': 'label-studio'
        }
    }
    
    # User authentication - matches monolithic version exactly
    import hashlib
    CLINICAL_USERS = {
        'clinical_admin': {
            'password_hash': hashlib.sha256(os.environ.get('CLINICAL_ADMIN_PASSWORD', 'admin').encode()).hexdigest(),
            'role': 'administrator',
            'permissions': {
                'control_services': True,
                'view_logs': True,
                'manage_users': True,
                'export_data': True
            }
        },
        'clinical_researcher': {
            'password_hash': hashlib.sha256(os.environ.get('CLINICAL_RESEARCHER_PASSWORD', 'researcher').encode()).hexdigest(),
            'role': 'researcher', 
            'permissions': {
                'control_services': False,
                'view_logs': True,
                'manage_users': False,
                'export_data': True
            }
        }
    }
    
    # Monitoring settings - OPTIMIZED for 1-3 users (60% less requests)
    POLLING_INTERVALS = {
        'services': 5000,        # Service status polling - Keep frequent (critical for UI)
        'logs': 10000,           # System logs - 8s → 10s (reasonable)
        'metrics': 30000,        # System metrics - 16s → 30s (less critical)
        'hardware': 30000,       # Hardware monitoring - NEW: 5s → 30s (changes slowly)
        'debug_images': 30000,   # Debug images - NEW: 12s → 30s (static content)
        'sse_check': 500,        # SSE data checks - Keep fast
    }
    
    # UI settings
    MAX_LOG_ENTRIES = 1000
    MAX_DEBUG_IMAGES = 50
    MAX_PROCESSING_TIMES = 100
    MAX_RECENT_TASKS_DISPLAY = 10
    
    @classmethod
    def get_service_config(cls, service_name):
        """Get configuration for a specific service"""
        return cls.SERVICES.get(service_name, {})
    
    @classmethod
    def get_user_permissions(cls, username):
        """Get user permissions"""
        user = cls.CLINICAL_USERS.get(username, {})
        return user.get('permissions', {})