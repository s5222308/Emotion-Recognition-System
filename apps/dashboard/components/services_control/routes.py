"""
Services Control API Routes
CRITICAL: This preserves EXACT API endpoints and permission system from monolithic system
All route paths, decorators, and response formats must match original implementation
"""

from flask import Blueprint, jsonify, request
from .service import ServicesControlService
from core import service_metrics  # Access to global service metrics

# Create blueprint with static file handling and templates
services_control_bp = Blueprint('services_control', __name__, 
                               template_folder='templates',
                               static_folder='static', 
                               static_url_path='/services_control/static')

# Service instance
service = ServicesControlService()

# CSRF instance will be set by init function
csrf = None


@services_control_bp.route('/api/services/start/<service_name>', methods=['POST'])
def api_start_service(service_name):
    """Start a service via web interface
    
    CRITICAL: This MUST match the EXACT endpoint from original app.py
    Path: /api/services/start/<service_name> (not /api/services/<name>/start)
    """
    try:
        print(f"[DEBUG] API request to start service: {service_name}")
        
        result = service.start_service(service_name)
        
        # Add to global service metrics for monitoring
        service_metrics[service_name]['status'] = 'starting'
        
        print(f"[DEBUG] Start service result: {result}")
        return jsonify(result)
        
    except ValueError as e:
        print(f"[ERROR] Invalid service name: {e}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        print(f"[ERROR] Failed to start {service_name}: {e}")
        return jsonify({'error': str(e)}), 500


@services_control_bp.route('/api/services/stop/<service_name>', methods=['POST'])
# CRITICAL: CSRF exempt on stop endpoint - will be applied in init function
def api_stop_service(service_name):
    """Stop a service via web interface
    
    CRITICAL: This MUST match the EXACT endpoint from original app.py
    Path: /api/services/stop/<service_name>
    Note: Original has @csrf.exempt on this endpoint specifically
    """
    try:
        print(f"[DEBUG] API request to stop service: {service_name}")
        
        result = service.stop_service(service_name)
        
        # Update global service metrics for monitoring
        service_metrics[service_name]['status'] = 'stopping'
        
        print(f"[DEBUG] Stop service result: {result}")
        return jsonify(result)
        
    except ValueError as e:
        print(f"[ERROR] Invalid service name: {e}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        print(f"[ERROR] Failed to stop {service_name}: {e}")
        return jsonify({'error': str(e)}), 500


@services_control_bp.route('/api/services/restart/<service_name>', methods=['POST'])
def api_restart_service(service_name):
    """Restart a service via web interface
    
    CRITICAL: This MUST match the EXACT endpoint from original app.py
    Path: /api/services/restart/<service_name>
    """
    try:
        print(f"[DEBUG] API request to restart service: {service_name}")
        
        result = service.restart_service(service_name)
        
        # Update global service metrics for monitoring
        service_metrics[service_name]['status'] = 'restarting'
        
        print(f"[DEBUG] Restart service result: {result}")
        return jsonify(result)
        
    except ValueError as e:
        print(f"[ERROR] Invalid service name: {e}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        print(f"[ERROR] Failed to restart {service_name}: {e}")
        return jsonify({'error': str(e)}), 500


@services_control_bp.route('/api/services/<service_name>/health', methods=['GET'])
def api_service_health(service_name):
    """Check service health status
    
    Used by frontend for health check polling after service operations
    """
    try:
        result = service.check_service_health(service_name)
        return jsonify(result)
        
    except Exception as e:
        print(f"[ERROR] Health check failed for {service_name}: {e}")
        return jsonify({'healthy': False, 'error': str(e)}), 500


def init_services_control(app):
    """Initialize services control component with Flask app
    
    CRITICAL: This follows the EXACT same pattern as other successful components
    """
    print("[DEBUG] Initializing Services Control component...")
    
    # Get CSRF protect instance from app and apply exemption to stop endpoint
    csrf_protect = app.extensions.get('csrf')
    if csrf_protect:
        # Apply CSRF exemption to the stop endpoint - matches original exactly
        csrf_protect.exempt(api_stop_service)
        print("[DEBUG] Applied CSRF exemption to stop service endpoint")
    
    # Register the blueprint with the main app
    app.register_blueprint(services_control_bp)
    
    print("[DEBUG] Services Control component initialized successfully")
    return services_control_bp