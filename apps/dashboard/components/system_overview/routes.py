"""
System Overview Routes
CRITICAL: These routes preserve EXACT API paths from monolithic system
DO NOT change paths - components expect these exact endpoints
"""

from flask import Blueprint, jsonify, render_template
from .service import SystemOverviewService

# Create blueprint for system overview
system_overview_bp = Blueprint(
    'system_overview', 
    __name__,
    template_folder='templates',
    static_folder='static',
    static_url_path='/system_overview/static'
)

# Initialize service
service = SystemOverviewService()

@system_overview_bp.route('/components/system_overview')
def render_component():
    """Render the system overview component HTML"""
    return render_template('system_overview.html')

@system_overview_bp.route('/api/system/metrics')
def api_system_metrics():
    """Get system performance metrics
    
    CRITICAL: This MUST preserve the exact API path /api/system/metrics
    Components expect this exact endpoint and response format
    """
    # Get metrics from service (preserving exact logic)
    metrics = service.get_system_metrics()
    
    return jsonify({
        'total_requests': metrics['total_requests'],
        'total_errors': metrics['total_errors'],
        'error_rate': metrics['error_rate'],
        'healthy_services': metrics['healthy_services'],
        'total_services': metrics['total_services'],
        'debug_images_count': metrics['debug_images_count'],
        'logs_count': metrics['logs_count']
    })

@system_overview_bp.route('/api/services/status')
def api_services_status():
    """Get all services status
    
    CRITICAL: This MUST preserve the exact API path /api/services/status
    System Overview component calls this endpoint for service counts
    """
    # Get services status from service (preserving exact logic)
    services_data = service.get_services_status()
    
    return jsonify(services_data)

def init_system_overview(app):
    """Initialize system overview component with Flask app"""
    # Register blueprint
    app.register_blueprint(system_overview_bp, url_prefix='')
    
    # No additional initialization needed - uses global monitoring system
    return system_overview_bp