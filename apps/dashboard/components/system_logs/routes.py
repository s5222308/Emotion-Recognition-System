"""
System Logs Component Routes
CRITICAL: This preserves EXACT API endpoints and response formats
"""

from flask import Blueprint, request, jsonify, render_template
from .service import SystemLogsService

# Create blueprint for system logs routes
system_logs_bp = Blueprint(
    'system_logs', 
    __name__,
    template_folder='templates',
    static_folder='static',
    static_url_path='/components/system_logs/static'
)

# Initialize service
service = SystemLogsService()

@system_logs_bp.route('/api/logs')
def api_logs():
    """Get system logs with filtering
    
    CRITICAL: This MUST preserve the EXACT same endpoint path and response format
    Original endpoint: /api/logs in main app
    """
    # Get query parameters (same as original)
    level_filter = request.args.get('level', 'ALL')
    limit = int(request.args.get('limit', 50))
    
    # Get logs using service (preserves original logic)
    logs = service.get_logs(level_filter=level_filter, limit=limit)
    
    # Return JSON response (same format as original)
    return jsonify(logs)

@system_logs_bp.route('/components/system_logs/template')
def system_logs_template():
    """Serve the system logs HTML template for inclusion"""
    return render_template('system_logs.html')