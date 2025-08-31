"""
System Hardware Component
Displays real-time system hardware metrics
"""

from flask import Blueprint

# Create blueprint for system hardware
system_hardware_bp = Blueprint(
    'system_hardware', 
    __name__,
    template_folder='templates',
    static_folder='static',
    static_url_path='/system_hardware/static'
)

# Import routes to register them
from . import routes

def init_system_hardware(app):
    """Initialize system hardware component with Flask app"""
    # Register blueprint
    app.register_blueprint(system_hardware_bp, url_prefix='')
    
    return system_hardware_bp