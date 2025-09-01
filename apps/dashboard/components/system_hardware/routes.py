"""
System Hardware Routes
"""

from flask import jsonify
from . import system_hardware_bp
from .service import SystemHardwareService

# Initialize service
service = SystemHardwareService()

@system_hardware_bp.route('/api/system/hardware')
def api_system_hardware():
    """Get system hardware information"""
    # Get hardware info from service
    hardware_info = service.get_hardware_info()
    
    return jsonify(hardware_info)
