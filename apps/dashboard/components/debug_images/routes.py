"""
Debug Images API Routes
CRITICAL: This preserves EXACT API endpoints from monolithic system
All route paths and response formats must match original implementation
"""

from flask import Blueprint, jsonify, send_from_directory
from .service import DebugImagesService

# Create blueprint - NO URL PREFIX for component-specific routes
debug_images_bp = Blueprint('debug_images', __name__, 
                           template_folder='templates',
                           static_folder='static', 
                           static_url_path='/debug_images/static')

# Service instance
service = DebugImagesService()

@debug_images_bp.route('/api/debug-images')
def api_debug_images():
    """Get debug images list
    
    CRITICAL: This MUST match the EXACT endpoint from original
    Original endpoint: /api/debug-images (dashboard.html line 1518)
    """
    try:
        images = service.get_debug_images_list()
        return jsonify(images)
    except Exception as e:
        print(f"[ERROR] Debug images API error: {e}")
        return jsonify([])

@debug_images_bp.route('/debug-images/<filename>')
def serve_debug_image(filename):
    """Serve debug image files
    
    CRITICAL: This MUST match the EXACT path from original
    Original path: /debug-images/${img.filename} (dashboard.html line 1535)
    """
    try:
        directory, filename = service.serve_debug_image(filename)
        if directory and filename:
            return send_from_directory(directory, filename)
        else:
            return "Image not found", 404
    except Exception as e:
        print(f"[ERROR] Debug image serving error: {e}")
        return "Image not found", 404

def init_debug_images(app):
    """Initialize debug images component with Flask app
    
    CRITICAL: This follows the EXACT same pattern as other successful components
    """
    print("[DEBUG] Initializing Debug Images component...")
    
    # Register the blueprint with the main app
    app.register_blueprint(debug_images_bp)
    
    print("[DEBUG] Debug Images component initialized successfully")
    return debug_images_bp