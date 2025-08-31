"""
Processing Queue API Routes
"""
from flask import Blueprint, jsonify
from .service import ProcessingQueueService
from .sse_handler import QueueSSEHandler

# Create processing queue blueprint
processing_queue_bp = Blueprint('processing_queue', __name__, 
                               template_folder='templates',
                               static_folder='static',
                               static_url_path='/processing_queue/static',
                               url_prefix='/api')

# Initialize service
queue_service = ProcessingQueueService()
sse_handler = QueueSSEHandler()

@processing_queue_bp.route('/queue/status')
def api_queue_status():
    """Get processing queue status"""
    result = queue_service.get_queue_status()
    if 'error' in result:
        return jsonify(result), 503
    return jsonify(result)

@processing_queue_bp.route('/queue/metrics')
def api_queue_metrics():
    """Get queue performance metrics"""
    result = queue_service.get_queue_metrics()
    if 'error' in result:
        return jsonify(result), 503
    return jsonify(result)

@processing_queue_bp.route('/processing/queue/stream')
def api_processing_queue_stream():
    """SSE endpoint for real-time processing queue updates"""
    return sse_handler.stream_queue_updates()