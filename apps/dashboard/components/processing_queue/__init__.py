"""
Processing Queue Component
Handles queue monitoring and real-time updates via SSE
"""
from .routes import processing_queue_bp
from .service import ProcessingQueueService
from .sse_handler import QueueSSEHandler

def init_processing_queue(app):
    """Initialize Processing Queue component with Flask app"""
    app.register_blueprint(processing_queue_bp)
    return processing_queue_bp

__all__ = ['processing_queue_bp', 'ProcessingQueueService', 'QueueSSEHandler', 'init_processing_queue']