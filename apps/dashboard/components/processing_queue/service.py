"""
Processing Queue Business Logic
"""
import requests
from components import register_component

@register_component('processing_queue')
class ProcessingQueueService:
    """Service for managing processing queue"""
    
    def __init__(self):
        self.ml_backend_url = 'http://localhost:9091'
    
    def get_queue_status(self):
        """Get processing queue status"""
        try:
            response = requests.get(f'{self.ml_backend_url}/queue/status', timeout=3)
            if response.status_code == 200:
                return response.json()
            else:
                return {'error': 'ML backend not available'}
        except Exception as e:
            return {'error': str(e)}
    
    def get_queue_metrics(self):
        """Get queue performance metrics"""
        try:
            queue_status = self.get_queue_status()
            if 'error' in queue_status:
                return queue_status
            
            stats = queue_status.get('stats', {})
            return {
                'queue_length': stats.get('queue_length', 0),
                'is_processing': stats.get('is_processing', False),
                'total_processed': stats.get('total_processed', 0),
                'total_failed': stats.get('total_failed', 0),
                'processing_rate': stats.get('processing_rate', 0),
                'average_processing_time': stats.get('avg_processing_time', 0)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def get_recent_tasks(self, limit=10):
        """Get recent completed tasks"""
        try:
            queue_status = self.get_queue_status()
            if 'error' in queue_status:
                return []
            
            recent_tasks = queue_status.get('recent_tasks', [])
            return recent_tasks[-limit:] if recent_tasks else []
        except Exception:
            return []
    
    def get_pending_tasks(self, limit=10):
        """Get pending tasks in queue"""
        try:
            queue_status = self.get_queue_status()
            if 'error' in queue_status:
                return []
            
            pending_tasks = queue_status.get('queue', [])
            return pending_tasks[:limit] if pending_tasks else []
        except Exception:
            return []