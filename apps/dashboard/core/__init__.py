"""
Core services for dashboard components
"""
from collections import defaultdict, deque
from .monitoring import SystemMonitor
from .sse_service import SSEService

# Global state - shared across all components
system_logs = deque(maxlen=1000)  # Keep last 1000 log entries
service_metrics = defaultdict(lambda: {'requests': 0, 'errors': 0, 'uptime': None, 'status': 'unknown', 'error_log': deque(maxlen=50)})
debug_images = []
processing_queue = []  # Track actual processing tasks
processing_times = deque(maxlen=100)  # Keep last 100 processing times for averaging

# Service configuration - import from centralized config
from config.settings import DashboardConfig
SERVICES = DashboardConfig.SERVICES

__all__ = [
    'SystemMonitor', 
    'SSEService',
    'system_logs',
    'service_metrics', 
    'debug_images',
    'processing_queue',
    'processing_times',
    'SERVICES'
]