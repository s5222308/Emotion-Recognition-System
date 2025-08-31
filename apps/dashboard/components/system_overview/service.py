"""
System Overview Service
CRITICAL: This preserves EXACT business logic from monolithic system
All calculations and data structures must match original implementation
"""

from collections import defaultdict, deque
from datetime import datetime


class SystemOverviewService:
    """Service for System Overview component
    
    CRITICAL: This accesses the SAME global state as the monolithic system
    DO NOT create new data structures - use existing global state
    """
    
    def __init__(self):
        # CRITICAL: Import global state from main app
        # This ensures we use the SAME data that the monolithic system uses
        pass
    
    def get_system_metrics(self):
        """Get system performance metrics
        
        CRITICAL: This MUST produce the EXACT same calculations as the original
        Original function: api_system_metrics() in app.py lines 901-917
        """
        # Import global state from core module
        from core import service_metrics, debug_images, system_logs, SERVICES
        
        # EXACT same calculations as original (lines 905-907)
        total_requests = sum(m.get('requests', 0) for m in service_metrics.values())
        total_errors = sum(m.get('errors', 0) for m in service_metrics.values())
        healthy_services = sum(1 for m in service_metrics.values() if m.get('status') == 'healthy')
        
        # EXACT same response structure as original (lines 909-917)
        return {
            'total_requests': total_requests,
            'total_errors': total_errors,
            'error_rate': (total_errors / max(total_requests, 1)) * 100,
            'healthy_services': healthy_services,
            'total_services': len(SERVICES),
            'debug_images_count': len(debug_images),
            'logs_count': len(system_logs)
        }
    
    def get_services_status(self):
        """Get all services status
        
        CRITICAL: This MUST produce the EXACT same data structure as the original
        This is used by System Overview for service counts and status display
        """
        # Import global state from core module
        from core import service_metrics, SERVICES
        
        # EXACT same logic as original api_services_status()
        services_data = {}
        
        print(f'[DEBUG API] service_metrics object ID: {id(service_metrics)}')
        
        for service_id, service_config in SERVICES.items():
            metrics = service_metrics[service_id]
            print(f'[DEBUG API] {service_id} status from service_metrics: {metrics["status"]}')
            
            # CRITICAL: Direct health check if metrics show unknown status
            actual_status = metrics['status']
            if actual_status == 'unknown':
                # Do direct health check
                try:
                    import requests
                    health_url = service_config['base_url'] + '/health'
                    if service_id == 'label_studio':
                        health_url = service_config['base_url'] + '/'
                    response = requests.get(health_url, timeout=2)
                    if response.status_code == 200:
                        actual_status = 'running'
                        print(f'[DEBUG] Direct health check: {service_id} is running')
                    else:
                        actual_status = 'down'
                        print(f'[DEBUG] Direct health check: {service_id} is down (HTTP {response.status_code})')
                except:
                    actual_status = 'down'
                    print(f'[DEBUG] Direct health check: {service_id} is down (connection error)')
                    
            services_data[service_id] = {
                'name': service_config['name'],
                'url': service_config['url'],
                'base_url': service_config['base_url'],
                'description': service_config['description'],
                'status': actual_status,
                'requests': metrics['requests'],
                'errors': metrics['errors'],
                'uptime': metrics['uptime'],
                'uptime_seconds': self._calculate_uptime_seconds(metrics, actual_status),
                'last_health_check': datetime.now().isoformat()
            }
        
        return services_data
    
    def _calculate_uptime_seconds(self, metrics, status):
        """Calculate uptime seconds exactly like the main monitoring service"""
        if status == 'down' or not metrics.get('uptime'):
            return 0
        
        uptime_start = metrics['uptime']
        if isinstance(uptime_start, str):
            # Handle string datetime (shouldn't happen but just in case)
            return 0
            
        return int((datetime.now() - uptime_start).total_seconds())