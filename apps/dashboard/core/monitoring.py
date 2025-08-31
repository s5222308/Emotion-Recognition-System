"""
System monitoring service
"""
import time
import threading
import requests
import subprocess
import psutil
import platform
import json
from collections import deque, defaultdict
from datetime import datetime
from config.settings import DashboardConfig

class SystemMonitor:
    """Centralized system monitoring service"""
    
    def __init__(self):
        self.running = True
        self.thread = None
        self.services_status = {}
        self.system_logs = deque(maxlen=DashboardConfig.MAX_LOG_ENTRIES)
        self.service_metrics = defaultdict(lambda: {
            'requests': 0, 
            'errors': 0, 
            'uptime': None, 
            'status': 'unknown', 
            'error_log': deque(maxlen=50)
        })
        self.debug_images = []
        
    def start(self):
        """Start monitoring thread"""
        if self.thread is None or not self.thread.is_alive():
            self.running = True
            self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.thread.start()
            print('[DEBUG] SystemMonitor thread started:', self.thread.is_alive())
        
    def stop(self):
        """Stop monitoring thread"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
            
    def _monitor_loop(self):
        """Main monitoring loop"""
        print('[DEBUG] Monitor loop started, entering main loop...')
        
        while self.running:
            try:
                self._check_all_services()
                time.sleep(3)
            except Exception as e:
                self._add_log('ERROR', f'Monitor loop error: {e}')
                time.sleep(5)
                
    def _check_all_services(self):
        """Check status of all configured services"""
        print('[DEBUG] Starting broadcast preparation at', datetime.now())
        
        # Import global service metrics
        from core import service_metrics
        print(f'[DEBUG] service_metrics object ID: {id(service_metrics)}')
        
        for service_id in DashboardConfig.SERVICES.keys():
            status = self._check_service_health(service_id)
            old_status = self.services_status.get(service_id, {}).get('status', 'unknown')
            
            if status != old_status:
                self._add_log('INFO', f'Service {service_id} status changed: {old_status} -> {status}')
                
            self.services_status[service_id] = {
                'status': status,
                'last_check': datetime.now().isoformat(),
                'name': DashboardConfig.SERVICES[service_id]['name']
            }
            
            # CRITICAL: Update global service_metrics that API endpoints use ALWAYS
            service_metrics[service_id]['status'] = status
            service_metrics[service_id]['last_check'] = datetime.now()
            if status == 'healthy' and service_metrics[service_id]['uptime'] is None:
                service_metrics[service_id]['uptime'] = datetime.now()
            print(f'[DEBUG] ALWAYS Updated service_metrics[{service_id}]["status"] = {status}')
            
    def _check_service_health(self, service_name):
        """Check health of a specific service"""
        config = DashboardConfig.get_service_config(service_name)
        if not config:
            return 'unknown'
            
        health_url = config.get('health_url', '')
        print(f'[DEBUG] Checking {service_name} health at {health_url}')
        
        try:
            response = requests.get(health_url, timeout=3)
            if response.status_code == 200:
                status = 'healthy'
                print(f'[DEBUG] {service_name}: HTTP 200 -> status \'{status}\' (was \'{self.services_status.get(service_name, {}).get("status", "unknown")}\')')
                return status
            else:
                status = 'unhealthy'
                error_msg = f'HTTP {response.status_code}'
                print(f'[DEBUG] {service_name}: {error_msg} -> status \'{status}\'')
                return status
                
        except Exception as e:
            status = 'down'
            error_msg = str(e)
            old_status = self.services_status.get(service_name, {}).get('status', 'unknown')
            print(f'[DEBUG] {service_name}: Request failed -> status \'{status}\' (was \'{old_status}\'), error: {error_msg}')
            return status
            
    def _add_log(self, level, message):
        """Add log entry"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'message': message
        }
        self.system_logs.append(log_entry)
        
    def get_logs(self, level_filter='ALL', limit=50):
        """Get filtered logs"""
        logs = list(self.system_logs)
        
        if level_filter != 'ALL':
            logs = [log for log in logs if log['level'] == level_filter]
            
        return logs[-limit:] if limit else logs
        
    def get_services_status(self):
        """Get current services status"""
        return dict(self.services_status)
        
    def get_system_hardware(self):
        """Get system hardware information"""
        try:
            cpu_count = psutil.cpu_count()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                'cpu': {
                    'count': cpu_count,
                    'usage': psutil.cpu_percent(interval=1)
                },
                'memory': {
                    'total': memory.total,
                    'available': memory.available,
                    'percent': memory.percent
                },
                'disk': {
                    'total': disk.total,
                    'used': disk.used,
                    'free': disk.free,
                    'percent': (disk.used / disk.total) * 100
                },
                'system': {
                    'platform': platform.system(),
                    'version': platform.version(),
                    'machine': platform.machine()
                }
            }
        except Exception as e:
            self._add_log('ERROR', f'Failed to get system hardware: {e}')
            return {}
            
    def start_service(self, service_name):
        """Start a service"""
        config = DashboardConfig.get_service_config(service_name)
        if not config:
            return {'status': 'error', 'message': f'Unknown service: {service_name}'}
            
        try:
            # Check if already running
            port = config.get('port')
            if port and self._is_port_in_use(port):
                return {'status': 'already_running', 'message': f'{service_name} is already running on port {port}'}
                
            # Start the service
            cmd = config['start_cmd']
            cwd = config.get('cwd', '.')
            
            process = subprocess.Popen(
                cmd,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            time.sleep(2)  # Give service time to start
            
            if process.poll() is None:  # Process is still running
                self._add_log('INFO', f'Started {service_name} (PID: {process.pid})')
                return {'status': 'started', 'message': f'{service_name} started successfully'}
            else:
                stderr = process.stderr.read().decode()
                return {'status': 'failed', 'message': f'Failed to start {service_name}: {stderr}'}
                
        except Exception as e:
            self._add_log('ERROR', f'Failed to start {service_name}: {e}')
            return {'status': 'error', 'message': str(e)}
            
    def stop_service(self, service_name):
        """Stop a service"""
        config = DashboardConfig.get_service_config(service_name)
        if not config:
            return {'status': 'error', 'message': f'Unknown service: {service_name}'}
            
        try:
            port = config.get('port')
            if port:
                # Kill processes using the port
                killed = self._kill_processes_on_port(port)
                if killed:
                    self._add_log('INFO', f'Stopped {service_name} processes')
                    return {'status': 'stopped', 'message': f'{service_name} stopped successfully'}
                else:
                    return {'status': 'not_running', 'message': f'{service_name} was not running'}
            else:
                return {'status': 'error', 'message': 'No port configured for service'}
                
        except Exception as e:
            self._add_log('ERROR', f'Failed to stop {service_name}: {e}')
            return {'status': 'error', 'message': str(e)}
            
    def _is_port_in_use(self, port):
        """Check if a port is in use"""
        try:
            connections = psutil.net_connections()
            return any(conn.laddr.port == port for conn in connections if conn.laddr)
        except:
            return False
            
    def _kill_processes_on_port(self, port):
        """Kill processes using a specific port"""
        killed = False
        try:
            connections = psutil.net_connections()
            for conn in connections:
                if conn.laddr and conn.laddr.port == port:
                    try:
                        process = psutil.Process(conn.pid)
                        process.terminate()
                        killed = True
                    except:
                        pass
        except:
            pass
        return killed