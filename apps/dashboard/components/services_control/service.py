"""
Services Control Service  
CRITICAL: This preserves EXACT service management logic from monolithic system
All subprocess commands and process management must match original implementation
"""

import subprocess
import os
import psutil
import requests
import time
from pathlib import Path
import signal
from config.settings import DashboardConfig


class ServicesControlService:
    """Service for Services Control component
    
    CRITICAL: This handles service start/stop/restart exactly as the original
    DO NOT modify subprocess commands - preserve existing service management
    """
    
    def __init__(self):
        # Get project root directory (same path resolution as original)
        self.project_root = Path(__file__).parent.parent.parent.parent.parent
        print(f"[DEBUG] ServicesControlService project_root: {self.project_root}")
        
        # CRITICAL: These MUST match the exact commands from monolithic app.py
        self.service_commands = {
            'ml_service': f'cd {self.project_root} && source venv/bin/activate && python apps/ml_service/app_modular.py',
            'ml_backend': f'cd {self.project_root} && source venv/bin/activate && python services/label_studio_connector/app.py',
            'label_studio': f'cd {self.project_root} && LOCAL_FILES_SERVING_ENABLED=true label-studio start --port 8200'
        }
        
        # Service configuration for health checks and shutdown - use centralized config
        self.service_configs = DashboardConfig.SERVICES
    
    def start_service(self, service_name):
        """Start a service via subprocess
        
        CRITICAL: This MUST match the EXACT logic from api_start_service() in app.py
        """
        if service_name not in self.service_commands:
            raise ValueError(f"Unknown service: {service_name}")
            
        command = self.service_commands[service_name]
        config = self.service_configs[service_name]
        
        # Enhanced check: test both health endpoint AND process existence (EXACT as original)
        import psutil
        
        # Check 1: Process existence
        running_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info['cmdline']
                if cmdline and any(config['process_name'] in cmd for cmd in cmdline):
                    running_processes.append(proc.info['pid'])
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        
        # Check 2: Port availability
        port_in_use = False
        for conn in psutil.net_connections():
            if conn.laddr.port == config['port'] and conn.status == 'LISTEN':
                port_in_use = True
                break
        
        # Check 3: Health endpoint
        health_ok = False
        try:
            response = requests.get(config['health_url'], timeout=3)
            health_ok = response.status_code == 200
        except:
            pass
        
        # If any check indicates the service is running, don't start another instance
        if running_processes or port_in_use or health_ok:
            message_parts = []
            if running_processes:
                message_parts.append(f"Process running (PIDs: {running_processes})")
            if port_in_use:
                message_parts.append(f"Port {config['port']} in use")
            if health_ok:
                message_parts.append("Health check OK")
            
            return {
                'message': f'{service_name} is already running - {", ".join(message_parts)}',
                'status': 'already_running',
                'details': {
                    'processes': running_processes,
                    'port_in_use': port_in_use,
                    'health_ok': health_ok
                }
            }
        
        print(f"[INFO] Starting {service_name} - no existing instances detected")
        
        try:
            print(f"[DEBUG] Starting {service_name} with command: {command}")
            
            # Use /bin/bash -c to properly handle the source command (EXACT as original)
            log_file = f"{service_name}.log"
            full_command = f"/bin/bash -c '{command} > {log_file} 2>&1 &'"
            result = subprocess.Popen(
                full_command,
                shell=True,
                cwd=str(self.project_root),
                preexec_fn=os.setsid if hasattr(os, 'setsid') else None
            )
            
            print(f"[DEBUG] {config['name']} started with PID: {result.pid}")
            
            return {
                'message': f'{service_name} startup initiated',
                'pid': result.pid,
                'status': 'starting'
            }
            
        except Exception as e:
            print(f"[ERROR] Failed to start {service_name}: {e}")
            raise e
    
    def stop_service(self, service_name):
        """Stop a service via shutdown endpoint or process kill
        
        CRITICAL: This MUST match the EXACT logic from api_stop_service() in app.py
        """
        if service_name not in self.service_configs:
            raise ValueError(f"Unknown service: {service_name}")
            
        config = self.service_configs[service_name]
        
        try:
            # First try to find if the process is running
            running_processes = self._find_service_processes(service_name)
            
            if not running_processes:
                return {
                    'message': f'{config["name"]} is not currently running',
                    'status': 'not_running'
                }
            
            # Try graceful shutdown via API endpoint first
            if service_name in ['ml_service', 'ml_backend']:
                try:
                    shutdown_url = config['shutdown_url']
                    print(f"[DEBUG] Attempting graceful shutdown for {service_name} via {shutdown_url}")
                    response = requests.post(shutdown_url, timeout=5)
                    if response.status_code == 200:
                        print(f"[DEBUG] {config['name']} shutdown via API successful")
                        
                        # Wait for process to actually stop (give it 3 seconds)
                        import time
                        for i in range(6):  # Check 6 times over 3 seconds
                            time.sleep(0.5)
                            remaining_processes = self._find_service_processes(service_name)
                            if not remaining_processes:
                                print(f"[DEBUG] {config['name']} shutdown completed successfully")
                                return {
                                    'message': f'{config["name"]} stopped successfully',
                                    'status': 'stopped'
                                }
                        
                        # Still running after 3 seconds, but shutdown was initiated
                        print(f"[DEBUG] {config['name']} shutdown initiated but still running")
                        return {
                            'message': f'{config["name"]} shutdown initiated',
                            'status': 'stopping'
                        }
                        
                except requests.exceptions.RequestException as e:
                    print(f"[DEBUG] API shutdown failed for {service_name}: {e}")
                    # Continue to process termination below
            
            # For label_studio or if API shutdown failed, terminate processes
            print(f"[DEBUG] Terminating processes for {service_name}")
            terminated_count = 0
            
            for proc in running_processes:
                try:
                    print(f"[DEBUG] Terminating PID {proc.pid} for {service_name}")
                    proc.terminate()
                    terminated_count += 1
                    
                    # Wait up to 5 seconds for graceful termination
                    try:
                        proc.wait(timeout=5)
                        print(f"[DEBUG] Process {proc.pid} terminated gracefully")
                    except subprocess.TimeoutExpired:
                        print(f"[DEBUG] Process {proc.pid} didn't terminate gracefully, killing...")
                        proc.kill()
                        proc.wait()
                        print(f"[DEBUG] Process {proc.pid} killed")
                        
                except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                    print(f"[DEBUG] Process {proc.pid} already gone or access denied: {e}")
                    continue
            
            if terminated_count > 0:
                return {
                    'message': f'{config["name"]} stopped ({terminated_count} processes terminated)',
                    'status': 'stopped'
                }
            else:
                return {
                    'message': f'{config["name"]} was not running',
                    'status': 'not_running'
                }
                
        except Exception as e:
            print(f"[ERROR] Error stopping {service_name}: {e}")
            raise e
    
    def restart_service(self, service_name):
        """Restart a service (stop then start)
        
        CRITICAL: This MUST match the EXACT logic from api_restart_service() in app.py
        """
        try:
            # First attempt to stop
            print(f"[DEBUG] Restarting {service_name}: stopping first...")
            stop_result = self.stop_service(service_name)
            print(f"[DEBUG] Stop result: {stop_result}")
            
            # Wait a moment for shutdown (same as original)
            time.sleep(3)
            
            # Then start
            print(f"[DEBUG] Restarting {service_name}: starting...")
            start_result = self.start_service(service_name)
            print(f"[DEBUG] Start result: {start_result}")
            
            return {
                'message': f'{service_name} restart initiated',
                'status': 'restarting',
                'stop_result': stop_result,
                'start_result': start_result
            }
            
        except Exception as e:
            print(f"[ERROR] Error restarting {service_name}: {e}")
            raise e
    
    def _find_service_processes(self, service_name):
        """Find running processes for a service
        
        Enhanced to check both command line and working directory
        """
        config = self.service_configs[service_name]
        process_name = config['process_name']
        
        running_processes = []
        
        try:
            for proc in psutil.process_iter(['pid', 'cmdline', 'cwd']):
                try:
                    cmdline = ' '.join(proc.info['cmdline'] or [])
                    cwd = proc.info.get('cwd', '')
                    
                    # For ML service, check if it's python services/ml_engine/app.py OR python app.py in ml_engine directory
                    if service_name == 'ml_service':
                        ml_match = ('python' in cmdline and 'app.py' in cmdline and 
                                   ('ml_engine' in cmdline or 'ml_engine' in cwd))
                        if ml_match:
                            running_processes.append(proc)
                            print(f"[DEBUG] Found {service_name} process: PID {proc.info['pid']} in {cwd}")
                    else:
                        # For other services, use original logic
                        if process_name in cmdline:
                            running_processes.append(proc)
                            print(f"[DEBUG] Found {service_name} process: PID {proc.info['pid']}")
                            
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                    
        except Exception as e:
            print(f"[ERROR] Error finding processes for {service_name}: {e}")
            
        return running_processes
    
    def check_service_health(self, service_name):
        """Check if a service is healthy via its health endpoint
        
        Used by frontend for health check polling after service start
        """
        if service_name not in self.service_configs:
            return {'healthy': False, 'error': f'Unknown service: {service_name}'}
            
        config = self.service_configs[service_name]
        health_url = config['health_url']
        
        try:
            response = requests.get(health_url, timeout=3)
            healthy = response.status_code == 200
            
            return {
                'healthy': healthy,
                'status_code': response.status_code,
                'url': health_url
            }
            
        except requests.exceptions.RequestException as e:
            return {
                'healthy': False, 
                'error': str(e),
                'url': health_url
            }