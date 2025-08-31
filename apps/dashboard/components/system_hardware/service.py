"""
System Hardware Service
CRITICAL: This preserves EXACT hardware detection logic from monolithic system
"""

import psutil
import platform
import subprocess
from datetime import datetime


class SystemHardwareService:
    """Service for System Hardware component
    
    CRITICAL: This provides the SAME hardware info as the monolithic system
    Original function: api_system_hardware() in app.py lines 2015-2069
    """
    
    def get_hardware_info(self):
        """Get system hardware information
        
        CRITICAL: This MUST produce the EXACT same data structure as the original
        Original function: api_system_hardware() in app.py lines 2015-2069
        """
        try:
            # CPU information
            cpu_info = {
                'name': platform.processor() or 'Unknown CPU',
                'cores': psutil.cpu_count(logical=False),
                'threads': psutil.cpu_count(logical=True),
                'usage': psutil.cpu_percent(interval=1),
                'frequency': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
            }
            
            # Memory information
            memory = psutil.virtual_memory()
            memory_info = {
                'total': memory.total // 1024 // 1024,  # MB
                'available': memory.available // 1024 // 1024,  # MB
                'used': memory.used // 1024 // 1024,  # MB
                'percent': memory.percent
            }
            
            # Disk information
            disk = psutil.disk_usage('/')
            disk_info = {
                'total': disk.total // 1024 // 1024 // 1024,  # GB
                'used': disk.used // 1024 // 1024 // 1024,  # GB
                'free': disk.free // 1024 // 1024 // 1024,  # GB
                'percent': (disk.used / disk.total) * 100
            }
            
            # GPU information
            gpu_info = self._get_gpu_info()
            
            # System information
            system_info = {
                'platform': platform.system(),
                'platform_release': platform.release(),
                'platform_version': platform.version(),
                'architecture': platform.machine(),
                'hostname': platform.node(),
                'boot_time': datetime.fromtimestamp(psutil.boot_time()).isoformat()
            }
            
            return {
                'cpu': cpu_info,
                'memory': memory_info,
                'disk': disk_info,
                'gpu': gpu_info,
                'system': system_info,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _get_gpu_info(self):
        """Get GPU information if available
        
        CRITICAL: This is EXACT same logic as get_gpu_info() in app.py lines 946-991
        """
        try:
            # Try nvidia-ml-py3 first
            try:
                import pynvml
                pynvml.nvmlInit()
                device_count = pynvml.nvmlDeviceGetCount()
                gpus = []
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    
                    gpus.append({
                        'id': i,
                        'name': name,
                        'memory_used': mem_info.used // 1024 // 1024,  # MB
                        'memory_total': mem_info.total // 1024 // 1024,  # MB
                        'memory_percent': (mem_info.used / mem_info.total) * 100,
                        'utilization': util.gpu
                    })
                return gpus
            except:
                # Fallback to nvidia-smi
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'], 
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    gpus = []
                    for i, line in enumerate(result.stdout.strip().split('\n')):
                        parts = line.split(', ')
                        if len(parts) >= 4:
                            gpus.append({
                                'id': i,
                                'name': parts[0].strip(),
                                'memory_used': int(parts[1]),
                                'memory_total': int(parts[2]),
                                'memory_percent': (int(parts[1]) / int(parts[2])) * 100,
                                'utilization': int(parts[3])
                            })
                    return gpus
                else:
                    return []
        except:
            return []