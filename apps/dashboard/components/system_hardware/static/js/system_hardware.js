/**
 * System Hardware Component JavaScript
 * Handles real-time hardware monitoring updates
 */

let hardwareUpdateInterval;

function initializeSystemHardware() {
    console.log('Initializing System Hardware component...');
    
    // Start hardware monitoring
    updateHardwareInfo();
    
    // Set up regular updates every 30 seconds (OPTIMIZED)
    if (hardwareUpdateInterval) {
        clearInterval(hardwareUpdateInterval);
    }
    hardwareUpdateInterval = setInterval(updateHardwareInfo, 30000);
    
    console.log('System Hardware component initialized');
}

function updateHardwareInfo() {
    // Use safeFetch if available, otherwise regular fetch (safe fallback)
    const fetchFn = window.DashboardState?.safeFetch || fetch;
    
    fetchFn('/api/system/hardware')
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                console.error('Hardware data error:', data.error);
                return;
            }
            
            // Update CPU info
            if (data.cpu) {
                const cpuUsage = Math.round(data.cpu.usage || 0);
                document.getElementById('cpu-usage').textContent = `${cpuUsage}%`;
                document.getElementById('cpu-progress').style.width = `${cpuUsage}%`;
                
                // Update CPU details
                const cpuName = data.cpu.name || 'Unknown CPU';
                const cpuCores = data.cpu.cores || 'N/A';
                const cpuThreads = data.cpu.threads || 'N/A';
                document.getElementById('cpu-name').textContent = `${cpuName} (${cpuCores} cores, ${cpuThreads} threads)`;
            }
            
            // Update Memory info
            if (data.memory) {
                const memoryPercent = Math.round(data.memory.percent || 0);
                document.getElementById('memory-usage').textContent = `${memoryPercent}%`;
                document.getElementById('memory-progress').style.width = `${memoryPercent}%`;
                
                // Update Memory details
                const memoryTotal = Math.round((data.memory.total || 0) / 1024); // Convert to GB
                const memoryUsed = Math.round((data.memory.used || 0) / 1024);
                const memoryAvailable = Math.round((data.memory.available || 0) / 1024);
                document.getElementById('memory-details').textContent = `${memoryUsed}GB / ${memoryTotal}GB (${memoryAvailable}GB available)`;
            }
            
            // Update Disk info
            if (data.disk) {
                const diskPercent = Math.round(data.disk.percent || 0);
                document.getElementById('disk-usage').textContent = `${diskPercent}%`;
                document.getElementById('disk-progress').style.width = `${diskPercent}%`;
            }
            
            // Update GPU info
            if (data.gpu && data.gpu.length > 0) {
                const gpu = data.gpu[0]; // Use first GPU
                const gpuUsage = Math.round(gpu.utilization || 0);
                document.getElementById('gpu-usage').textContent = `${gpuUsage}%`;
                document.getElementById('gpu-progress').style.width = `${gpuUsage}%`;
                
                // Update GPU details
                const gpuName = gpu.name || 'Unknown GPU';
                const gpuMemory = Math.round(gpu.memory_used || 0);
                const gpuMemoryTotal = Math.round(gpu.memory_total || 0);
                document.getElementById('gpu-name').textContent = `${gpuName} (${gpuMemory}MB / ${gpuMemoryTotal}MB)`;
            } else {
                document.getElementById('gpu-usage').textContent = 'N/A';
                document.getElementById('gpu-progress').style.width = '0%';
                document.getElementById('gpu-name').textContent = 'No GPU detected';
            }
            
            // Update System info
            if (data.system) {
                document.getElementById('platform').textContent = `${data.system.platform} ${data.system.platform_release}`;
                document.getElementById('architecture').textContent = data.system.architecture;
                document.getElementById('hostname').textContent = data.system.hostname;
            }
            
        })
        .catch(error => {
            console.error('Failed to fetch hardware info:', error);
        });
}

// Export function for global access
window.initializeSystemHardware = initializeSystemHardware;