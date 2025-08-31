/**
 * System Logs Component JavaScript
 * CRITICAL: These functions MUST preserve exact signatures and behavior
 * The main polling system calls these functions globally
 */

// System Logs Component Namespace
window.Dashboard = window.Dashboard || {};
window.Dashboard.SystemLogs = (function() {
    
    /**
     * Update logs display via polling
     * CRITICAL: This function is called by main polling system every 2 polls (8 seconds)
     * Original function: updateLogs() lines 1345-1374 in dashboard.html
     */
    function updateLogs() {
        const filter = document.getElementById('log-filter').value;
        fetch(`/api/logs?level=${filter}&limit=50`)
            .then(response => response.json())
            .then(logs => {
                const container = document.getElementById('logs-container');
                container.innerHTML = '';
                
                logs.reverse().forEach(log => {
                    const timestamp = new Date(log.timestamp).toLocaleTimeString();
                    const logDiv = document.createElement('div');
                    logDiv.className = `log-entry log-${log.level}`;
                    
                    const message = log.message.length > 120 ? 
                        log.message.substring(0, 117) + '...' : log.message;
                    
                    const levelColor = log.level === 'ERROR' ? 'red' : log.level === 'WARN' ? 'orange' : 'blue';
                    
                    logDiv.innerHTML = `
                        <span style="color: gray;">[${timestamp}]</span>
                        <span style="background: ${levelColor}; color: white; padding: 2px 6px; margin: 0 8px; border-radius: 3px; font-size: 0.8em;">
                            ${log.level}
                        </span>
                        <span title="${log.message}">${message}</span>
                    `;
                    container.appendChild(logDiv);
                });
            })
            .catch(error => console.error('Error updating logs:', error));
    }
    
    /**
     * Update logs display with live data (used by WebSocket/SSE if available)
     * CRITICAL: This function is called for real-time log updates
     * Original function: updateLogsLive() lines 1556-1588 in dashboard.html
     */
    function updateLogsLive(logs) {
        console.log('[DEBUG] updateLogsLive called with', logs.length, 'logs');
        const container = document.getElementById('logs-container');
        const filter = document.getElementById('log-filter').value;
        
        // Filter logs if needed
        let filteredLogs = logs;
        if (filter !== 'ALL') {
            filteredLogs = logs.filter(log => log.level === filter);
        }
        
        container.innerHTML = '';
        filteredLogs.slice().reverse().forEach(log => {
            const timestamp = new Date(log.timestamp).toLocaleTimeString();
            const logDiv = document.createElement('div');
            logDiv.className = `log-entry log-${log.level}`;
            
            const message = log.message.length > 120 ? 
                log.message.substring(0, 117) + '...' : log.message;
            
            const levelColor = log.level === 'ERROR' ? 'red' : log.level === 'WARN' ? 'orange' : 'blue';
            
            logDiv.innerHTML = `
                <span style="color: gray;">[${timestamp}]</span>
                <span style="background: ${levelColor}; color: white; padding: 2px 6px; margin: 0 8px; border-radius: 3px; font-size: 0.8em;">
                    ${log.level}
                </span>
                <span title="${log.message}">${message}</span>
            `;
            container.appendChild(logDiv);
        });
        console.log('[DEBUG] Updated logs container with', filteredLogs.length, 'filtered logs');
    }
    
    /**
     * Initialize System Logs component
     */
    function init() {
        console.log('[DEBUG] System Logs component initialized');
        
        // Set up filter change listener (preserves original behavior from line 1886)
        const logFilter = document.getElementById('log-filter');
        if (logFilter) {
            logFilter.addEventListener('change', updateLogs);
        }
        
        // Set initial loading state
        const container = document.getElementById('logs-container');
        if (container) {
            container.innerHTML = '<p style="color: gray;">Loading logs...</p>';
        }
        
        // Component is ready
        console.log('[DEBUG] System Logs component ready');
    }
    
    // Public interface
    return {
        init: init,
        updateLogs: updateLogs,
        updateLogsLive: updateLogsLive
    };
})();

// CRITICAL: Export functions globally for main polling system compatibility
// The main polling system expects these exact function names to exist globally
window.updateLogs = window.Dashboard.SystemLogs.updateLogs;
window.updateLogsLive = window.Dashboard.SystemLogs.updateLogsLive;

// Initialize component when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    if (window.Dashboard.SystemLogs) {
        window.Dashboard.SystemLogs.init();
    }
});