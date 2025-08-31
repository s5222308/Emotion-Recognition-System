/**
 * Main Dashboard Polling System
 * CRITICAL: This preserves the exact polling behavior from the monolithic dashboard
 */

// Dashboard polling main function
function startDashboardPolling() {
    console.log('[POLLING] Starting dashboard polling system');
    console.log('[POLLING] POLLING_INTERVAL:', window.DashboardState.POLLING_INTERVAL);
    console.log('[POLLING] DashboardState object:', window.DashboardState);
    
    // Use shared state from global
    const state = window.DashboardState;
    
    function pollForUpdates() {
        console.log('[DEBUG POLLING] Polling for updates... count:', state.pollCount);
        state.pollCount++;
        
        // Get system metrics - every 4 polls (16 seconds)
        if (state.pollCount % 4 === 0) {
            fetch('/api/system/metrics')
                .then(response => response.json())
                .then(data => {
                    if (typeof updateMetricsLive === 'function') {
                        updateMetricsLive(data);
                    } else if (window.Dashboard.SystemOverview && window.Dashboard.SystemOverview.updateMetricsLive) {
                        window.Dashboard.SystemOverview.updateMetricsLive(data);
                    }
                    state.isConnected = true;
                    state.updateConnectionStatus();
                })
                .catch(error => {
                    console.error('Error fetching system metrics:', error);
                    state.isConnected = false;
                    state.updateConnectionStatus();
                });
        }
        
        // Get services status - every poll (4 seconds)
        fetch('/api/services/status')
            .then(response => response.json())
            .then(data => {
                console.log('[DEBUG] Received services data:', data);
                if (typeof updateServicesStatusLive === 'function') {
                    updateServicesStatusLive(data);
                } else {
                    console.log('[DEBUG] updateServicesStatusLive not available - component not extracted yet');
                }
                state.isConnected = true;
                state.updateConnectionStatus();
            })
            .catch(error => {
                console.error('Error polling services status:', error);
                state.isConnected = false;
                state.updateConnectionStatus();
            });
            
        // Get logs - every 2 polls (8 seconds)
        if (state.pollCount % 2 === 0) {
            if (typeof updateLogs === 'function') {
                updateLogs();
            } else if (window.updateLogs) {
                window.updateLogs();
            }
        }
        
        // Get debug images - every 3 polls (12 seconds)
        if (state.pollCount % 3 === 0) {
            if (typeof updateDebugImages === 'function') {
                updateDebugImages();
            } else {
                console.log('[DEBUG] updateDebugImages not available - component not extracted yet');
            }
        }
        
        // Get configuration - every 5 polls (20 seconds)
        if (state.pollCount % 5 === 0) {
            if (typeof updateConfiguration === 'function') {
                updateConfiguration();
            } else {
                console.log('[DEBUG] updateConfiguration not available - component not extracted yet');
            }
        }
        
        // Get hardware info - every 6 polls (24 seconds)
        if (state.pollCount % 6 === 0) {
            if (typeof updateHardware === 'function') {
                updateHardware();
            } else {
                console.log('[DEBUG] updateHardware not available - component not extracted yet');
            }
        }
        
        // Reset counter to prevent overflow
        if (state.pollCount >= 60) state.pollCount = 0;
    }
    
    // Start polling system
    console.log('Starting REST polling for status updates');
    state.isConnected = true;
    state.updateConnectionStatus();
    
    // Initial update
    pollForUpdates();
    
    // Set up interval
    state.pollingInterval = setInterval(() => {
        if (!state.isUserInteracting) {
            pollForUpdates();
        } else {
            console.log('[DEBUG POLLING] Skipping poll - user is interacting');
        }
    }, state.POLLING_INTERVAL);
}

// Stop polling function
function stopDashboardPolling() {
    console.log('Stopping REST polling');
    const state = window.DashboardState;
    state.isConnected = false;
    state.updateConnectionStatus();
    if (state.pollingInterval) {
        clearInterval(state.pollingInterval);
        state.pollingInterval = null;
    }
}

// Track user interaction to pause polling
document.addEventListener('mousedown', function() {
    window.DashboardState.isUserInteracting = true;
});

document.addEventListener('mouseup', function() {
    window.DashboardState.isUserInteracting = false;
});

// Handle page visibility changes
document.addEventListener('visibilitychange', function() {
    if (document.hidden) {
        stopDashboardPolling();
    } else {
        startDashboardPolling();
    }
});

// Global functions needed by the dashboard
function openLabelStudio() {
    // Smart Label Studio opening based on user role and setup - EXACT from original
    const userRole = window.userPermissions?.user_role || 'unknown';
    
    // Check if automated setup exists
    fetch('/api/labelstudio/user-project')
        .then(response => response.json())
        .then(data => {
            if (data.success && data.project_url) {
                // Open user's specific project
                window.open(data.project_url, '_blank');
            } else {
                // Fallback - open Label Studio with instructions
                const popup = window.open('http://localhost:8200', '_blank');
                
                // Show instructions
                if (data.setup_required) {
                    alert(
                        'Label Studio Setup Required:\n\n' +
                        '1. First time? Run setup script first:\n' +
                        '   python setup_label_studio.py\n\n' +
                        '2. Or login to Label Studio manually:\n' +
                        '   - Admin: admin@clinical.local\n' +
                        '   - Researcher: researcher@clinical.local\n\n' +
                        '3. Look for your project: Clinical_' + userRole + '_workspace'
                    );
                }
            }
        })
        .catch(error => {
            console.error('Error fetching Label Studio project:', error);
            // Fallback - just open Label Studio
            window.open('http://localhost:8200', '_blank');
        });
}

console.log('[DEBUG] Dashboard.js loaded successfully');