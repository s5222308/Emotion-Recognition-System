/**
 * System Overview Component JavaScript
 * CRITICAL: These functions MUST preserve exact signatures and behavior
 * The main polling system calls these functions globally
 */

// System Overview Component Namespace
window.Dashboard = window.Dashboard || {};
window.Dashboard.SystemOverview = (function() {
    
    /**
     * Update system metrics display
     * CRITICAL: This function is called by main polling system every 4 polls (16 seconds)
     * Original function: updateMetricsLive() lines 1549-1554 in dashboard.html
     */
    function updateMetricsLive(data) {
        // EXACT same implementation as original
        document.getElementById('healthy-services').textContent = `${data.healthy_services}/${data.total_services}`;
        document.getElementById('total-requests').textContent = data.total_requests;
        document.getElementById('error-rate').textContent = `${data.error_rate.toFixed(1)}%`;
        document.getElementById('debug-images').textContent = data.debug_images_count;
    }
    
    /**
     * Update services status display (used for service counts in System Overview)
     * CRITICAL: This function is called by main polling system every poll (4 seconds)
     * Original function: updateServicesStatusLive() lines 1083+ in dashboard.html
     * 
     * NOTE: For System Overview component, we only need the service counts
     * The full service control functionality will be in the Services Control component
     */
    function updateServicesStatusLive(data) {
        // For System Overview, we only care about the service counts
        // The detailed service control functionality is handled by Services Control component
        console.log('[DEBUG] System Overview: updateServicesStatusLive called with', Object.keys(data).length, 'services');
        
        // Update healthy services count (this feeds into the metrics display)
        const healthyCount = Object.values(data).filter(service => service.status === 'healthy').length;
        const totalCount = Object.keys(data).length;
        
        // Store this data for metrics display
        if (window.DashboardState) {
            window.DashboardState.lastServicesUpdate = {
                healthy_services: healthyCount,
                total_services: totalCount
            };
        }
        
        // Note: The actual metrics display is updated by updateMetricsLive()
        // which gets the data from the /api/system/metrics endpoint
    }
    
    /**
     * Initialize System Overview component
     */
    function init() {
        console.log('[DEBUG] System Overview component initialized');
        
        // Set initial values
        document.getElementById('healthy-services').textContent = '-';
        document.getElementById('total-requests').textContent = '-';
        document.getElementById('error-rate').textContent = '-';
        document.getElementById('debug-images').textContent = '-';
        
        // Component is ready
        console.log('[DEBUG] System Overview component ready');
    }
    
    // Public interface
    return {
        init: init,
        updateMetricsLive: updateMetricsLive,
        updateServicesStatusLive: updateServicesStatusLive
    };
})();

// CRITICAL: Export functions globally for main polling system compatibility
// The main polling system expects these exact function names to exist globally
window.updateMetricsLive = window.Dashboard.SystemOverview.updateMetricsLive;
window.updateServicesStatusLive = window.Dashboard.SystemOverview.updateServicesStatusLive;

// Global initialization function for main dashboard to call
window.initializeSystemOverview = function() {
    if (window.Dashboard.SystemOverview) {
        window.Dashboard.SystemOverview.init();
        console.log('System Overview component initialized');
    }
};

// Initialize component when DOM is ready (fallback for standalone usage)
document.addEventListener('DOMContentLoaded', function() {
    if (window.Dashboard.SystemOverview && !window.initializeSystemOverview.called) {
        window.Dashboard.SystemOverview.init();
    }
});