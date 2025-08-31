/**
 * Global Dashboard State Module
 * CRITICAL: This module preserves all global state variables that components depend on
 * DO NOT MODIFY without checking ALL components for dependencies
 */

// Initialize global dashboard state
window.DashboardState = {
    // Polling system state
    pollCount: 0,
    isConnected: false,
    POLLING_INTERVAL: 4000, // 4 seconds
    pollingInterval: null,
    
    // User interaction state
    isUserInteracting: false,
    
    // Service state management
    servicesStarting: new Set(),
    servicesStopping: new Set(),
    previousServiceStates: {},
    
    // Component data caches
    lastMetricsUpdate: null,
    lastServicesUpdate: null,
    
    // Global utilities
    getCSRFToken: function() {
        const token = document.querySelector('meta[name="csrf-token"]');
        return token ? token.getAttribute('content') : null;
    },
    
    // Secure fetch wrapper with CSRF
    secureFetch: function(url, options = {}) {
        if (options.method === 'POST') {
            options.headers = options.headers || {};
            options.headers['X-CSRFToken'] = this.getCSRFToken();
        }
        return fetch(url, options);
    },
    
    // Connection status updater
    updateConnectionStatus: function() {
        const refreshIcon = document.getElementById('refresh-icon');
        if (refreshIcon) {
            refreshIcon.style.color = this.isConnected ? '#28a745' : '#dc3545';
            refreshIcon.textContent = this.isConnected ? '●' : '●';
        }
    },
    
    // Clock updater (preserving exact timing)
    updateClock: function() {
        const timeElement = document.getElementById('current-time');
        if (timeElement) {
            const now = new Date();
            timeElement.textContent = now.toLocaleTimeString();
        }
    }
};

// Global namespace for components
window.Dashboard = window.Dashboard || {};

// Initialize clock (preserved from original)
document.addEventListener('DOMContentLoaded', function() {
    // Start clock updates immediately
    setInterval(window.DashboardState.updateClock, 1000);
    window.DashboardState.updateClock();
    
    console.log('[DEBUG] Dashboard shared state initialized');
});