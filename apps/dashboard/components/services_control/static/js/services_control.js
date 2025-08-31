/**
 * Services Control Component JavaScript
 * CRITICAL: This preserves EXACT functionality and state management from original dashboard
 * All function signatures, timeout logic, and state handling must match original implementation
 */

// Namespace for this component
window.Dashboard = window.Dashboard || {};
window.Dashboard.ServicesControl = {};

(function() {
    'use strict';
    
    console.log('Loading Services Control component...');
    
    // CRITICAL: Timeout constants - MUST match original exactly
    const STARTING_TIMEOUT = 60000;  // 60 seconds - same as original
    const STOPPING_TIMEOUT = 30000;  // 30 seconds - same as original
    const HEALTH_CHECK_INTERVAL = 2000; // 2 seconds - reduce log spam
    const MAX_HEALTH_CHECK_ATTEMPTS = 30; // 30 attempts = 30 seconds
    const MAX_STOP_CHECK_ATTEMPTS = 30;   // 30 attempts = 30 seconds
    
    // Service order for consistent display - same as original
    const SERVICE_ORDER = ['ml_service', 'ml_backend', 'label_studio'];
    
    // CRITICAL: Reference global state - DO NOT create new Sets
    const state = {
        get servicesStarting() { return window.DashboardState.servicesStarting; },
        get servicesStopping() { return window.DashboardState.servicesStopping; },
        get previousServiceStates() { return window.DashboardState.previousServiceStates; },
        get isUserInteracting() { return window.DashboardState.isUserInteracting; },
        set isUserInteracting(value) { window.DashboardState.isUserInteracting = value; }
    };
    
    // CRITICAL: Service control functions - EXACT signatures from original
    function startService(serviceId) {
        console.log(`[START] User clicked start for ${serviceId}`);
        console.log(`[START] servicesStarting current state:`, Array.from(state.servicesStarting));
        
        // Check if service is already starting
        if (state.servicesStarting.has(serviceId)) {
            console.log(`[START] ${serviceId} already starting - ignoring duplicate click`);
            return;
        }
        
        // Add to starting services set with timestamp - EXACT logic from original
        state.servicesStarting.add(serviceId);
        if (!window.serviceStartTimes) window.serviceStartTimes = {};
        window.serviceStartTimes[serviceId] = Date.now();
        
        // Pause status updates during user interaction - same as original
        state.isUserInteracting = true;
        
        // IMMEDIATELY set button to "Starting..." - EXACT match to original
        const serviceCard = document.querySelector(`[data-service="${serviceId}"]`);
        if (serviceCard) {
            const startBtn = serviceCard.querySelector('[data-action="start"]');
            if (startBtn) {
                startBtn.textContent = 'Starting...';
                startBtn.disabled = true;
                startBtn.style.cssText = 'background: #ffc107; color: white; border: 1px solid #ffc107; cursor: not-allowed; padding: 5px 10px; margin-right: 5px;';
                console.log(`[BUTTON] Set "${serviceId}" button to "Starting..." immediately`);
            }
            
            // Set status circle to blinking green for starting
            const statusCircle = serviceCard.querySelector('.status-circle');
            if (statusCircle) {
                statusCircle.className = 'status-circle starting';
                statusCircle.setAttribute('data-status', 'starting');
                console.log(`[ANIMATION] ${serviceId} status circle set to blinking green (starting)`);
            }
        }
        
        console.log(`[START] Added ${serviceId} to servicesStarting set:`, Array.from(state.servicesStarting));
        
        // Make API call to start service - SIMPLIFIED approach
        window.DashboardState.secureFetch(`/api/services/start/${serviceId}`, {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            console.log(`[START] ${serviceId} API response:`, data);
            
            if (data.error) {
                console.error(`[START] ${serviceId} API error:`, data.error);
                // Reset button state on error
                state.servicesStarting.delete(serviceId);
                state.isUserInteracting = false;
                updateServiceCardButtons(serviceId, null, 'normal');
                return;
            }
            
            // Service start initiated - let natural polling handle the rest
            console.log(`[START] ${serviceId} start initiated - relying on natural polling for completion`);
            
            // Resume natural polling after 2 seconds to allow service to start
            setTimeout(() => {
                state.isUserInteracting = false;
                console.log(`[DEBUG] ${serviceId} resuming natural polling`);
            }, 2000);
            
        })
        .catch(error => {
            console.error(`Error starting ${serviceId}:`, error);
            
            // Remove from starting set on error - same as original
            state.servicesStarting.delete(serviceId);
            delete window.serviceStartTimes?.[serviceId];
            
            // Resume status updates on error
            state.isUserInteracting = false;
            removeServiceAnimation(serviceId, 'glow-starting');
            updateServiceCardButtons(serviceId, null, 'normal');
        });
    }
    
    function stopService(serviceId) {
        console.log(`[STOP] User clicked stop for ${serviceId}`);
        console.log(`[STOP] servicesStopping current state:`, Array.from(state.servicesStopping));
        
        // Check if service is already stopping
        if (state.servicesStopping.has(serviceId)) {
            console.log(`[STOP] ${serviceId} already stopping - ignoring duplicate click`);
            return;
        }
        
        // Add to stopping services set with timestamp - EXACT logic from original
        state.servicesStopping.add(serviceId);
        if (!window.serviceStopTimes) window.serviceStopTimes = {};
        window.serviceStopTimes[serviceId] = Date.now();
        
        // Pause status updates during user interaction
        state.isUserInteracting = true;
        
        // IMMEDIATELY set button to "Stopping..." - EXACT match to original
        const serviceCard = document.querySelector(`[data-service="${serviceId}"]`);
        if (serviceCard) {
            const stopBtn = serviceCard.querySelector('[data-action="stop"]');
            if (stopBtn) {
                stopBtn.textContent = 'Stopping...';
                stopBtn.disabled = true;
                stopBtn.style.cssText = 'background: #dc3545; color: white; border: 1px solid #dc3545; cursor: not-allowed; padding: 5px 10px; margin-right: 5px;';
                console.log(`[BUTTON] Set "${serviceId}" button to "Stopping..." immediately`);
            }
            
            // Set status circle to blinking red for stopping
            const statusCircle = serviceCard.querySelector('.status-circle');
            if (statusCircle) {
                statusCircle.className = 'status-circle stopping';
                statusCircle.setAttribute('data-status', 'stopping');
                console.log(`[ANIMATION] ${serviceId} status circle set to blinking red (stopping)`);
            }
        }
        
        console.log(`[STOP] Added ${serviceId} to servicesStopping set:`, Array.from(state.servicesStopping));
        
        // Make API call to stop service - SIMPLIFIED approach
        window.DashboardState.secureFetch(`/api/services/stop/${serviceId}`, {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            console.log(`[STOP] ${serviceId} API response:`, data);
            
            if (data.error) {
                console.error(`[STOP] ${serviceId} API error:`, data.error);
                // Reset button state on error
                state.servicesStopping.delete(serviceId);
                state.isUserInteracting = false;
                updateServiceCardButtons(serviceId, null, 'normal');
                return;
            }
            
            // Service stop initiated - let natural polling handle the rest
            console.log(`[STOP] ${serviceId} stop initiated - relying on natural polling for completion`);
            
            // Resume natural polling after 2 seconds to allow service to stop
            setTimeout(() => {
                state.isUserInteracting = false;
                console.log(`[DEBUG] ${serviceId} resuming natural polling`);
            }, 2000);
            
        })
        .catch(error => {
            console.error(`Error stopping ${serviceId}:`, error);
            
            // Remove from stopping set on error
            state.servicesStopping.delete(serviceId);
            delete window.serviceStopTimes?.[serviceId];
            
            // Resume status updates on error
            state.isUserInteracting = false;
            removeServiceAnimation(serviceId, 'glow-stopping');
            updateServiceCardButtons(serviceId, null, 'normal');
        });
    }
    
    function restartService(serviceId) {
        console.log(`[RESTART] User clicked restart for ${serviceId}`);
        
        // Disable restart button immediately - same as original
        const restartBtn = document.querySelector(`[data-service="${serviceId}"] [data-action="restart"]`);
        if (restartBtn) {
            restartBtn.disabled = true;
            restartBtn.innerHTML = 'Restarting...';
        }
        
        // Make API call to restart service - EXACT endpoint path from analysis
        window.DashboardState.secureFetch(`/api/services/restart/${serviceId}`, {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            console.log(`[RESTART] ${serviceId} API response:`, data);
            
            if (data.error) {
                console.error(`[RESTART] ${serviceId} API error:`, data.error);
                // Reset restart button on error
                if (restartBtn) {
                    restartBtn.disabled = false;
                    restartBtn.innerHTML = 'Restart';
                }
                return;
            }
            
            // Reset restart button after successful restart initiation
            if (restartBtn) {
                restartBtn.disabled = false;
                restartBtn.innerHTML = 'Restart';
            }
            
            // Trigger status update to refresh all services
            if (typeof window.updateServicesStatus === 'function') {
                window.updateServicesStatus();
            }
        })
        .catch(error => {
            console.error(`Error restarting ${serviceId}:`, error);
            
            // Reset restart button on error
            if (restartBtn) {
                restartBtn.disabled = false;
                restartBtn.innerHTML = 'Restart';
            }
        });
    }
    
    function startAllServices() {
        console.log('[START_ALL] Starting all services...');
        
        // Update button to show starting state
        const startAllBtn = document.getElementById('start-all-btn');
        if (startAllBtn) {
            startAllBtn.disabled = true;
            startAllBtn.textContent = 'Starting All...';
            startAllBtn.style.background = '#6c757d';
            startAllBtn.style.cursor = 'not-allowed';
        }
        
        // Get current services status first - same logic as original
        fetch('/api/services/status')
            .then(response => response.json())
            .then(data => {
                const servicesToStart = [];
                
                // Find services that need to be started - same logic as original
                SERVICE_ORDER.forEach(serviceId => {
                    if (data[serviceId]) {
                        const service = data[serviceId];
                        const isRunning = service.status === 'running' || service.status === 'healthy';
                        const isStarting = state.servicesStarting.has(serviceId);
                        
                        if (!isRunning && !isStarting) {
                            servicesToStart.push(serviceId);
                        }
                    }
                });
                
                console.log('[START_ALL] Services to start:', servicesToStart);
                
                if (servicesToStart.length === 0) {
                    console.log('[START_ALL] No services need starting');
                    updateStartAllButtonState(0);
                    return;
                }
                
                // Update button to show progress
                if (startAllBtn) {
                    startAllBtn.textContent = `Starting All (${servicesToStart.length})...`;
                }
                
                // Start services with staggered timing - same as original
                servicesToStart.forEach((serviceId, index) => {
                    setTimeout(() => {
                        console.log(`Starting ${serviceId}...`);
                        startService(serviceId);
                    }, index * 2000); // Stagger starts by 2 seconds
                });
            })
            .catch(error => {
                console.error('[START_ALL] Error getting services status:', error);
                // Reset button on error
                if (startAllBtn) {
                    startAllBtn.disabled = false;
                    startAllBtn.textContent = 'Start All';
                    startAllBtn.style.background = 'green';
                    startAllBtn.style.cursor = 'pointer';
                }
            });
    }
    
    // Update Start All button state - EXACT logic from original
    function updateStartAllButtonState(stoppedServiceCount) {
        const startAllBtn = document.getElementById('start-all-btn');
        if (!startAllBtn) return; // Button might not exist if user lacks permissions
        
        // Check if any services are currently starting
        const startingCount = state.servicesStarting.size;
        
        console.log('[START_ALL_BUTTON] Updating - stopped:', stoppedServiceCount, 'starting:', startingCount);
        
        if (startingCount > 0) {
            // Some services are starting - show progress
            startAllBtn.disabled = true;
            startAllBtn.textContent = `Starting All (${startingCount})...`;
            startAllBtn.style.background = '#6c757d';
            startAllBtn.style.cursor = 'not-allowed';
        } else if (stoppedServiceCount === 0) {
            // All services running - disable start all button
            startAllBtn.disabled = true;
            startAllBtn.style.background = '#6c757d';
            startAllBtn.style.color = '#ffffff80';
            startAllBtn.style.cursor = 'not-allowed';
            startAllBtn.textContent = 'All Running';
        } else {
            // Some services stopped - enable start all button
            startAllBtn.disabled = false;
            startAllBtn.style.background = 'green';
            startAllBtn.style.color = 'white';
            startAllBtn.style.cursor = 'pointer';
            startAllBtn.textContent = `Start All (${stoppedServiceCount})`;
        }
    }
    
    // CRITICAL: Button state management - EXACT logic from original
    function updateServiceCardButtons(serviceId, status, forceState = null) {
        const serviceCard = document.querySelector(`[data-service="${serviceId}"]`);
        if (!serviceCard) return;
        
        // Determine states - ENHANCED to handle race conditions
        const isInStartingSet = state.servicesStarting.has(serviceId);
        const isInStoppingSet = state.servicesStopping.has(serviceId);
        const isRunning = status === 'running' || status === 'healthy';
        const isStopped = status === 'stopped' || status === 'down' || status === 'unhealthy';
        const isApiStarting = status === 'starting';
        const isApiStopping = status === 'stopping';
        
        // FIXED: Consider both Set membership AND API status to handle race conditions
        const isStarting = isInStartingSet || isApiStarting;
        const isStopping = isInStoppingSet || isApiStopping;
        
        let buttonState = forceState || (isStarting ? 'starting' : (isStopping ? 'stopping' : 'normal'));
        
        console.log(`[BUTTONS] ${serviceId}: state=${buttonState}, starting=${isStarting}, stopping=${isStopping}, running=${isRunning}, status=${status}, servicesStarting=${Array.from(state.servicesStarting)}`);
        
        // Update start button
        const startBtn = serviceCard.querySelector('[data-action="start"]');
        if (startBtn) {
            switch (buttonState) {
                case 'starting':
                    startBtn.disabled = true;
                    startBtn.textContent = 'Starting...';
                    startBtn.style.background = '#6c757d';
                    startBtn.style.cursor = 'not-allowed';
                    break;
                case 'stopping':
                    startBtn.disabled = true;
                    startBtn.textContent = 'Start';
                    startBtn.style.background = '#6c757d';
                    startBtn.style.cursor = 'not-allowed';
                    break;
                default:
                    startBtn.disabled = isRunning;
                    startBtn.textContent = 'Start';
                    startBtn.style.background = isRunning ? '#6c757d' : '#28a745';
                    startBtn.style.cursor = isRunning ? 'not-allowed' : 'pointer';
            }
        }
        
        // Update stop button
        const stopBtn = serviceCard.querySelector('[data-action="stop"]');
        if (stopBtn) {
            switch (buttonState) {
                case 'starting':
                    stopBtn.disabled = true;
                    stopBtn.textContent = 'Stop';
                    stopBtn.style.background = '#6c757d';
                    stopBtn.style.cursor = 'not-allowed';
                    break;
                case 'stopping':
                    stopBtn.disabled = true;
                    stopBtn.textContent = 'Stopping...';
                    stopBtn.style.background = '#6c757d';
                    stopBtn.style.cursor = 'not-allowed';
                    break;
                default:
                    stopBtn.disabled = !isRunning;
                    stopBtn.textContent = 'Stop';
                    stopBtn.style.background = isRunning ? '#dc3545' : '#6c757d';
                    stopBtn.style.cursor = isRunning ? 'pointer' : 'not-allowed';
            }
        }
        
        // Update restart button  
        const restartBtn = serviceCard.querySelector('[data-action="restart"]');
        if (restartBtn && restartBtn.textContent !== 'Restarting...') {
            switch (buttonState) {
                case 'starting':
                case 'stopping':
                    restartBtn.disabled = true;
                    restartBtn.style.background = '#6c757d';
                    restartBtn.style.cursor = 'not-allowed';
                    break;
                default:
                    restartBtn.disabled = !isRunning;
                    restartBtn.style.background = isRunning ? '#007bff' : '#6c757d';
                    restartBtn.style.cursor = isRunning ? 'pointer' : 'not-allowed';
            }
        }
    }
    
    // Animation management functions
    function addServiceAnimation(serviceId, animationClass) {
        const serviceCard = document.querySelector(`[data-service="${serviceId}"] .service-card`);
        if (serviceCard) {
            serviceCard.classList.add(animationClass);
            console.log(`[ANIMATION] Added ${animationClass} to ${serviceId}`);
        }
    }
    
    function removeServiceAnimation(serviceId, animationClass) {
        const serviceCard = document.querySelector(`[data-service="${serviceId}"] .service-card`);
        if (serviceCard) {
            serviceCard.classList.remove(animationClass);
            console.log(`[ANIMATION] Removed ${animationClass} from ${serviceId}`);
        }
    }
    
    // CRITICAL: Service Card Rendering Function - EXACT from original
    function updateServicesStatusLive(data) {
        const servicesRow = document.getElementById('services-row');
        if (!servicesRow) {
            console.error('[ERROR] services-row element not found!');
            return;
        }
        
        const isFirstLoad = !servicesRow.hasChildNodes();
        
        console.log('[DEBUG] updateServicesStatusLive called, isFirstLoad:', isFirstLoad, 'isUserInteracting:', state.isUserInteracting, 'servicesData:', Object.keys(data));
        
        // Clean up stale stopping states on any load - EXACT match to original
        state.servicesStopping.forEach(serviceId => {
            const service = data[serviceId];
            if (service && (service.status === 'down' || service.status === 'stopped')) {
                console.log(`[CLEANUP] Removing ${serviceId} from stopping set - service is already stopped`);
                state.servicesStopping.delete(serviceId);
            }
        });
        
        if (state.isUserInteracting && !isFirstLoad) {
            // During user interactions, skip the update to preserve button states  
            console.log('[DEBUG] Skipping service update during user interaction');
            return;
        }
        
        if (!isFirstLoad && !state.isUserInteracting) {
            // Check if service cards actually exist - if not, force recreation
            const existingCards = servicesRow.querySelectorAll('[data-service]');
            if (existingCards.length > 0) {
                // Cards exist - do selective update
                console.log('[DEBUG] Using selective update for existing cards');
                updateExistingServiceCards(data);
                return;
            } else {
                // Cards don't exist - force recreation
                console.log('[DEBUG] No existing cards found, forcing recreation');
            }
        }
        
        // First load or forced refresh - create all cards
        console.log('[DEBUG] Creating service cards from scratch');
        servicesRow.innerHTML = '';
        
        // CRITICAL: Apply state clearing logic before creating cards - same as selective update
        Object.entries(data).forEach(([serviceId, service]) => {
            const isHealthy = service.status === 'running' || service.status === 'healthy';
            const isStopped = service.status === 'down' || service.status === 'stopped';
            const isStarting = state.servicesStarting.has(serviceId);
            const isStopping = state.servicesStopping.has(serviceId);
            
            // Clear starting state when status becomes healthy - EXACT from original
            if (isStarting && isHealthy) {
                console.log(`[STARTING CLEARED] ${serviceId} status became healthy - clearing starting state`);
                state.servicesStarting.delete(serviceId);
                delete window.serviceStartTimes?.[serviceId];
            }
            
            // Clear stopping state when status becomes stopped - EXACT from original  
            if (isStopping && isStopped) {
                console.log(`[STOPPING CLEARED] ${serviceId} status became stopped - clearing stopping state`);
                state.servicesStopping.delete(serviceId);
                delete window.serviceStopTimes?.[serviceId];
            }
        });
        
        let stoppedServices = [];
        
        // Define consistent service order to prevent jumping - EXACT from original
        const serviceOrder = ['ml_service', 'ml_backend', 'label_studio'];
        
        // Process services in consistent order
        serviceOrder.forEach(serviceId => {
            if (!data[serviceId]) return; // Skip if service not found
            
            const service = data[serviceId];
            
            // Track stopped services - more comprehensive status check
            const isRunning = service.status === 'running' || service.status === 'healthy';
            const isStarting = state.servicesStarting.has(serviceId);
            
            if (!isRunning && !isStarting) {
                stoppedServices.push(serviceId);
                console.log(`[DEBUG] Service ${serviceId} added to stoppedServices - status: ${service.status}, isRunning: ${isRunning}, isStarting: ${isStarting}`);
            }
            
            // Handle uptime calculation - support both uptime_seconds and uptime fields
            let uptime = 'Unknown';
            if (service.uptime_seconds && service.uptime_seconds > 0) {
                // Original API format
                uptime = `${Math.floor(service.uptime_seconds / 60)}m ${service.uptime_seconds % 60}s`;
            } else if (service.uptime) {
                // Modular API format - calculate seconds from uptime timestamp
                const uptimeDate = new Date(service.uptime);
                const uptimeSeconds = Math.floor((Date.now() - uptimeDate.getTime()) / 1000);
                if (uptimeSeconds > 0) {
                    uptime = `${Math.floor(uptimeSeconds / 60)}m ${uptimeSeconds % 60}s`;
                }
            }
            
            // Build service controls HTML - CRITICAL: Check user permissions
            let serviceControlsHTML = '';
            const userPermissions = window.userPermissions || {};
            
            if (userPermissions.control_services) {
                // Dynamic button states based on service status - EXACT logic from original
                const isRunning = service.status === 'running' || service.status === 'healthy';
                const isStopped = service.status === 'stopped' || service.status === 'down' || service.status === 'unhealthy' || service.status === 'stopping';
                const isStarting = state.servicesStarting.has(serviceId);
                
                // Debug logging for button states
                if (serviceId === 'ml_service' || serviceId === 'ml_backend') {
                    console.log(`[DEBUG] ${serviceId} button state: running=${isRunning}, starting=${isStarting}, status=${service.status}, servicesStarting=${Array.from(state.servicesStarting)}`);
                }
                
                // Start button - ALWAYS prioritize starting state over service status (EXACT as original)
                let startButtonStyle, startButtonText, startButtonDisabled;
                if (isStarting) {
                    // Service is starting - always show starting regardless of current status
                    startButtonStyle = 'background: #ffc107; color: white; border: 1px solid #ffc107; cursor: not-allowed;';
                    startButtonText = 'Starting...';
                    startButtonDisabled = 'disabled';
                } else if (isRunning) {
                    // Service is running - disable start button  
                    startButtonStyle = 'background: #6c757d; color: #ffffff80; border: 1px solid #6c757d; cursor: not-allowed;';
                    startButtonText = 'Start';
                    startButtonDisabled = 'disabled';
                } else {
                    // Service is stopped - enable start button
                    startButtonStyle = 'background: green; color: white; border: 1px solid green; cursor: pointer;';
                    startButtonText = 'Start';
                    startButtonDisabled = '';
                }
                
                // Stop button - enabled only when service is running (RED theme to match CSS)
                const stopButtonStyle = isRunning ?
                    'background: #dc3545; color: white; border: 1px solid #dc3545; cursor: pointer;' :
                    'background: #6c757d; color: #ffffff80; border: 1px solid #6c757d; cursor: not-allowed;';
                
                // Restart button - always enabled (BLUE theme to match CSS)
                const restartButtonStyle = 'background: #007bff; color: white; border: 1px solid #007bff; cursor: pointer;';
                
                // Open button - enabled only when service is running
                const openButtonStyle = isRunning ?
                    'background: #6f42c1; color: white; border: 1px solid #6f42c1; cursor: pointer;' :
                    'background: #6c757d; color: #ffffff80; border: 1px solid #6c757d; cursor: not-allowed;';
                
                serviceControlsHTML = `
                    <button onclick="startService('${serviceId}')" 
                            data-action="start"
                            ${startButtonDisabled} 
                            style="${startButtonStyle} padding: 5px 10px; margin-right: 5px;">
                        ${startButtonText}
                    </button>
                    <button onclick="stopService('${serviceId}')" 
                            data-action="stop"
                            ${isRunning ? '' : 'disabled'} 
                            style="${stopButtonStyle} padding: 5px 10px; margin-right: 5px;">
                        Stop
                    </button>
                    <button onclick="restartService('${serviceId}')" 
                            data-action="restart"
                            style="${restartButtonStyle} padding: 5px 10px; margin-right: 5px;">
                        Restart
                    </button>
                    <button onclick="window.open('${serviceId === 'label_studio' ? service.base_url : service.base_url + '/status'}', '_blank')" 
                            data-action="open"
                            ${isRunning ? '' : 'disabled'}
                            style="${openButtonStyle} padding: 5px 10px; margin-left: 5px;">
                        Open
                    </button>
                `;
            } else {
                serviceControlsHTML = '<span style="color: gray; font-size: 0.9em;">Service controls restricted to administrators</span>';
            }
            
            // Create service card HTML - EXACT structure from original
            servicesRow.innerHTML += `
                <div style="flex: 1; min-width: 300px; margin-bottom: 15px;" data-service="${serviceId}">
                    <div class="service-card">
                        <h5>
                            <span class="status-circle ${service.status}" data-status="${service.status}"></span>
                            ${service.name}
                        </h5>
                        <p style="color: gray; font-size: 0.9em;">${service.description}</p>
                        <div style="display: flex; justify-content: space-around; margin-bottom: 10px;">
                            <div style="text-align: center;">
                                <small style="color: gray;">Uptime</small><br>
                                <strong>${uptime}</strong>
                            </div>
                            <div style="text-align: center;">
                                <small style="color: gray;">Errors</small><br>
                                <strong>${service.errors}</strong>
                            </div>
                        </div>
                        <div class="service-controls" style="text-align: center;">
                            ${serviceControlsHTML}
                        </div>
                    </div>
                </div>
            `;
        });
        
        // stoppedServices already populated in the loop above
        
        // Update "Start All" button state - same logic as original
        updateStartAllButtonState(stoppedServices.length);
        
        // Check if setup wizard needs to be hidden due to Label Studio status
        if (typeof window.checkSetupWizardStatusSmart === 'function') {
            window.checkSetupWizardStatusSmart(data);
        }
        
        console.log('[DEBUG] Service cards created successfully');
    }
    
    // Selective update function for efficiency - EXACT from original
    function updateExistingServiceCards(data) {
        console.log('[DEBUG UPTIME] updateExistingServiceCards called with data:', data);
        
        Object.entries(data).forEach(([serviceId, service]) => {
            const serviceCard = document.querySelector(`[data-service="${serviceId}"]`);
            if (!serviceCard) return;
            
            // Update status circle
            const statusCircle = serviceCard.querySelector('.status-circle');
            if (statusCircle) {
                statusCircle.className = `status-circle ${service.status}`;
                statusCircle.setAttribute('data-status', service.status);
            }
            
            // CRITICAL: Update uptime display in live updates - find the uptime element specifically
            const uptimeElements = serviceCard.querySelectorAll('strong');
            const uptimeElement = uptimeElements[0]; // First strong element is uptime
            if (uptimeElement) {
                let uptime = 'Unknown';
                if (service.uptime_seconds && service.uptime_seconds > 0) {
                    // Original API format
                    uptime = `${Math.floor(service.uptime_seconds / 60)}m ${service.uptime_seconds % 60}s`;
                } else if (service.uptime) {
                    // Modular API format - calculate seconds from uptime timestamp
                    const uptimeDate = new Date(service.uptime);
                    const uptimeSeconds = Math.floor((Date.now() - uptimeDate.getTime()) / 1000);
                    if (uptimeSeconds > 0) {
                        uptime = `${Math.floor(uptimeSeconds / 60)}m ${uptimeSeconds % 60}s`;
                    }
                }
                console.log(`[DEBUG UPTIME] Updating ${serviceId} uptime from "${uptimeElement.textContent}" to "${uptime}"`);
                uptimeElement.textContent = uptime;
            }
            
            // Handle status-based state transitions - EXACT match to original logic
            const isHealthy = service.status === 'running' || service.status === 'healthy';
            const isStopped = service.status === 'down' || service.status === 'stopped';
            const isStarting = state.servicesStarting.has(serviceId);
            const isStopping = state.servicesStopping.has(serviceId);
            
            // Clear starting state when status becomes healthy
            if (isStarting && isHealthy) {
                console.log(`[STARTING CLEARED] ${serviceId} status became healthy - clearing starting state`);
                state.servicesStarting.delete(serviceId);
                delete window.serviceStartTimes?.[serviceId];
            }
            
            // Clear stopping state when status becomes stopped
            if (isStopping && isStopped) {
                console.log(`[STOPPING CLEARED] ${serviceId} status became stopped - clearing stopping state`);
                state.servicesStopping.delete(serviceId);
                delete window.serviceStopTimes?.[serviceId];
            }
            
            // Update service card buttons
            updateServiceCardButtons(serviceId, service.status);
        });
        
        // CRITICAL: Update Start All button after selective updates - this was missing!
        const stoppedServices = [];
        SERVICE_ORDER.forEach(serviceId => {
            if (data[serviceId]) {
                const service = data[serviceId];
                const isStopped = service.status === 'stopped' || service.status === 'down' || service.status === 'unhealthy';
                if (isStopped && !state.servicesStarting.has(serviceId)) {
                    stoppedServices.push(serviceId);
                }
            }
        });
        updateStartAllButtonState(stoppedServices.length);
        
        // Check if setup wizard needs to be hidden due to Label Studio status
        if (typeof window.checkSetupWizardStatusSmart === 'function') {
            window.checkSetupWizardStatusSmart(data);
        }
    }
    
    // Main update function that fetches data and renders cards
    function updateServicesStatus() {
        console.log('[DEBUG] updateServicesStatus called - forcing full recreation');
        console.log('[DEBUG] About to fetch /api/services/status');
        fetch('/api/services/status')
            .then(response => response.json())
            .then(data => {
                console.log('[DEBUG] updateServicesStatus got data:', data);
                
                // Clean up stale starting states on initial load
                state.servicesStarting.forEach(serviceId => {
                    const service = data[serviceId];
                    if (service && (service.status === 'running' || service.status === 'healthy')) {
                        console.log(`[CLEANUP] Removing ${serviceId} from starting set - service is already healthy`);
                        state.servicesStarting.delete(serviceId);
                    }
                });
                
                // Clean up stale stopping states on initial load
                state.servicesStopping.forEach(serviceId => {
                    const service = data[serviceId];
                    if (service && (service.status === 'down' || service.status === 'stopped')) {
                        console.log(`[CLEANUP] Removing ${serviceId} from stopping set - service is already stopped`);
                        state.servicesStopping.delete(serviceId);
                    }
                });
                
                // Force full recreation by clearing the services row first
                const servicesRow = document.getElementById('services-row');
                if (servicesRow) {
                    console.log('[DEBUG] Clearing services row for forced recreation');
                    servicesRow.innerHTML = '';
                    state.isUserInteracting = false; // Make sure we're not blocked
                }
                
                updateServicesStatusLive(data);
            })
            .catch(error => console.error('[ERROR] updateServicesStatus failed:', error));
    }

    // Store functions in component namespace
    window.Dashboard.ServicesControl = {
        startService,
        stopService,
        restartService,
        startAllServices,
        updateStartAllButtonState,
        updateServiceCardButtons,
        addServiceAnimation,
        removeServiceAnimation,
        updateServicesStatus,
        updateServicesStatusLive
    };
    
    // CRITICAL: Export global functions for template onclick handlers
    window.startService = startService;
    window.stopService = stopService;
    window.restartService = restartService;
    window.startAllServices = startAllServices;
    
    // CRITICAL: Export service status update functions for dashboard integration
    window.updateServicesStatus = updateServicesStatus;
    window.updateServicesStatusLive = updateServicesStatusLive;
    
    console.log('Services Control component loaded successfully');
    
})();

// Initialize function for the component
function initializeServicesControl() {
    console.log('Initializing Services Control component...');
    console.log('[DEBUG] Container exists:', document.getElementById('services-row'));
    
    // CRITICAL: Call updateServicesStatus to populate service cards immediately
    setTimeout(() => {
        console.log('[DEBUG] Calling updateServicesStatus to populate service cards...');
        updateServicesStatus();
    }, 100); // Small delay to ensure DOM is ready
    
    console.log('Services Control component initialized');
}

// Export initialization function
window.initializeServicesControl = initializeServicesControl;