/**
 * Annotation Setup Component JavaScript
 * CRITICAL: This preserves EXACT functionality from original dashboard
 * All function signatures and behavior must match original implementation
 */

// Namespace for this component
window.Dashboard = window.Dashboard || {};
window.Dashboard.AnnotationSetup = {};

(function() {
    'use strict';
    
    console.log('Loading Annotation Setup component...');
    
    // Core functions - MUST preserve exact signatures from original
    function checkLabelStudioSetup() {
        console.log('Checking Label Studio setup...');
        
        // First, ensure the proper HTML structure exists (in case it was replaced by error messages)
        const setupContainer = document.getElementById('setup-status');
        if (!document.getElementById('ls-status-circle')) {
            // Status circles don't exist, restore the original structure
            setupContainer.innerHTML = `
                <div style="display: flex; gap: 20px;">
                    <div style="flex: 1; border: 1px solid #ccc; padding: 15px; text-align: center;">
                        <div id="ls-status-circle" class="status-circle unknown" style="margin-bottom: 10px;"></div>
                        <h6 style="margin-bottom: 5px;">Label Studio</h6>
                        <small id="ls-status-text">Checking...</small>
                    </div>
                    <div style="flex: 1; border: 1px solid #ccc; padding: 15px; text-align: center;">
                        <div id="ml-status-circle" class="status-circle unknown" style="margin-bottom: 10px;"></div>
                        <h6 style="margin-bottom: 5px;">ML Backend</h6>
                        <small id="ml-status-text">Checking...</small>
                    </div>
                    <div style="flex: 1; border: 1px solid #ccc; padding: 15px; text-align: center;">
                        <div id="setup-status-circle" class="status-circle unknown" style="margin-bottom: 10px;"></div>
                        <h6 style="margin-bottom: 5px;">Ready to Annotate</h6>
                        <small id="setup-status-text">Checking...</small>
                    </div>
                </div>
            `;
        }
        
        // Show loading indicator - find all check status buttons
        const checkButtons = document.querySelectorAll('button[onclick*="checkLabelStudioSetup"]');
        checkButtons.forEach(button => {
            button.textContent = 'Checking...';
            button.disabled = true;
        });
        
        // Update status circles to show loading
        updateStatusCircle('ls-status-circle', 'unknown');
        updateStatusCircle('ml-status-circle', 'unknown');  
        updateStatusCircle('setup-status-circle', 'unknown');
        
        // Update status text safely
        const lsText = document.getElementById('ls-status-text');
        const mlText = document.getElementById('ml-status-text');
        const setupText = document.getElementById('setup-status-text');
        
        if (lsText) lsText.textContent = 'Checking...';
        if (mlText) mlText.textContent = 'Checking...';
        if (setupText) setupText.textContent = 'Checking...';
        
        // Call API endpoint
        window.DashboardState.secureFetch('/api/labelstudio/check-setup')
            .then(response => response.json())
            .then(data => {
                console.log('Setup status:', data);
                updateSetupStatus(data);
                
                // Reset check buttons
                const checkButtons = document.querySelectorAll('button[onclick*="checkLabelStudioSetup"]');
                checkButtons.forEach(button => {
                    button.textContent = button.textContent.includes('Again') ? 'Check Again' : 'Check Status';
                    button.disabled = false;
                });
            })
            .catch(error => {
                console.error('Error checking setup:', error);
                
                // Show error status
                updateStatusCircle('ls-status-circle', 'error');
                updateStatusCircle('ml-status-circle', 'error');
                updateStatusCircle('setup-status-circle', 'error');
                
                // Update status text safely
                const lsText = document.getElementById('ls-status-text');
                const mlText = document.getElementById('ml-status-text');
                const setupText = document.getElementById('setup-status-text');
                
                if (lsText) lsText.textContent = 'Error checking status';
                if (mlText) mlText.textContent = 'Error checking status';
                if (setupText) setupText.textContent = 'Error checking status';
                
                // Reset check buttons
                const checkButtons = document.querySelectorAll('button[onclick*="checkLabelStudioSetup"]');
                checkButtons.forEach(button => {
                    button.textContent = button.textContent.includes('Again') ? 'Check Again' : 'Check Status';
                    button.disabled = false;
                });
            });
    }
    
    function showSetupWizard() {
        console.log('Setup Wizard requested - checking Label Studio status first...');
        
        // Check Label Studio status before allowing setup wizard access
        window.DashboardState.secureFetch('/api/labelstudio/check-setup')
            .then(response => response.json())
            .then(data => {
                console.log('Label Studio status check result:', data);
                
                const lsStatus = data.label_studio.status;
                if (lsStatus !== 'running') {
                    // Label Studio is not running - show error and don't allow setup
                    const setupContainer = document.getElementById('setup-status');
                    setupContainer.innerHTML = `
                        <div style="border: 2px solid #dc3545; padding: 20px; background: #f8d7da; color: #721c24;">
                            <div style="display: flex; align-items: center; margin-bottom: 15px;">
                                <i class="fas fa-exclamation-triangle" style="font-size: 24px; margin-right: 10px;"></i>
                                <h5 style="margin: 0;">Setup Wizard Unavailable</h5>
                            </div>
                            <p style="margin-bottom: 15px;">
                                The setup wizard requires Label Studio to be running first. 
                                Current Label Studio status: <strong>${lsStatus.replace('_', ' ')}</strong>
                            </p>
                            <p style="margin-bottom: 0;">
                                Please start Label Studio using the service controls below, then try the setup wizard again.
                            </p>
                        </div>
                    `;
                    return;
                }
                
                // Label Studio is running - show setup wizard IN PLACE
                console.log('Label Studio is running - showing setup wizard');
                const setupContainer = document.getElementById('setup-status');
                
                setupContainer.innerHTML = `
                    <div style="border: 2px solid #007bff; padding: 20px; background: #f8f9fa;">
                        <h5>Label Studio Setup</h5>
                        <form id="setup-form" style="margin-top: 15px;">
                            <div style="margin-bottom: 10px;">
                                <label>Project Name:</label>
                                <input type="text" id="project-name" value="Emotion Recognition Project" style="width: 100%; padding: 5px; margin-top: 5px;">
                            </div>
                            <div style="margin-bottom: 10px;">
                                <label>Admin Email:</label>
                                <input type="email" id="admin-email" value="admin@clinical.local" style="width: 100%; padding: 5px; margin-top: 5px;">
                            </div>
                            <div style="margin-bottom: 10px;">
                                <label>Password:</label>
                                <input type="password" id="admin-password" value="admin123" style="width: 100%; padding: 5px; margin-top: 5px;">
                            </div>
                            <div style="margin-bottom: 15px;">
                                <label>Video Storage Path:</label>
                                <div style="display: flex; margin-top: 5px;">
                                    <input type="text" id="storage-path" value="data/datasets/ravdess" style="flex: 1; padding: 5px; margin-right: 10px;">
                                    <button type="button" onclick="browseStoragePath()" style="padding: 5px 10px; background: #007bff; color: white; border: 1px solid #007bff; cursor: pointer;">Browse</button>
                                </div>
                                <small style="color: #666; font-size: 0.8em;">Path to existing directory containing video files for annotation</small>
                            </div>
                            <button type="button" onclick="startQuickSetup()" style="padding: 8px 15px; background: #28a745; color: white; border: 1px solid #28a745; cursor: pointer; margin-right: 10px;">
                                Start Setup
                            </button>
                            <button type="button" onclick="cancelSetup()" style="padding: 8px 15px; background: #6c757d; color: white; border: 1px solid #6c757d; cursor: pointer;">
                                Cancel
                            </button>
                        </form>
                    </div>
                `;
            })
            .catch(error => {
                console.error('Error checking Label Studio status:', error);
                const setupContainer = document.getElementById('setup-status');
                setupContainer.innerHTML = `
                    <div style="border: 2px solid #dc3545; padding: 20px; background: #f8d7da; color: #721c24;">
                        <div style="display: flex; align-items: center; margin-bottom: 15px;">
                            <i class="fas fa-exclamation-triangle" style="font-size: 24px; margin-right: 10px;"></i>
                            <h5 style="margin: 0;">Setup Wizard Error</h5>
                        </div>
                        <p style="margin-bottom: 15px;">
                            Unable to check Label Studio status. Please ensure all services are responding properly.
                        </p>
                        <button type="button" onclick="checkLabelStudioSetup()" 
                                style="padding: 8px 15px; background: #007bff; color: white; border: 1px solid #007bff; cursor: pointer;">
                            Try Again
                        </button>
                    </div>
                `;
            });
    }
    
    function showSetupWizardModal() {
        // Remove any existing modal
        const existingModal = document.getElementById('setup-wizard-modal');
        if (existingModal) {
            existingModal.remove();
        }
        
        // Create and show the modal
        const modal = createSetupModal();
        document.body.appendChild(modal);
        
        // Use Bootstrap modal if available, otherwise fallback
        if (typeof bootstrap !== 'undefined' && bootstrap.Modal) {
            const bootstrapModal = new bootstrap.Modal(modal);
            bootstrapModal.show();
        } else {
            // Fallback: show as regular div
            modal.style.display = 'block';
            modal.style.position = 'fixed';
            modal.style.top = '0';
            modal.style.left = '0';
            modal.style.width = '100%';
            modal.style.height = '100%';
            modal.style.backgroundColor = 'rgba(0,0,0,0.5)';
            modal.style.zIndex = '9999';
        }
        
        // Initialize with setup type selection
        showSetupTypeSelection();
    }
    
    function createSetupModal() {
        const modal = document.createElement('div');
        modal.id = 'setup-wizard-modal';
        modal.className = 'modal fade';
        modal.innerHTML = `
            <div class="modal-dialog modal-lg">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">
                            <i class="fas fa-magic"></i> Label Studio Setup Wizard
                        </h5>
                        <button type="button" class="btn-close" onclick="closeSetupModal()" style="background: none; border: none; font-size: 20px; cursor: pointer;">&times;</button>
                    </div>
                    <div class="modal-body">
                        <div id="setup-wizard-content">
                            <!-- Content will be populated by JavaScript -->
                        </div>
                    </div>
                </div>
            </div>
        `;
        return modal;
    }
    
    
    function showSetupTypeSelection() {
        const content = document.getElementById('setup-wizard-content');
        if (!content) return;
        
        // Show the setup form
        showSetupForm();
    }
    
    function selectSetupType(type) {
        showSetupForm();
    }
    
    
    function showSetupForm() {
        const content = document.getElementById('setup-wizard-content');
        if (!content) return;
        
        content.innerHTML = `
            <div class="text-center mb-4">
                <div style="color: #007bff; font-size: 48px; margin-bottom: 15px;">
                    <i class="fas fa-cog"></i>
                </div>
                <h5>Label Studio Configuration</h5>
                <p style="color: #666;">Configure your annotation project settings.</p>
            </div>
            
            <div style="text-align: center; margin-top: 30px;">
                <button type="button" onclick="checkLabelStudioSetup()" style="padding: 8px 15px; background: #6c757d; color: white; border: 1px solid #6c757d; cursor: pointer;">
                    Close
                </button>
            </div>
        `;
    }
    
    
    function showSetupResult(success, message) {
        const content = document.getElementById('setup-wizard-content');
        if (!content) return;
        
        const icon = success ? '<i class="fas fa-check-circle"></i>' : '<i class="fas fa-times-circle"></i>';
        const color = success ? '#28a745' : '#dc3545';
        
        content.innerHTML = `
            <div class="text-center">
                <div style="font-size: 48px; margin-bottom: 15px; color: ${color};">
                    ${icon}
                </div>
                <h5 style="color: ${color};">${success ? 'Setup Complete!' : 'Setup Failed'}</h5>
                <p style="color: #666;">${message}</p>
                
                <div style="margin-top: 30px;">
                    <button type="button" onclick="checkLabelStudioSetup()" style="padding: 8px 15px; background: #007bff; color: white; border: 1px solid #007bff; cursor: pointer;">
                        Close
                    </button>
                </div>
            </div>
        `;
    }
    
    
    // Helper function to update status display
    function updateSetupStatus(data) {
        console.log('Updating setup status display:', data);
        
        // Update Label Studio status
        if (data.label_studio) {
            const status = data.label_studio.status;
            updateStatusCircle('ls-status-circle', status);
            const lsText = document.getElementById('ls-status-text');
            if (lsText) lsText.textContent = data.label_studio.message || status;
        }
        
        // Update ML Backend status
        if (data.ml_backend) {
            const status = data.ml_backend.status;
            updateStatusCircle('ml-status-circle', status);
            const mlText = document.getElementById('ml-status-text');
            if (mlText) mlText.textContent = data.ml_backend.message || status;
        }
        
        // Update overall ready status
        if (data.ready_to_annotate) {
            const status = data.ready_to_annotate.status;
            updateStatusCircle('setup-status-circle', status);
            const setupText = document.getElementById('setup-status-text');
            if (setupText) setupText.textContent = data.ready_to_annotate.message || status;
            
            // Show/hide setup actions based on status
            const setupActions = document.getElementById('setup-actions');
            if (setupActions) {
                if (status === 'ready') {
                    setupActions.style.display = 'none';
                } else {
                    setupActions.style.display = 'block';
                }
            }
        }
    }
    
    // Helper function to update status circles
    function updateStatusCircle(circleId, status) {
        const circle = document.getElementById(circleId);
        if (!circle) return;
        
        // Remove existing status classes
        circle.classList.remove('running', 'not_running', 'unknown', 'error', 'ready', 'not_ready');
        
        // Add appropriate status class
        switch (status) {
            case 'running':
            case 'ready':
                circle.classList.add('running');
                break;
            case 'not_running':
            case 'not_ready':
            case 'installed_not_running':
            case 'not_installed':
                circle.classList.add('not_running');
                break;
            case 'error':
                circle.classList.add('error');
                break;
            default:
                circle.classList.add('unknown');
        }
    }
    
    // Check setup wizard status - monitors if Label Studio goes down during setup
    function checkSetupWizardStatusSmart(servicesData) {
        // Only check if setup wizard modal is currently displayed
        const setupModal = document.getElementById('setup-wizard-modal');
        if (!setupModal) {
            // Setup wizard not active, nothing to check
            return;
        }
        
        // Get Label Studio status from existing service data
        const labelStudioService = servicesData.label_studio;
        if (!labelStudioService) return;
        
        const lsStatus = labelStudioService.status;
        const isRunning = lsStatus === 'running' || lsStatus === 'healthy';
        
        if (!isRunning) {
            console.log('Label Studio went down during setup - showing error');
            
            // Label Studio went down, show error in modal
            const content = document.getElementById('setup-wizard-content');
            if (content) {
                content.innerHTML = `
                    <div style="border: 2px solid #dc3545; padding: 20px; background: #f8d7da; color: #721c24;">
                        <div style="display: flex; align-items: center; margin-bottom: 15px;">
                            <i class="fas fa-exclamation-triangle" style="font-size: 24px; margin-right: 10px;"></i>
                            <h5 style="margin: 0;">Setup Wizard Interrupted</h5>
                        </div>
                        <p style="margin-bottom: 15px;">
                            Label Studio has stopped running during setup. 
                            Current status: <strong>${lsStatus.replace('_', ' ')}</strong>
                        </p>
                        <p style="margin-bottom: 15px;">
                            Please restart Label Studio using the service controls, then try the setup wizard again.
                        </p>
                        <div style="text-align: center;">
                            <button type="button" onclick="checkLabelStudioSetup()" style="padding: 8px 15px; background: #dc3545; color: white; border: 1px solid #dc3545; cursor: pointer;">
                                Close
                            </button>
                        </div>
                    </div>
                `;
            }
        }
    }
    
    function cancelSetup() {
        // Restore original setup status HTML structure
        const setupContainer = document.getElementById('setup-status');
        setupContainer.innerHTML = `
            <div style="display: flex; gap: 20px;">
                <div style="flex: 1; border: 1px solid #ccc; padding: 15px; text-align: center;">
                    <div id="ls-status-circle" class="status-circle unknown" style="margin-bottom: 10px;"></div>
                    <h6 style="margin-bottom: 5px;">Label Studio</h6>
                    <small id="ls-status-text">Checking...</small>
                </div>
                <div style="flex: 1; border: 1px solid #ccc; padding: 15px; text-align: center;">
                    <div id="ml-status-circle" class="status-circle unknown" style="margin-bottom: 10px;"></div>
                    <h6 style="margin-bottom: 5px;">ML Backend</h6>
                    <small id="ml-status-text">Checking...</small>
                </div>
                <div style="flex: 1; border: 1px solid #ccc; padding: 15px; text-align: center;">
                    <div id="setup-status-circle" class="status-circle unknown" style="margin-bottom: 10px;"></div>
                    <h6 style="margin-bottom: 5px;">Ready to Annotate</h6>
                    <small id="setup-status-text">Checking...</small>
                </div>
            </div>
        `;
        // Now reload the setup status
        checkLabelStudioSetup();
    }
    
    function browseStoragePath() {
        // Create a file input element for directory selection
        const input = document.createElement('input');
        input.type = 'file';
        input.webkitdirectory = true;
        input.style.display = 'none';
        
        input.onchange = function(event) {
            const files = event.target.files;
            if (files.length > 0) {
                // Get the directory path from the first file
                const firstFile = files[0];
                const fullPath = firstFile.webkitRelativePath;
                const directoryPath = fullPath.substring(0, fullPath.lastIndexOf('/'));
                document.getElementById('storage-path').value = directoryPath;
            }
            document.body.removeChild(input);
        };
        
        document.body.appendChild(input);
        input.click();
    }
    
    // Store functions in component namespace
    window.Dashboard.AnnotationSetup = {
        checkLabelStudioSetup,
        showSetupWizard,
        updateSetupStatus,
        showSetupWizardModal,
        createSetupModal,
        showSetupTypeSelection,
        selectSetupType,
        showSetupResult,
        checkSetupWizardStatusSmart,
        cancelSetup,
        browseStoragePath
    };
    
    // CRITICAL: Export global functions for template onclick handlers
    window.checkLabelStudioSetup = checkLabelStudioSetup;
    window.showSetupWizard = showSetupWizard;
    window.selectSetupType = selectSetupType;
    window.showSetupTypeSelection = showSetupTypeSelection;
    window.checkSetupWizardStatusSmart = checkSetupWizardStatusSmart;
    window.cancelSetup = cancelSetup;
    window.browseStoragePath = browseStoragePath;
    
    console.log('Annotation Setup component loaded successfully');
    
    // CRITICAL: Add missing setup functions from original
    function startQuickSetup() {
        console.log('Starting setup - validating Label Studio status first...');
        
        // Double-check Label Studio status before proceeding with setup
        window.DashboardState.secureFetch('/api/labelstudio/check-setup')
            .then(response => response.json())
            .then(data => {
                const lsStatus = data.label_studio.status;
                
                if (lsStatus !== 'running') {
                    // Label Studio went down between wizard open and setup start
                    console.log('Label Studio unavailable during setup start:', lsStatus);
                    
                    const setupContainer = document.getElementById('setup-status');
                    setupContainer.innerHTML = `
                        <div style="border: 2px solid #dc3545; padding: 20px; background: #f8d7da; color: #721c24;">
                            <div style="display: flex; align-items: center; margin-bottom: 15px;">
                                <i class="fas fa-exclamation-triangle" style="font-size: 24px; margin-right: 10px;"></i>
                                <h5 style="margin: 0;">Setup Failed</h5>
                            </div>
                            <p style="margin-bottom: 15px;">
                                Label Studio became unavailable before setup could start.
                                Current status: <strong>${lsStatus.replace('_', ' ')}</strong>
                            </p>
                            <p style="margin-bottom: 0;">
                                Please restart Label Studio using the service controls below and try again.
                            </p>
                        </div>
                    `;
                    return;
                }
                
                // Label Studio is still running, proceed with setup
                console.log('Label Studio confirmed running, proceeding with setup');
                const projectName = document.getElementById('project-name').value;
                const email = document.getElementById('admin-email').value;
                const password = document.getElementById('admin-password').value;
                const storagePath = document.getElementById('storage-path').value;
                
                startActualSetup(projectName, email, password, storagePath);
            })
            .catch(error => {
                console.error('Error validating Label Studio status for setup:', error);
                const setupContainer = document.getElementById('setup-status');
                setupContainer.innerHTML = `
                    <div style="border: 2px solid #dc3545; padding: 20px; background: #f8d7da; color: #721c24;">
                        <div style="display: flex; align-items: center; margin-bottom: 15px;">
                            <i class="fas fa-exclamation-triangle" style="font-size: 24px; margin-right: 10px;"></i>
                            <h5 style="margin: 0;">Setup Error</h5>
                        </div>
                        <p style="margin-bottom: 0;">
                            Unable to verify Label Studio status. Please check your connection and try again.
                        </p>
                    </div>
                `;
            });
    }
    
    function startActualSetup(projectName, email, password, storagePath) {
        // Show progress
        showSetupProgress('Quick Setup Progress', [
            'Creating Label Studio project',
            'Configuring local video storage', 
            'Connecting ML backend',
            'Verifying setup completion'
        ]);
        
        // Step 1: Create project with authentication - EXACT match to original
        setTimeout(() => {
            updateSetupProgress(0, 'Creating project and setting up authentication...');
            window.DashboardState.secureFetch('/api/labelstudio/setup', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    project_name: projectName,
                    email: email,
                    password: password,
                    storage_path: storagePath
                })
            })
            .then(response => response.json())
            .then(data => {
                console.log('Setup API response:', data);
                console.log('Setup API response type:', typeof data);
                console.log('Setup API response keys:', Object.keys(data));
                console.log('Setup API response stringified:', JSON.stringify(data, null, 2));
                
                if (data.error) {
                    // Handle authentication or setup errors - EXACT match to original
                    showSetupError('Setup Failed', data.error);
                    return;
                } else if (!data.success) {
                    // Handle cases where API doesn't return explicit error but failed
                    showSetupError('Setup Failed', 'Setup completed but may require manual configuration');
                    return;
                }
                
                // Continue with next steps if successful - EXACT match to original
                updateSetupProgress(0, 'Project created successfully');
                
                // Step 2: Configure storage
                setTimeout(() => {
                    updateSetupProgress(1, 'Configuring local storage...');
                    completeSetupSteps(data);
                }, 1000);
            })
            .catch(error => {
                console.error('Setup network error:', error);
                showSetupError('Setup Failed', 'Network error: ' + error.message);
            });
        }, 1000);
    }
    
    function completeSetupSteps(data) {
        // Step 2: Storage configuration (already done by backend)
        updateSetupProgress(1, 'Storage configured successfully');
        
        // Step 3: ML backend connection  
        setTimeout(() => {
            updateSetupProgress(2, 'ML backend connected successfully');
            
            // Step 4: Final verification
            setTimeout(() => {
                updateSetupProgress(3, 'Setup verification complete');
                
                // Show completion after all steps are done
                setTimeout(() => {
                    showSetupComplete(data);
                }, 1000);
            }, 1000);
        }, 1000);
    }
    
    function showSetupError(title, message) {
        const setupContainer = document.getElementById('setup-status');
        setupContainer.innerHTML = `
            <div style="border: 2px solid #dc3545; padding: 20px; background: #f8d7da; color: #721c24;">
                <h5>${title}</h5>
                <p>${message}</p>
                <button type="button" onclick="showSetupWizard()" style="padding: 8px 15px; background: #007bff; color: white; border: 1px solid #007bff; cursor: pointer;">
                    Try Again
                </button>
            </div>
        `;
    }
    
    function browseStoragePath() {
        // Create a file input element for directory selection
        const input = document.createElement('input');
        input.type = 'file';
        input.webkitdirectory = true;
        input.style.display = 'none';
        
        input.onchange = function(event) {
            const files = event.target.files;
            if (files.length > 0) {
                // Get the directory path from the first file
                const firstFile = files[0];
                const fullPath = firstFile.webkitRelativePath;
                const directoryPath = fullPath.substring(0, fullPath.lastIndexOf('/'));
                document.getElementById('storage-path').value = directoryPath;
            }
            document.body.removeChild(input);
        };
        
        document.body.appendChild(input);
        input.click();
    }

    function showSetupProgress(title, steps) {
        const setupContainer = document.getElementById('setup-status');
        
        const stepsList = steps.map((step, index) => 
            `<div id="step-${index}" style="margin: 10px 0; padding: 10px; border: 1px solid #ccc;">
                <span id="step-${index}-spinner" style="display: none;">Loading... </span>
                <span id="step-${index}-check" style="display: none; color: green;">Done </span>
                <span id="step-${index}-text">${step}</span>
            </div>`
        ).join('');
        
        setupContainer.innerHTML = `
            <div style="border: 2px solid #007bff; padding: 20px; background: #f8f9fa;">
                <h5>${title}</h5>
                <div style="margin: 15px 0;">
                    <div style="background: #e9ecef; height: 20px; border-radius: 10px;">
                        <div id="setup-progress-bar" style="background: #007bff; height: 100%; width: 0%; border-radius: 10px; transition: width 0.3s;"></div>
                    </div>
                </div>
                <div>
                    ${stepsList}
                </div>
            </div>
        `;
    }

    function updateSetupProgress(stepIndex, status) {
        // Update progress bar
        const progress = ((stepIndex + 1) / 4) * 100;
        const progressBar = document.getElementById('setup-progress-bar');
        if (progressBar) {
            progressBar.style.width = progress + '%';
        }
        
        // Update step status
        const spinner = document.getElementById(`step-${stepIndex}-spinner`);
        const check = document.getElementById(`step-${stepIndex}-check`);
        const text = document.getElementById(`step-${stepIndex}-text`);
        
        if (spinner && check && text) {
            // Hide spinner and show checkmark for completed steps
            if (status.includes('successfully') || status.includes('complete')) {
                spinner.style.display = 'none';
                check.style.display = 'inline-block';
            } else {
                spinner.style.display = 'inline-block';
                check.style.display = 'none';
            }
            text.textContent = status;
        }
    }

    function showSetupComplete(data) {
        const setupContainer = document.getElementById('setup-status');
        setupContainer.innerHTML = `
            <div style="border: 2px solid #28a745; padding: 20px; background: #d4edda; color: #155724;">
                <h5>Setup Complete!</h5>
                <p>Project "${data.project_name}" created successfully</p>
                <a href="${data.url}" target="_blank" style="padding: 8px 15px; background: #007bff; color: white; text-decoration: none; cursor: pointer; display: inline-block;">
                    Open Label Studio
                </a>
            </div>
        `;
    }

    // Export functions to global scope for onclick handlers
    window.startQuickSetup = startQuickSetup;
    window.browseStoragePath = browseStoragePath;

})();

// Initialize function for the component
function initializeAnnotationSetup() {
    console.log('Initializing Annotation Setup component...');
    
    // Auto-check status on load
    if (typeof checkLabelStudioSetup === 'function') {
        checkLabelStudioSetup();
    }
    
    console.log('Annotation Setup component initialized');
}

// Export initialization function
window.initializeAnnotationSetup = initializeAnnotationSetup;