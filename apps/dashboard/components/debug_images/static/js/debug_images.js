/**
 * Debug Images Component
 * CRITICAL: This preserves EXACT functionality from monolithic dashboard
 * All functions and behavior must match original implementation
 */

(function() {
    'use strict';

    // Create component namespace
    window.Dashboard = window.Dashboard || {};
    window.Dashboard.DebugImages = {
        
        /**
         * Update debug images display
         * CRITICAL: This MUST match the EXACT behavior from original
         * Original function: updateDebugImages() in dashboard.html lines 1517-1547
         */
        updateDebugImages: function() {
            console.log('[DEBUG] Debug Images: updateDebugImages called');
            
            fetch('/api/debug-images')
                .then(response => response.json())
                .then(images => {
                    console.log('[DEBUG] Debug Images: received', images.length, 'images');
                    const grid = document.getElementById('debug-images-grid');
                    
                    if (!grid) {
                        console.log('[DEBUG] Debug Images: grid element not found');
                        return;
                    }
                    
                    grid.innerHTML = '';
                    
                    if (images.length === 0) {
                        grid.innerHTML = '<p class="text-muted">No debug images yet</p>';
                        return;
                    }
                    
                    // EXACT same display logic as original (lines 1529-1543)
                    images.slice(0, 6).forEach(img => {
                        const timestamp = new Date(img.timestamp).toLocaleTimeString();
                        const emotion = img.filename.split('_')[1] || 'unknown';
                        
                        grid.innerHTML += `
                            <div style="width: 120px; margin-bottom: 10px;">
                                <img src="/debug-images/${img.filename}" 
                                     class="debug-image" 
                                     title="${img.filename} (${timestamp})"
                                     onerror="this.style.display='none'">
                                <div style="text-align: center; margin-top: 5px;">
                                    <small style="background: gray; color: white; padding: 2px 6px; border-radius: 3px;">${emotion}</small>
                                </div>
                            </div>
                        `;
                    });
                    
                })
                .catch(error => {
                    console.error('[ERROR] Debug Images: Error updating debug images:', error);
                    const grid = document.getElementById('debug-images-grid');
                    if (grid) {
                        grid.innerHTML = '<p style="color: red;">Error loading debug images</p>';
                    }
                });
        },

        /**
         * Update debug images with live data  
         * CRITICAL: This MUST match the EXACT behavior from original
         * Original function: updateDebugImagesLive() in dashboard.html lines 1590-1618
         */
        updateDebugImagesLive: function(images) {
            console.log('[DEBUG] Debug Images: updateDebugImagesLive called with', images.length, 'images');
            
            const grid = document.getElementById('debug-images-grid');
            if (!grid) {
                console.log('[DEBUG] Debug Images: grid element not found');
                return;
            }
            
            grid.innerHTML = '';
            
            if (images.length === 0) {
                grid.innerHTML = '<p class="text-muted">No debug images yet</p>';
                return;
            }
            
            // EXACT same display logic as original (lines 1601-1616)
            images.forEach(img => {
                const timestamp = new Date(img.timestamp).toLocaleTimeString();
                const emotion = img.filename.split('_')[1] || 'unknown';
                
                grid.innerHTML += `
                    <div style="width: 120px; margin-bottom: 10px;">
                        <div class="debug-image-container">
                            <img src="/debug-images/${img.filename}" 
                                 class="debug-image" 
                                 title="${img.filename} (${timestamp})"
                                 onerror="this.style.display='none'">
                            <small style="background: gray; color: white; padding: 2px 6px; border-radius: 3px; margin-top: 5px; display: inline-block;">${emotion}</small>
                        </div>
                    </div>
                `;
            });
        }
    };

    // CRITICAL: Export global functions for polling system compatibility
    // This ensures the polling system can call these functions
    window.updateDebugImages = window.Dashboard.DebugImages.updateDebugImages;
    window.updateDebugImagesLive = window.Dashboard.DebugImages.updateDebugImagesLive;

    console.log('[DEBUG] Debug Images component JavaScript loaded');

})();