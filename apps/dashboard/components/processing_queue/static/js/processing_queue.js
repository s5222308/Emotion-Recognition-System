/**
 * Processing Queue Component JavaScript
 */

(function() {
    'use strict';
    
    console.log('Processing Queue component loaded');
    
    // Component namespace
    window.Dashboard = window.Dashboard || {};
    window.Dashboard.ProcessingQueue = {};
    
    // Update queue status
    function updateQueueStatus(data) {
        if (data.queue_length !== undefined) {
            const queueElement = document.getElementById('queue-length');
            if (queueElement) queueElement.textContent = data.queue_length;
        }
        
        if (data.is_processing !== undefined) {
            const processingElement = document.getElementById('is-processing');
            if (processingElement) {
                processingElement.textContent = data.is_processing ? 'Yes' : 'No';
            }
        }
        
        if (data.total_processed !== undefined) {
            const processedElement = document.getElementById('total-processed');
            if (processedElement) processedElement.textContent = data.total_processed;
        }
        
        if (data.total_failed !== undefined) {
            const failedElement = document.getElementById('total-failed');
            if (failedElement) failedElement.textContent = data.total_failed;
        }
        
        // Update current processing task
        const currentProcessingDiv = document.getElementById('current-processing');
        const currentTaskDiv = document.getElementById('current-task');
        
        if (data.current_task && currentProcessingDiv && currentTaskDiv) {
            currentProcessingDiv.style.display = 'block';
            currentTaskDiv.textContent = data.current_task;
        } else if (currentProcessingDiv) {
            currentProcessingDiv.style.display = 'none';
        }
        
        // Update queue items
        updateQueueItems(data.queue_items || []);
        
        // Update recent tasks
        updateRecentTasks(data.recent_tasks || []);
    }
    
    // Update queue items list
    function updateQueueItems(items) {
        const queueItemsDiv = document.getElementById('queue-items');
        if (!queueItemsDiv) return;
        
        if (items.length === 0) {
            queueItemsDiv.innerHTML = '<div style="padding: 10px; color: gray;">No tasks in queue</div>';
        } else {
            queueItemsDiv.innerHTML = items.map(item => 
                `<div style="padding: 10px; border-bottom: 1px solid #eee;">
                    <strong>${item.type || 'Task'}</strong>: ${item.description || 'Processing...'}
                </div>`
            ).join('');
        }
    }
    
    // Update recent tasks list
    function updateRecentTasks(tasks) {
        const recentItemsDiv = document.getElementById('recent-items');
        if (!recentItemsDiv) return;
        
        if (tasks.length === 0) {
            recentItemsDiv.innerHTML = '<div style="padding: 10px; color: gray;">No recent tasks</div>';
        } else {
            recentItemsDiv.innerHTML = tasks.map(task => {
                const statusColor = task.status === 'completed' ? 'green' : 
                                  task.status === 'failed' ? 'red' : 'orange';
                return `<div style="padding: 10px; border-bottom: 1px solid #eee;">
                    <strong>${task.type || 'Task'}</strong>: ${task.description || 'Unknown'}
                    <span style="color: ${statusColor}; float: right;">${task.status || 'Unknown'}</span>
                </div>`;
            }).join('');
        }
    }
    
    // Initialize function
    function initializeProcessingQueue() {
        console.log('Initializing Processing Queue component...');
        
        // Initial load of queue status
        fetch('/api/queue/status')
            .then(response => response.json())
            .then(data => {
                updateQueueStatus(data);
            })
            .catch(error => {
                console.error('Error loading queue status:', error);
            });
        
        console.log('Processing Queue component initialized');
    }
    
    // Export functions
    window.Dashboard.ProcessingQueue.updateQueueStatus = updateQueueStatus;
    window.Dashboard.ProcessingQueue.initializeProcessingQueue = initializeProcessingQueue;
    
    // Also export to global scope for dashboard.js
    window.updateQueueStatus = updateQueueStatus;
    window.initializeProcessingQueue = initializeProcessingQueue;
    
})();