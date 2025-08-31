"""
Queue-specific SSE Handler
Handles real-time queue updates
"""
import time
import json
import requests
from flask import Response

class QueueSSEHandler:
    """Handle SSE streaming for processing queue updates"""
    
    @staticmethod
    def stream_queue_updates():
        """SSE endpoint for real-time processing queue updates"""
        def generate():
            last_state = None
            while True:
                try:
                    # Get detailed queue state from ML backend
                    response = requests.get('http://localhost:9091/queue/status', timeout=2)
                    if response.status_code == 200:
                        queue_data = response.json()
                        
                        stats = queue_data.get('stats', {})
                        current_state = {
                            'queue_length': stats.get('queue_length', 0),
                            'is_processing': stats.get('is_processing', False),
                            'total_processed': stats.get('total_processed', 0),
                            'total_failed': stats.get('total_failed', 0),
                            'pending_tasks': queue_data.get('queue', []),
                            'recent_tasks': queue_data.get('recent_tasks', []),
                            'timestamp': time.time()
                        }
                        
                        # Only send if meaningful data changed (ignore timestamp)
                        current_data = {k: v for k, v in current_state.items() if k != 'timestamp'}
                        last_data = {k: v for k, v in (last_state or {}).items() if k != 'timestamp'}
                        
                        if current_data != last_data:
                            yield f"data: {json.dumps(current_state)}\\n\\n"
                            last_state = current_state
                    else:
                        # ML backend down - send error state
                        error_state = {
                            'queue_length': 0,
                            'is_processing': False,
                            'total_processed': 0,
                            'total_failed': 0,
                            'pending_tasks': [],
                            'recent_tasks': [],
                            'error': 'ML backend unavailable'
                        }
                        if error_state != last_state:
                            yield f"data: {json.dumps(error_state)}\\n\\n"
                            last_state = error_state
                            
                except Exception as e:
                    print(f'[DEBUG] Queue SSE stream error: {e}')
                
                # Send heartbeat every 30 seconds
                if int(time.time()) % 30 == 0:
                    yield ": heartbeat\\n\\n"
                
                time.sleep(0.5)  # Check for updates every 500ms
        
        return Response(
            generate(),
            mimetype="text/event-stream",
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no'
            }
        )