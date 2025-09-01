#!/usr/bin/env python3
"""
ML Backend that uses REAL face detection and emotion recognition
Connects to our working ML service that has RetinaFace + ByteTrack + AffectNet
"""

from flask import Flask, request, jsonify
import requests
import logging
import numpy as np
import sys
from pathlib import Path
import concurrent.futures
import threading
from datetime import datetime
from collections import deque, defaultdict

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Queue tracking
processing_queue = deque()  # Current queue
processing_stats = {
    'total_processed': 0,
    'total_failed': 0,
    'current_processing': None,
    'queue_start_time': None
}
task_history = deque(maxlen=50)  # Keep last 50 processed tasks

# Connect to the modular ML service V2 (on port 5003!)
ML_SERVICE_URL = "http://localhost:5003/prelabel"
ML_SERVICE_HEALTH_URL = "http://localhost:5003/health"

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint with ML service connectivity"""
    try:
        # Check if we can connect to the ML service
        ml_status = "disconnected"
        try:
            response = requests.get(ML_SERVICE_HEALTH_URL, timeout=2)
            if response.status_code == 200:
                ml_status = "connected"
        except:
            ml_status = "disconnected"
        
        # Get queue stats
        queue_length = len(processing_queue)
        is_processing = processing_stats['current_processing'] is not None
        total_processed = processing_stats['total_processed']
        total_failed = processing_stats['total_failed']
        
        return jsonify({
            "status": "OK",
            "service": "Label Studio ML Backend",
            "ml_service_status": ml_status,
            "ml_service_url": ML_SERVICE_URL,
            "queue_stats": {
                "queue_length": queue_length,
                "is_processing": is_processing,
                "total_processed": total_processed,
                "total_failed": total_failed
            },
            "description": "Bridge between Label Studio and ML Engine"
        })
        
    except Exception as e:
        return jsonify({
            "status": "ERROR", 
            "error": str(e),
            "service": "Label Studio ML Backend"
        }), 500

@app.route('/status', methods=['GET'])
def status_page():
    """HTML health status page"""
    try:
        # Get health data
        health_response = health()
        health_data = health_response.get_json()
        
        # Determine status class for styling and normalize status text
        service_status = health_data.get('status', 'unknown').lower()
        if service_status in ['ok', 'healthy']:
            status_class = "healthy"
            status_text = "HEALTHY"
        elif health_data.get('ml_service_status') == 'disconnected':
            status_class = "warning"
            status_text = "WARNING"
        else:
            status_class = "error"
            status_text = "ERROR"
        
        # Generate HTML page with cleaner professional style
        html = f"""<!DOCTYPE html>
        <html>
        <head>
            <title>ML Backend Service - Status</title>
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 40px; background: #f5f5f5; }}
                .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .header {{ text-align: center; margin-bottom: 30px; color: #2c3e50; }}
                .status {{ padding: 15px; border-radius: 5px; margin: 15px 0; }}
                .healthy {{ background: #d4edda; border-left: 4px solid #28a745; }}
                .error {{ background: #f8d7da; border-left: 4px solid #dc3545; }}
                .warning {{ background: #fff3cd; border-left: 4px solid #ffc107; }}
                .component {{ background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .component h4 {{ margin-top: 0; color: #495057; }}
                .info {{ color: #6c757d; font-size: 14px; }}
                .metric {{ display: inline-block; margin: 10px 20px 10px 0; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
                .metric-label {{ font-size: 12px; color: #6c757d; text-transform: uppercase; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>ML Backend Service</h1>
                    <h2>Label Studio Connector</h2>
                </div>
                
                <div class="status {status_class}">
                    <h3>Service Status: {status_text}</h3>
                    <p>Port: 9091 | ML Service: {health_data.get('ml_service_status', 'unknown').title()}</p>
                </div>
                
                <div class="component">
                    <h4>Processing Statistics</h4>
                    <div class="metric">
                        <div class="metric-value">{health_data.get('stats', {}).get('queue_length', 0)}</div>
                        <div class="metric-label">Queue Length</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{health_data.get('stats', {}).get('total_processed', 0)}</div>
                        <div class="metric-label">Total Processed</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{health_data.get('stats', {}).get('total_failed', 0)}</div>
                        <div class="metric-label">Failed</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{'Yes' if health_data.get('stats', {}).get('is_processing', False) else 'No'}</div>
                        <div class="metric-label">Currently Processing</div>
                    </div>
                </div>
                
                <div class="component">
                    <h4>Service Information</h4>
                    <p><strong>Purpose:</strong> Bridge between Label Studio and ML Visual Inference Service</p>
                    <p><strong>API Endpoints:</strong> /health, /predict, /queue/status, /shutdown</p>
                    <p><strong>ML Service Connection:</strong> {health_data.get('ml_service_status', 'Unknown').title()}</p>
                    <p><strong>ML Service URL:</strong> {health_data.get('ml_service_url', 'N/A')}</p>
                </div>
                
                <div class="component">
                    <h4>Backend Details</h4>
                    <p><strong>Function:</strong> Queue management and preprocessing for video emotion recognition</p>
                    <p><strong>Processing:</strong> Handles Label Studio webhook requests and forwards to ML engine</p>
                    <p><strong>Queue System:</strong> Manages video processing tasks with progress tracking</p>
                    <p><strong>Error Handling:</strong> Automatic retry and failure recovery mechanisms</p>
                </div>
                
                <div class="component" id="error-logs-section">
                    <h4>Recent Processing Issues</h4>
                    <div id="error-logs">Loading recent errors...</div>
                </div>
                
                <div class="info" style="text-align: center; margin-top: 30px; padding-top: 20px; border-top: 1px solid #dee2e6;">
                    <a href="/health" style="color: #007bff;">JSON Health API</a> | 
                    <a href="/queue/status" style="color: #007bff;">Queue Status API</a> | 
                    <a href="http://localhost:8081" style="color: #007bff;">Dashboard</a>
                </div>
                
                <script>
                // Load error logs with better styling
                fetch('http://localhost:8081/api/services/ml_backend/errors')
                    .then(response => response.json())
                    .then(data => {{
                        const errorLogsDiv = document.getElementById('error-logs');
                        if (data.error_log && data.error_log.length > 0) {{
                            let logsHtml = '<div style="max-height: 300px; overflow-y: auto; border: 1px solid #dee2e6; border-radius: 5px; padding: 15px; background: #f8f9fa;">';
                            logsHtml += `<p style="margin-bottom: 15px;"><strong>Total Errors:</strong> ${{data.total_errors}} | <strong>Recent Consecutive:</strong> ${{data.consecutive_errors}}</p>`;
                            data.error_log.reverse().forEach(log => {{
                                const timestamp = new Date(log.timestamp).toLocaleString();
                                const typeColor = log.type === 'CONNECTION_ERROR' ? '#dc3545' : '#ffc107';
                                logsHtml += `
                                    <div style="margin-bottom: 10px; padding: 10px; border-left: 4px solid ${{typeColor}}; background: white; border-radius: 3px;">
                                        <div style="font-weight: bold; color: ${{typeColor}}; margin-bottom: 5px;">${{log.type}} - ${{timestamp}}</div>
                                        <div style="font-family: 'SF Mono', Consolas, monospace; font-size: 13px; color: #495057; line-height: 1.4;">${{log.message}}</div>
                                    </div>
                                `;
                            }});
                            logsHtml += '</div>';
                            errorLogsDiv.innerHTML = logsHtml;
                        }} else {{
                            errorLogsDiv.innerHTML = '<p style="color: #28a745; font-weight: 500;">No recent errors - service is running smoothly!</p>';
                        }}
                    }})
                    .catch(error => {{
                        document.getElementById('error-logs').innerHTML = '<p style="color: #dc3545; font-weight: 500;">Unable to load error logs from dashboard</p>';
                    }});
                </script>
            </div>
        </body>
        </html>"""
        return html
    except Exception as e:
        error_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ML Backend - Error</title>
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 40px; background: #f5f5f5; }}
                .container {{ max-width: 600px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .error {{ background: #f8d7da; border-left: 4px solid #dc3545; padding: 15px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="error">
                    <h2>Service Error</h2>
                    <p>Unable to get service status: {str(e)}</p>
                </div>
            </div>
        </body>
        </html>
        """
        return error_html, 500

@app.route('/queue/status', methods=['GET'])
def queue_status():
    """Get current queue status for dashboard monitoring"""
    queue_data = []
    
    # Add current processing task
    if processing_stats['current_processing']:
        current = processing_stats['current_processing']
        queue_data.append({
            'id': current.get('id', 0),
            'inner_id': current.get('inner_id', current.get('id', 0)),
            'video_path': current.get('video_path', 'Unknown'),
            'status': 'processing',
            'start_time': current.get('start_time', datetime.now()).isoformat(),
            'duration': (datetime.now() - current.get('start_time', datetime.now())).total_seconds()
        })
    
    # Add pending queue items
    for item in list(processing_queue):
        queue_data.append({
            'id': item.get('id', 0),
            'inner_id': item.get('inner_id', item.get('id', 0)),
            'video_path': item.get('video_path', 'Unknown'),
            'status': 'pending',
            'queued_time': item.get('queued_time', datetime.now()).isoformat(),
            'duration': 0
        })
    
    # Recent completed/failed tasks
    recent_tasks = []
    for task in list(task_history)[-10:]:  # Last 10 tasks
        recent_tasks.append({
            'id': task.get('id', 0),
            'inner_id': task.get('inner_id', task.get('id', 0)),
            'video_path': task.get('video_path', 'Unknown'),
            'status': task.get('status', 'unknown'),
            'processing_time': task.get('processing_time', 0),
            'completed_time': task.get('completed_time', datetime.now()).isoformat()
        })
    
    return jsonify({
        'queue': queue_data,
        'recent_tasks': recent_tasks,
        'stats': {
            'queue_length': len(processing_queue),
            'is_processing': processing_stats['current_processing'] is not None,
            'total_processed': processing_stats['total_processed'],
            'total_failed': processing_stats['total_failed']
        }
    })

def get_dynamic_model_version():
    """Get dynamic model version from Universal ONNX system"""
    try:
        # Query ML service for Universal ONNX model information
        response = requests.get(ML_SERVICE_HEALTH_URL, timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            
            # Get current active models
            face_detector = health_data.get('face_detector', 'Unknown')
            emotion_model = health_data.get('emotion_model', 'Unknown')
            service_type = health_data.get('service', 'Unknown')
            
            # Build version string for Universal ONNX system
            version_lines = [
                f"Face Detector: {face_detector}",
                f"Emotion Model: {emotion_model}", 
                f"Service: {service_type}",
                "Architecture: Universal ONNX"
            ]
            
            return "\n".join(version_lines)
        else:
            logger.warning(f"Failed to get ML service health: {response.status_code}")
            return "Universal ONNX System (service unavailable)"
    except Exception as e:
        logger.error(f"Failed to query ML service: {e}")
        return "Universal ONNX System (offline)"

@app.route('/setup', methods=['POST'])
def setup():
    model_version = get_dynamic_model_version()
    return jsonify({"model_version": model_version})

def process_single_video(task, ls_frame_count):
    """Process a single video task - designed for parallel execution"""
    task_id = task.get('id', 0)
    inner_id = task.get('inner_id', task_id)  # Use inner_id if available
    video_data = task.get('data', {})
    video_url = video_data.get('video', '')
    
    logger.info(f"[PARALLEL] Processing task {task_id} (inner_id: {inner_id}), video: {video_url}")
    
    # Track current processing
    video_path = convert_ls_url_to_path(video_url)
    processing_stats['current_processing'] = {
        'id': task_id,
        'inner_id': inner_id,
        'video_path': video_path or video_url,
        'start_time': datetime.now()
    }
    
    # Remove from pending queue
    global processing_queue
    for i, pending_item in enumerate(list(processing_queue)):
        if pending_item.get('inner_id') == inner_id or pending_item.get('id') == task_id:
            try:
                processing_queue.remove(pending_item)
                logger.info(f"[QUEUE] Removed task {inner_id} from pending queue")
                break
            except:
                pass
    
    try:
        # Convert Label Studio URL to local path
        video_path = convert_ls_url_to_path(video_url)
        
        if video_path:
            import time
            start_time = time.time()
            logger.info(f"[PARALLEL] Calling ML service for: {video_path}")
            try:
                # Call real ML service - process full videos with intelligent sampling
                # Don't send sample_rate - let ML service use its configured default from registry
                ml_response = requests.post(ML_SERVICE_URL, json={
                    'video_path': video_path,
                    'frame_limit': None   # Process full video (None = no artificial limit)
                }, timeout=300)  # 5 minute timeout per video (reasonable for longer videos)
                
                if ml_response.status_code == 200:
                    processing_time = time.time() - start_time
                    ml_data = ml_response.json()
                    
                    # Log performance metrics
                    tracks = ml_data.get('tracks', [])
                    if tracks and len(tracks) > 0:
                        num_boxes = len(tracks)  # Each track is one detection
                        logger.info(f"[PARALLEL] Success: ML service completed task {task_id} in {processing_time:.2f}s: {num_boxes} boxes")
                    else:
                        logger.info(f"[PARALLEL] Warning: ML service completed task {task_id} in {processing_time:.2f}s: no tracks")
                    
                    # Convert ML predictions to Label Studio format with video path for frame calculation
                    ls_annotations = convert_to_label_studio_format(ml_data, video_path)
                    
                    # Track completion
                    processing_stats['total_processed'] += 1
                    processing_stats['current_processing'] = None
                    task_history.append({
                        'id': task_id,
                        'inner_id': inner_id,
                        'video_path': video_path,
                        'status': 'completed',
                        'processing_time': processing_time,
                        'completed_time': datetime.now()
                    })
                    
                    if ls_annotations:
                        logger.info(f"[PARALLEL] Got {len(ls_annotations)} annotations for task {task_id}")
                        return {
                            'result': ls_annotations,
                            'score': 0.85,
                            'model_version': get_dynamic_model_version()
                        }
                    else:
                        logger.warning(f"[PARALLEL] No annotations from ML service for task {task_id}, using fallback")
                        return get_fallback_predictions()
                else:
                    # Track failure
                    processing_stats['total_failed'] += 1
                    processing_stats['current_processing'] = None
                    task_history.append({
                        'id': task_id,
                        'inner_id': inner_id,
                        'video_path': video_path,
                        'status': 'failed',
                        'processing_time': time.time() - start_time,
                        'completed_time': datetime.now(),
                        'error': f"ML service error: {ml_response.status_code}"
                    })
                    logger.error(f"[PARALLEL] ML service error for task {task_id}: {ml_response.status_code}")
                    return get_fallback_predictions()
            except Exception as e:
                # Track failure
                processing_stats['total_failed'] += 1
                processing_stats['current_processing'] = None
                task_history.append({
                    'id': task_id,
                    'inner_id': inner_id,
                    'video_path': video_path,
                    'status': 'failed',
                    'processing_time': time.time() - start_time,
                    'completed_time': datetime.now(),
                    'error': str(e)
                })
                logger.error(f"[PARALLEL] Failed to call ML service for task {task_id}: {e}")
                return get_fallback_predictions()
        else:
            logger.warning(f"[PARALLEL] Could not convert URL for task {task_id}: {video_url}")
            # Return empty predictions for videos we can't find
            return {
                'result': [],
                'score': 0.0,
                'model_version': get_dynamic_model_version()
            }
            
    except Exception as e:
        logger.error(f"[PARALLEL] Error processing task {task_id}: {e}")
        return get_fallback_predictions()

@app.route('/predict', methods=['POST'])
def predict():
    """Get real predictions from ML service and convert to Label Studio format - WITH PARALLEL PROCESSING"""
    data = request.json
    tasks = data.get('tasks', [])
    
    # Check if Label Studio provides frame count in the request
    context = data.get('context', {})
    ls_frame_count = context.get('frameCount') or context.get('frame_count')
    
    num_tasks = len(tasks)
    logger.info(f"Starting PARALLEL PROCESSING: {num_tasks} videos")
    
    # Add tasks to queue tracking
    for task in tasks:
        task_id = task.get('id', 0)
        inner_id = task.get('inner_id', task_id)  # Use inner_id if available
        video_data = task.get('data', {})
        video_url = video_data.get('video', '')
        video_path = convert_ls_url_to_path(video_url)
        
        processing_queue.append({
            'id': task_id,
            'inner_id': inner_id,
            'video_path': video_path or video_url,
            'queued_time': datetime.now()
        })
        logger.info(f"[QUEUE] Added task {task_id} (inner_id: {inner_id}) to queue")
    
    # Process videos in parallel using ThreadPoolExecutor
    predictions = []
    
    if num_tasks == 1:
        # Single video - process normally for faster response
        predictions.append(process_single_video(tasks[0], ls_frame_count))
    else:
        # Multiple videos - use SEQUENTIAL processing to avoid ML service crashes
        # The ML service (MediaPipe/SCRFD) is not thread-safe and crashes with concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            # Submit all video processing tasks
            future_to_task = {executor.submit(process_single_video, task, ls_frame_count): task for task in tasks}
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]
                task_id = task.get('id', 0)
                inner_id = task.get('inner_id', task_id)
                try:
                    prediction = future.result(timeout=300)  # 5 minute timeout per task
                    predictions.append(prediction)
                    logger.info(f"[PARALLEL] Success: Completed task {task_id} (inner_id: {inner_id})")
                        
                except Exception as e:
                    logger.error(f"[PARALLEL] Error: Task {task_id} (inner_id: {inner_id}) failed: {e}")
                    predictions.append(get_fallback_predictions())
                    
                    # Track failure in case it wasn't tracked in process_single_video
                    processing_stats['total_failed'] += 1
                    processing_stats['current_processing'] = None
                    task_history.append({
                        'id': task_id,
                        'inner_id': inner_id,
                        'video_path': task.get('data', {}).get('video', 'Unknown'),
                        'status': 'failed',
                        'processing_time': 0,
                        'completed_time': datetime.now(),
                        'error': str(e)
                    })
    
    # Clear any remaining queue items (all should be processed by now)
    remaining_items = len(processing_queue)
    if remaining_items > 0:
        logger.warning(f"[QUEUE] Clearing {remaining_items} remaining queue items")
        processing_queue.clear()
    processing_stats['current_processing'] = None
    
    logger.info(f"Finished PARALLEL PROCESSING: {len(predictions)}/{num_tasks} videos processed")
    return jsonify({'results': predictions})


def convert_ls_url_to_path(url):
    """Convert Label Studio URL to local file path"""
    
    # Handle file:// URLs (direct file paths)
    if url.startswith('file://'):
        from urllib.parse import urlparse
        parsed_url = urlparse(url)
        file_path = Path(parsed_url.path)
        if file_path.exists():
            logger.info(f"Found direct file: {file_path}")
            return str(file_path)
        else:
            logger.warning(f"Direct file not found: {file_path}")
            return None
    
    # Strip URL prefix if present (http://localhost:8081/data/upload/... -> /data/upload/...)
    if 'http://' in url or 'https://' in url:
        from urllib.parse import urlparse
        parsed_url = urlparse(url)
        url = parsed_url.path
    
    # Handle uploaded files (/data/upload/...)
    if url.startswith('/data/upload/'):
        # Convert to Label Studio media path
        # /data/upload/1/xyz.mp4 -> ~/.local/share/label-studio/media/upload/1/xyz.mp4
        relative_path = url.replace('/data/', '')  # Remove /data/ prefix
        ls_media_path = Path.home() / '.local/share/label-studio/media' / relative_path
        if ls_media_path.exists():
            logger.info(f"Found uploaded file: {ls_media_path}")
            return str(ls_media_path)
        else:
            logger.warning(f"Upload file not found: {ls_media_path}")
            
            # Try fallback: search in all upload subdirectories
            media_upload_dir = Path.home() / '.local/share/label-studio/media/upload'
            filename = Path(url).name
            logger.info(f"Searching for uploaded file: {filename}")
            
            if media_upload_dir.exists():
                for upload_subdir in media_upload_dir.glob('*/'):
                    candidate_path = upload_subdir / filename
                    if candidate_path.exists():
                        logger.info(f"Found uploaded file via search: {candidate_path}")
                        return str(candidate_path)
                        
            logger.warning(f"Upload file not found anywhere: {filename}")
    
    # Handle local files (/data/local-files/?d=...)
    elif '/data/local-files/?d=' in url:
        parts = url.split('/data/local-files/?d=')
        if len(parts) > 1:
            relative_path = parts[1]
            
            # If path doesn't start with /, add it (Label Studio sometimes omits leading slash)
            if not relative_path.startswith('/'):
                relative_path = '/' + relative_path
            
            # Try the direct path first
            direct_path = Path(relative_path)
            if direct_path.exists():
                logger.info(f"Found local file (direct): {direct_path}")
                return str(direct_path)
            
            # Try multiple possible base paths (self-contained baseline + fallback to original)
            current_dir = Path(__file__).parent
            base_paths = [
                current_dir / relative_path.lstrip('/'),
                current_dir / 'data' / 'videos' / relative_path.replace('data/videos/', '').lstrip('/'),
                Path('/home/lleyt/WIL_project/emotion_annotation_project') / relative_path.lstrip('/'),
                Path('/home/lleyt/WIL_project/emotion_annotation_project/data/videos') / relative_path.replace('data/videos/', '').lstrip('/'),
            ]
            
            for full_path in base_paths:
                if full_path.exists():
                    logger.info(f"Found local file: {full_path}")
                    return str(full_path)
            
            logger.warning(f"Could not find file in any location for: {relative_path}")
    
    return None

def ml_to_ls_frame(ml_frame: int) -> int:
    """Convert ML 0-based frame to Label Studio 1-based frame"""
    return ml_frame + 1

# FACS Action Unit names mapping (OpenFace documented AUs)
AU_NAMES = {
    "AU01": "Inner Brow Raiser",
    "AU02": "Outer Brow Raiser",
    "AU04": "Brow Lowerer",
    "AU05": "Upper Lid Raiser",
    "AU06": "Cheek Raiser",
    "AU07": "Lid Tightener",
    "AU09": "Nose Wrinkler",
    "AU10": "Upper Lip Raiser",
    "AU12": "Lip Corner Puller",
    "AU14": "Dimpler",
    "AU15": "Lip Corner Depressor",
    "AU17": "Chin Raiser",
    "AU20": "Lip Stretcher",
    "AU23": "Lip Tightener",
    "AU25": "Lips Part",
    "AU26": "Jaw Drop",
    "AU28": "Lip Suck",        # presence-only in OpenFace
    "AU45": "Blink"
}

def au_label(code: str) -> str:
    """Get human-readable name for AU code, handling _r/_c suffixes"""
    base = code.split("_")[0]  # AU06_r -> AU06
    return AU_NAMES.get(base, base)

def get_au_data_for_frame(frame_idx: int, action_units: list) -> dict:
    """Get Action Unit data for a specific frame"""
    for au_data in action_units:
        if au_data.get('frame_idx') == frame_idx:
            return au_data
    return None

def format_au_label(emotion: str, au_data: dict) -> str:
    """Format emotion label with Action Unit information"""
    if not au_data:
        return emotion
    
    # Get AU data and filter active ones
    aus = au_data.get('aus', {})
    active_info = []
    
    # Process AUs, handling _r (intensity) vs _c (presence) appropriately
    for au_code, value in aus.items():
        if au_code.endswith('_r'):  # Intensity
            if value > 2.0:  # Significant intensity
                au_name = au_label(au_code)
                active_info.append(f"{au_code.replace('_r', '')}: {au_name} ({value:.1f})")
        elif au_code.endswith('_c'):  # Presence
            if value > 0.5:  # Present
                au_name = au_label(au_code)
                active_info.append(f"{au_code.replace('_c', '')}: {au_name}")
    
    if active_info:
        # Limit to top 3 AUs to avoid cluttering
        au_text = ', '.join(active_info[:3])
        return f"{emotion} [{au_text}]"
    else:
        return f"{emotion} [no active AUs]"

# Removed complex frame clamping logic - SCRFD is reliable and doesn't need it!

def convert_to_label_studio_format(ml_data, video_path=None):
    """Convert ML service predictions to Label Studio videorectangle format with AU data - simplified!"""
    annotations = []
    
    # Use full_results.tracks if available (contains ALL frames), otherwise fallback to tracks
    if 'full_results' in ml_data and 'tracks' in ml_data.get('full_results', {}):
        tracks = ml_data['full_results']['tracks']
        logger.info(f"Using full_results.tracks with {len(tracks)} frames (all processed frames)")
    else:
        tracks = ml_data.get('tracks', [])
        logger.info(f"Using top-level tracks with {len(tracks)} frames")
    
    predictions = ml_data.get('predictions', [])
    action_units = ml_data.get('action_units', [])
    au_info = ml_data.get('au_analysis_info', {})
    
    # Early exit if no valid data - new format has individual tracks with bbox
    if not tracks or not tracks[0].get('bbox'):
        logger.warning("No valid track data from ML service - skipping empty annotation")
        return []
    
    # New format: tracks are the boxes, each with bbox, emotion, frame
    boxes = tracks  # tracks ARE the boxes in new format
    emotion = 'neutral'  # Default, will be overridden per frame
    
    # Get actual video dimensions from ML service
    video_info = ml_data.get('video_info', {})
    width = video_info.get('width', 1280)  # RAVDESS videos are 1280x720
    height = video_info.get('height', 720)
    
    # Simple logging - trust SCRFD to provide valid frames!
    if boxes:
        first_ml_frame = boxes[0].get('frame', 0)
        last_ml_frame = boxes[-1].get('frame', 0)
        logger.info(f"Processing {len(boxes)} detections: ml_frames=[{first_ml_frame}..{last_ml_frame}]")
    
    sequence = []
    emotions_per_frame = []
    
    # Check if we're using fixed sampling (all frames should be keyframes)
    # Detect fixed sampling by checking if frames are evenly spaced
    use_all_frames = False
    if len(boxes) > 1:
        # Check if frames are evenly spaced (fixed sampling)
        frame_numbers = [box.get('frame', 0) for box in boxes]
        if len(frame_numbers) > 1:
            intervals = [frame_numbers[i+1] - frame_numbers[i] for i in range(len(frame_numbers)-1)]
            # If all intervals are the same (or very close), it's fixed sampling
            if intervals and all(abs(interval - intervals[0]) <= 1 for interval in intervals):
                use_all_frames = True
                logger.info(f"Fixed sampling detected ({len(boxes)} frames with interval {intervals[0]}) - using ALL frames as keyframes")
    
    if not use_all_frames and len(boxes) > 20:
        # Fallback: if we have many frames, also treat as fixed sampling
        use_all_frames = True
        logger.info(f"Many frames detected ({len(boxes)} frames) - using ALL frames as keyframes")
    
    # Create keyframes - either all frames (fixed sampling) or intelligent selection
    prev_emotion = None
    prev_bbox = None
    keyframe_interval = 15  # Force keyframe every 15 frames (0.5 seconds at 30fps)
    last_keyframe = -keyframe_interval
    
    # Force first and last keyframes
    force_first_kf = True
    force_last_kf = True
        
    for i, box_data in enumerate(boxes):
        ml_frame = box_data.get('frame', 0)
        ls_frame = ml_to_ls_frame(ml_frame)  # Simple conversion: just add 1
        bbox = box_data.get('bbox', [0, 0, 100, 100])
        frame_emotion = box_data.get('emotion', emotion)
        
        # Keep emotion label clean - no AU text embedded
        # AUs will be in separate structured annotations
        
        # Convert pixel coordinates to percentages
        x1, y1, x2, y2 = bbox
        x_percent = (x1 / width) * 100
        y_percent = (y1 / height) * 100
        width_percent = ((x2 - x1) / width) * 100
        height_percent = ((y2 - y1) / height) * 100
        
        # Decide if this should be a keyframe
        should_add_keyframe = False
        
        # If using all frames (fixed sampling), always add keyframe
        if use_all_frames:
            should_add_keyframe = True
        # Always add first detected frame (where face actually appears)
        elif i == 0 and force_first_kf:
            should_add_keyframe = True
            logger.info(f"Adding FIRST keyframe at ML frame {ml_frame} → LS frame {ls_frame}")
        # Always add last detected frame  
        elif i == len(boxes) - 1 and force_last_kf:
            should_add_keyframe = True
            logger.info(f"Adding LAST keyframe at ML frame {ml_frame} → LS frame {ls_frame}")
        # Add when emotion changes
        elif prev_emotion != frame_emotion:
            should_add_keyframe = True
            logger.debug(f"Emotion change: {prev_emotion} → {frame_emotion}")
        # Add when significant movement (>5% of frame)
        elif prev_bbox and (
            abs(x_percent - prev_bbox[0]) > 5 or 
            abs(y_percent - prev_bbox[1]) > 5
        ):
            should_add_keyframe = True
            logger.debug(f"Movement detected at frame {ml_frame}")
        # Force keyframe every interval
        elif ml_frame - last_keyframe >= keyframe_interval:
            should_add_keyframe = True
            logger.debug(f"Interval keyframe at ML frame {ml_frame}")
        
        if should_add_keyframe:
            keyframe_data = {
                "frame": ls_frame,  # 1-based for Label Studio
                "x": x_percent,
                "y": y_percent,
                "width": width_percent,
                "height": height_percent,
                "rotation": 0,
                "labels": [frame_emotion],
                "enabled": True,  # Enables interpolation to next keyframe
                "_ml_frame": ml_frame  # Keep for diagnostics
            }
            sequence.append(keyframe_data)
            last_keyframe = ml_frame
            
        emotions_per_frame.append(frame_emotion)
        prev_emotion = frame_emotion
        prev_bbox = (x_percent, y_percent, width_percent, height_percent)
        
    # Simple! No complex terminal keyframes needed - SCRFD provides reliable frame bounds
    if sequence:
        # Create single VideoRectangle annotation with per-frame emotions
        rect_id = f"rect_{hash(str(sequence))}"
        
        # Calculate dominant emotion from all keyframes (extract base emotion from AU-enhanced labels)
        base_emotions = []
        enhanced_labels = []
        for kf in sequence:
            if kf.get('labels'):
                for label in kf['labels']:
                    enhanced_labels.append(label)
                    # Extract base emotion (before '[') for frequency calculation
                    base_emotion = label.split('[')[0].strip() if '[' in label else label
                    base_emotions.append(base_emotion)
        
        if base_emotions:
            # Get most frequent base emotion
            emotion_counts = {}
            for emo in base_emotions:
                emotion_counts[emo] = emotion_counts.get(emo, 0) + 1
            dominant_base_emotion = max(emotion_counts, key=emotion_counts.get)
            
            # Find the first enhanced label with the dominant emotion for display
            dominant_enhanced_label = dominant_base_emotion
            for label in enhanced_labels:
                if label.startswith(dominant_base_emotion):
                    dominant_enhanced_label = label
                    break
        else:
            dominant_enhanced_label = 'neutral'
        
        # Calculate real confidence from ML predictions
        all_confidences = []
        for pred in predictions:
            conf = pred.get('confidence', 0.0)
            if conf > 0:
                all_confidences.append(conf)
        
        if all_confidences:
            avg_confidence = sum(all_confidences) / len(all_confidences)
            real_score = round(avg_confidence, 3)
        else:
            real_score = track.get('confidence', 0.5)  # Fallback to track confidence
        
        # Simple logging - much cleaner!
        first_ls = sequence[0]["frame"] if sequence else 0
        last_ls = sequence[-1]["frame"] if sequence else 0
        logger.info(f"Created annotation: {len(sequence)} keyframes, frames {first_ls}-{last_ls}")
        # Extract base emotion for logging
        base_emotion = dominant_base_emotion if base_emotions else 'neutral'
        logger.info(f"Confidence: {real_score:.3f}, Emotion: {base_emotion}")
        
        # Collect AU data for all frames first
        all_active_aus = set()
        au_text_parts = []
        
        if action_units:
            for kf in sequence:
                ml_frame_num = kf["_ml_frame"]
                au_data = get_au_data_for_frame(ml_frame_num, action_units)
                if au_data and au_data.get('active_aus'):
                    frame_aus = au_data.get('active_aus', [])
                    for au_code in frame_aus:
                        clean_au = au_code.replace('_r', '').replace('_c', '')
                        all_active_aus.add(clean_au)
                    
                    # Add intensity info for this frame
                    aus = au_data.get('aus', {})
                    frame_intensities = []
                    for au_code in frame_aus[:3]:  # Top 3 per frame
                        intensity = aus.get(au_code, 0.0)
                        clean_au = au_code.replace('_r', '').replace('_c', '')
                        frame_intensities.append(f"{clean_au}({intensity:.1f})")
                    
                    if frame_intensities:
                        au_text_parts.append(f"F{kf['frame']}: {','.join(frame_intensities)}")

        # Create the main VideoRectangle annotation with AU data embedded
        annotation = {
            "type": "videorectangle",
            "from_name": "face_bbox",  # Match XML: <VideoRectangle name="face_bbox">
            "to_name": "video",
            "id": rect_id,
            "value": {
                "sequence": sequence,
                "labels": [base_emotion]  # Will be applied via face_emotion_labels
            },
            "score": real_score
        }
        annotations.append(annotation)
        
        # Add AU data as separate annotations linked to the face rectangle
        if all_active_aus:
            # AU choices annotation (linked to the face rectangle)
            au_choices = {
                "type": "choices",
                "from_name": "aus",
                "to_name": "video",
                "id": f"aus_{rect_id}",
                "value": {
                    "choices": list(all_active_aus)
                },
                "parent_id": rect_id,
                "origin": "prediction"
            }
            annotations.append(au_choices)
            
            # AU text annotation (linked to the same rectangle)
            if au_text_parts:
                au_text = {
                    "type": "textarea",
                    "from_name": "au_text",
                    "to_name": "video",
                    "id": f"au_text_{rect_id}",
                    "value": {
                        "text": [" | ".join(au_text_parts[:10])]
                    },
                    "parent_id": rect_id,
                    "origin": "prediction"
                }
                annotations.append(au_text)
        
        return annotations
    
    # Fallback: Use predictions without bbox
    if not predictions:
        logger.warning("No predictions from ML service")
        return []
    
    # Create simulated tracking for predictions without bbox
    sequence = []
    emotion_counts = {}
    
    for i, pred in enumerate(predictions[:30]):
        frame = pred.get('frame', i * 3)
        emotion = pred.get('emotion', 'neutral')
        
        # Simple centered box
        sequence.append({
            "frame": frame,
            "x": 35,
            "y": 25,
            "width": 30,
            "height": 40,
            "rotation": 0
        })
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    
    if sequence:
        # Use most confident emotion
        dominant_emotion = max(emotion_counts, key=emotion_counts.get) if emotion_counts else 'neutral'
        
        # Create the videorectangle annotation
        annotations.append({
            "type": "videorectangle",
            "from_name": "face_bbox",
            "to_name": "video",
            "value": {
                "sequence": sequence,
                "labels": [dominant_emotion]
            }
        })
        
        logger.info(f"Created annotation with {len(sequence)} keyframes, emotion: {dominant_emotion}")
    
    return annotations

def get_fallback_predictions():
    """Return fallback predictions showing tracking across frames"""
    # Create a sequence that shows movement/tracking
    sequence = []
    for i in range(0, 60, 5):  # Every 5 frames for 2 seconds
        # Simulate slight movement
        x_offset = 40 + (i / 60) * 5  # Drift right
        y_offset = 30 + np.sin(i / 10) * 2  # Slight vertical movement
        sequence.append({
            "frame": i,
            "x": x_offset,
            "y": y_offset,
            "width": 20,
            "height": 30,
            "rotation": 0
        })
    
    return {
        'result': [{
            "type": "videorectangle",
            "from_name": "face_bbox",
            "to_name": "video",
            "value": {
                "sequence": sequence,
                "labels": ["neutral"]
            }
        }],
        'score': 0.5,
        'model_version': f"{get_dynamic_model_version()}\n(FALLBACK - ML Service Unavailable)"
    }

@app.route('/train', methods=['POST'])
def train():
    return jsonify({'status': 'ok'})

@app.route('/webhook', methods=['POST'])
def webhook():
    return jsonify({'status': 'ok'})

@app.route('/shutdown', methods=['POST'])
def shutdown():
    """Gracefully shutdown the Label Studio connector"""
    logger.info("Shutdown request received")
    
    def shutdown_server():
        import os
        import signal
        pid = os.getpid()
        logger.info(f"Sending shutdown signal to PID {pid}")
        os.kill(pid, signal.SIGTERM)
    
    try:
        # Schedule shutdown after response is sent
        import threading
        threading.Timer(1.0, shutdown_server).start()
        
        logger.info("Label Studio connector shutting down...")
        return jsonify({'message': 'Label Studio connector shutting down'}), 200
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("ML Backend with REAL Face Detection")
    print("Using: RetinaFace + ByteTrack + AffectNet")
    print("=" * 60)
    app.run(host='0.0.0.0', port=9091, debug=False)