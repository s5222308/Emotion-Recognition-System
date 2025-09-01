"""
Health and Status Routes
System health monitoring and service control endpoints
PRESERVES EXACT functionality from original app.py
"""
import logging
import threading
import os
from flask import Blueprint, jsonify, request
from services.ml_service import get_emotion_service

logger = logging.getLogger(__name__)

health_bp = Blueprint('health', __name__)

@health_bp.route('/health', methods=['GET'])
def health():
    """Health check endpoint - EXACT copy from original app.py"""
    try:
        emotion_service = get_emotion_service()
        if emotion_service is None:
            return jsonify({'status': 'initializing'}), 503
        
        if hasattr(emotion_service, 'get_model_info'):
            model_info = emotion_service.get_model_info()
            health_info = {
                'status': 'healthy' if emotion_service.ok else 'error',
                'face_detector': model_info.get('face_detector', 'Unknown'),
                'emotion_model': model_info.get('emotion_model', 'Unknown'),
                'au_detector': model_info.get('au_detector', 'None'),
                'service': 'Universal ML Service'
            }
        else:
            health_info = {
                'status': 'healthy' if emotion_service.ok else 'error',
                'face_detector': 'Legacy',
                'emotion_model': 'Legacy',
                'service': 'Legacy Service'
            }
        
        if emotion_service.ok:
            return jsonify(health_info), 200
        else:
            return jsonify(health_info), 503
            
    except Exception as e:
        logger.exception(f"Health check failed: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 503

@health_bp.route('/status', methods=['GET'])
def status():
    """Status page with UI - EXACT copy from original app.py"""
    try:
        emotion_service = get_emotion_service()
        if emotion_service is None:
            service_status = "initializing"
            status_class = "warning"
        elif emotion_service.ok:
            service_status = "healthy"
            status_class = "healthy"
        else:
            service_status = "error"
            status_class = "error"
        
        html = f"""<!DOCTYPE html>
        <html>
        <head>
            <title>ML Visual Inference Service - Status</title>
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
                    <h1>ML Visual Inference Service</h1>
                    <h2>Emotion Recognition Engine</h2>
                </div>
                
                <div class="status {status_class}">
                    <h3>Service Status: {service_status.upper()}</h3>
                    <p>Port: 5003 | Pipeline: Two-Stage (SCRFD + EfficientNet-B2)</p>
                </div>
                
                <div class="component">
                    <h4>Model Information</h4>
                    <div class="metric">
                        <div class="metric-value">InsightFace</div>
                        <div class="metric-label">Face Detector</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">EfficientNet-B2</div>
                        <div class="metric-label">Emotion Model</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">MediaPipe</div>
                        <div class="metric-label">Face Alignment</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">260x260</div>
                        <div class="metric-label">Input Size</div>
                    </div>
                </div>
                
                <div class="component">
                    <h4>Service Information</h4>
                    <p><strong>Purpose:</strong> Two-stage emotion recognition for video analysis</p>
                    <p><strong>API Endpoints:</strong> /health, /prelabel, /shutdown</p>
                    <p><strong>Debug Output:</strong> debug_preprocessed/[video_session]/</p>
                    <p><strong>Supported Emotions:</strong> angry, disgust, fear, happy, sad, surprise, neutral</p>
                </div>
                
                <div class="component">
                    <h4>Pipeline Details</h4>
                    <p><strong>Stage 1:</strong> Face detection using InsightFace SCRFD</p>
                    <p><strong>Stage 2:</strong> Face alignment with MediaPipe FaceMesh (468 landmarks)</p>
                    <p><strong>Stage 3:</strong> Emotion classification with EfficientNet-B2</p>
                    <p><strong>Processing:</strong> Confidence threshold = 0.5, Debug frames saved per video</p>
                </div>
                
                <div class="info" style="text-align: center; margin-top: 30px; padding-top: 20px; border-top: 1px solid #dee2e6;">
                    <a href="/health" style="color: #007bff;">JSON Health API</a> | 
                    <a href="http://localhost:8081" style="color: #007bff;">Dashboard</a>
                </div>
            </div>
        </body>
        </html>"""
        return html
    except Exception as e:
        return f"<h1>Status Error</h1><p>{str(e)}</p>", 500

@health_bp.route('/shutdown', methods=['POST'])
def shutdown():
    """Shutdown endpoint for graceful service termination - EXACT copy from original app.py"""
    try:
        logger.info("Shutdown request received")
        
        def shutdown_server():
            # Try Werkzeug shutdown first
            func = request.environ.get('werkzeug.server.shutdown')
            if func is not None:
                func()
            else:
                # Alternative: Send SIGTERM to self
                import os
                import signal
                os.kill(os.getpid(), signal.SIGTERM)
        
        # Run shutdown in separate thread to return response first
        threading.Thread(target=shutdown_server, daemon=True).start()
        
        return jsonify({'message': 'Service shutting down...'}), 200
        
    except Exception as e:
        logger.exception(f"Shutdown failed: {e}")
        return jsonify({'error': str(e)}), 500

@health_bp.route('/models/current', methods=['GET'])
def get_current_models():
    """Get current active models"""
    try:
        emotion_service = get_emotion_service()
        if emotion_service and hasattr(emotion_service, 'get_model_info'):
            model_info = emotion_service.get_model_info()
            return jsonify(model_info), 200
        else:
            return jsonify({'error': 'Service not available'}), 503
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@health_bp.route('/models/switch', methods=['POST'])
def switch_models():
    """Switch active models"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        emotion_service = get_emotion_service()
        if emotion_service and hasattr(emotion_service, 'switch_models'):
            face_detector = data.get('face_detector')
            emotion_model = data.get('emotion_model')
            sample_rate = data.get('sample_rate')
            
            result = emotion_service.switch_models(face_detector, emotion_model, sample_rate)
            return jsonify({'success': True, 'models': result}), 200
        else:
            return jsonify({'error': 'Service not available'}), 503
    except Exception as e:
        return jsonify({'error': str(e)}), 500