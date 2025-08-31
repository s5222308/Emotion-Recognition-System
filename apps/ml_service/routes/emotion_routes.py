"""
Emotion Recognition Routes
Handles video processing and emotion prediction endpoints
"""
import os
import logging
from flask import Blueprint, request, jsonify
from services.ml_service import get_emotion_service, process_video

logger = logging.getLogger(__name__)

emotion_bp = Blueprint('emotion', __name__)

@emotion_bp.route('/prelabel', methods=['POST'])
def prelabel():
    """Process video and return emotion predictions"""
    
    emotion_service = get_emotion_service()
    if emotion_service is None or not emotion_service.ok:
        return jsonify({'error': 'Two-stage service not initialized'}), 503
    
    data = request.json
    video_path = data.get('video_path', '')
    frame_limit = data.get('frame_limit', None)
    sample_rate = data.get('sample_rate', 3)
    
    if not video_path or not os.path.exists(video_path):
        return jsonify({'error': 'Video path not found'}), 400
    
    try:
        results = process_video(video_path, frame_limit, sample_rate)
        return jsonify(results)
    
    except Exception as e:
        logger.exception(f"Error processing video: {e}")
        return jsonify({'error': str(e)}), 500