"""
ML Service Business Logic
Core emotion recognition service and video processing functions
EXTRACTED from original app.py - preserves all functionality
"""
import os
import cv2
import json
import numpy as np
import logging
import time
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional

# Import ML models from models directory
from models.emotion_inference import EnhancedEmotionModel
from models.action_unit_detector import ActionUnitDetector, ActionUnitOutput
from models import POSTER_V2_AVAILABLE
if POSTER_V2_AVAILABLE:
    from models.poster_v2_model import PosterV2EmotionModel
from insightface.app import FaceAnalysis

logger = logging.getLogger(__name__)

# Global service instance
emotion_service = None

class SimpleTwoStageService:
    """Enhanced three-stage emotion recognition: InsightFace + Emotion Model + Action Units
    EXACT copy from original app.py
    """
    
    def __init__(self, use_poster_v2=False, enable_au_detection=False):
        self.ok = False
        self.face_detector = None
        self.emotion_model = None
        self.au_detector = None
        self.conf_threshold = 0.5
        self.use_poster_v2 = use_poster_v2  # Feature flag to switch models
        self.enable_au_detection = enable_au_detection  # Feature flag for AU detection
        
        self._init_components()
    
    def _init_components(self):
        """Initialize InsightFace detector and emotion model"""
        try:
            logger.info("Initializing InsightFace detector...")
            
            # Stage 1: InsightFace Detection (proven working approach)
            self.face_detector = FaceAnalysis(
                providers=['CPUExecutionProvider'],
                allowed_modules=['detection']
            )
            self.face_detector.prepare(ctx_id=-1, det_size=(640, 640))
            
            # Stage 2: Emotion Recognition - choose between models
            if self.use_poster_v2 and POSTER_V2_AVAILABLE:
                logger.info("Initializing POSTER V2 emotion model...")
                self.emotion_model = PosterV2EmotionModel(
                    device="cpu",
                    enable_temporal=False
                )
            else:
                if self.use_poster_v2 and not POSTER_V2_AVAILABLE:
                    logger.warning("POSTER V2 requested but not available - falling back to EfficientNet-B2")
                logger.info("Initializing EfficientNet-B2 emotion model...")
                model_path = "/home/lleyt/WIL_project/emotion_labelstudio_final/models/emotieff/fer_enet_b2_7.onnx"
                self.emotion_model = EnhancedEmotionModel(
                    onnx_path=model_path,
                    device="cpu",
                    enable_temporal=False
                )
            
            # Stage 3: Action Unit Detection (optional)
            if self.enable_au_detection:
                logger.info("Initializing Action Unit detector...")
                self.au_detector = ActionUnitDetector()
            
            self.ok = self.emotion_model.ok
            
            if self.ok:
                model_name = "POSTER V2" if self.use_poster_v2 else "EfficientNet-B2"
                au_status = "Enabled" if self.enable_au_detection else "Disabled"
                logger.info("Enhanced pipeline initialized successfully!")
                logger.info("Face Detector: InsightFace (True)")
                logger.info(f"Emotion Model: {model_name} ({self.emotion_model.ok})")
                logger.info(f"Action Units: {au_status}")
            else:
                logger.error("Failed to initialize pipeline components")
                
        except Exception as e:
            logger.error(f"Pipeline initialization failed: {e}")
            self.ok = False
    
    def process_frame(self, frame, debug_info=None):
        """Process a single frame: InsightFace detection -> Emotion recognition
        EXACT copy from original app.py
        """
        if not self.ok:
            return []
        
        debug_enabled = debug_info and debug_info.get('enabled', False)
        
        try:
            # Stage 1: InsightFace Detection (clean working approach)
            faces = self.face_detector.get(frame)
            results = []
            
            for face_idx, face in enumerate(faces):
                # InsightFace gives bbox as [x1, y1, x2, y2] and det_score
                if face.det_score >= self.conf_threshold:
                    bbox = face.bbox.astype(int)
                    x1, y1, x2, y2 = bbox
                    face_conf = face.det_score
                    
                    # Stage 2: Emotion recognition on detected face
                    if debug_enabled:
                        # Pass debug info to emotion model for step-by-step visualization
                        debug_context = {
                            'session_dir': debug_info['session_dir'],
                            'frame_idx': debug_info['frame_idx'],
                            'face_idx': face_idx,
                            'input_frame': frame
                        }
                        emotion_result = self.emotion_model.predict(frame, (x1, y1, x2, y2), debug_context)
                    else:
                        emotion_result = self.emotion_model.predict(frame, (x1, y1, x2, y2))
                    
                    emotion = emotion_result.label
                    emotion_conf = emotion_result.confidence
                    
                    results.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'emotion': emotion,
                        'confidence': float(emotion_conf),
                        'face_confidence': float(face_conf)
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Frame processing failed: {e}")
            return []

def process_video(video_path: str, frame_limit: Optional[int] = None, 
                 sample_rate: int = 3) -> Dict[str, Any]:
    """Process video using two-stage pipeline
    EXACT copy from original app.py
    """
    
    if emotion_service is None or not emotion_service.ok:
        return {'error': 'Two-stage service not initialized'}
    
    # Create unique debug session ID for this video
    video_name = Path(video_path).stem
    session_id = f"{video_name}_{int(time.time())}"
    # Use absolute path to project root for consistent debug location
    project_root = Path(__file__).parent.parent.parent.parent.parent / "refactor"
    debug_session_dir = str(project_root / "debug_preprocessed" / session_id)
    
    start_time = time.time()
    logger.info(f"Processing video with two-stage pipeline: {video_path}")
    logger.info(f"Debug session: {session_id}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return {'tracks': [], 'predictions': [], 'video_info': {}}
    
    # Video info
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    
    video_info = {
        'fps': fps,
        'total_frames': total_frames,
        'width': width,
        'height': height,
        'duration': duration
    }
    
    logger.info(f"Video info: {total_frames} frames, {fps:.1f} FPS, {width}x{height}")
    
    all_predictions = []
    frame_idx = 0
    processed_frames = 0
    debug_frame_saved = False  # Only save one debug frame per video
    best_debug_frame = None    # Track highest confidence frame for debug
    best_debug_confidence = 0.0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply frame limit
        if frame_limit and processed_frames >= frame_limit:
            break
        
        # Skip frames based on sample rate
        if frame_idx % sample_rate != 0:
            frame_idx += 1
            continue
        
        # Process frame with two-stage pipeline - collect confidence to find best debug frame
        try:
            # Process frame without debug first to get confidence
            frame_results = emotion_service.process_frame(frame)
            
            # Find highest confidence face in this frame
            frame_max_confidence = 0.0
            if frame_results:
                frame_max_confidence = max(result['confidence'] for result in frame_results)
                
                # Track best frame for debug visualization
                if frame_max_confidence > best_debug_confidence:
                    best_debug_confidence = frame_max_confidence
                    best_debug_frame = {
                        'frame_idx': frame_idx,
                        'frame': frame.copy(),
                        'results': frame_results,
                        'confidence': frame_max_confidence
                    }
            
            # Add frame info and append results
            for result in frame_results:
                result['frame'] = frame_idx
                all_predictions.append(result)
            
            processed_frames += 1
            frame_idx += 1
            
        except Exception as e:
            logger.error(f"Frame {frame_idx} processing failed: {e}")
            frame_idx += 1
            continue
    
    cap.release()
    
    # Save debug visualization for the best frame (highest confidence)
    if best_debug_frame and not debug_frame_saved:
        try:
            debug_info = {
                'enabled': True,
                'session_dir': debug_session_dir,
                'frame_idx': best_debug_frame['frame_idx']
            }
            
            # Process the best frame again with debug enabled
            logger.info(f"Saving debug for frame {best_debug_frame['frame_idx']} (confidence: {best_debug_confidence:.3f})")
            emotion_service.process_frame(best_debug_frame['frame'], debug_info)
            debug_frame_saved = True
            
        except Exception as e:
            logger.error(f"Debug frame save failed: {e}")
    
    # Action Unit Detection (if enabled and available)
    au_results = []
    if emotion_service.enable_au_detection and emotion_service.au_detector:
        try:
            logger.info("Running Action Unit detection on processed faces...")
            au_results = emotion_service.au_detector.detect_from_predictions(all_predictions, video_path)
            logger.info(f"Action Unit detection completed: {len(au_results)} results")
        except Exception as e:
            logger.error(f"Action Unit detection failed: {e}")
    
    processing_time = time.time() - start_time
    logger.info(f"Processed {processed_frames} frames in {processing_time:.2f}s")
    
    # Convert predictions to tracks format that Label Studio expects
    tracks = []
    for pred in all_predictions:
        track = {
            'id': len(tracks),  # Simple ID assignment
            'bbox': pred['bbox'],
            'emotion': pred['emotion'],
            'confidence': pred['confidence'],
            'frame': pred['frame']
        }
        tracks.append(track)
    
    # Package results in expected format (include AU data if available)
    results = {
        'predictions': all_predictions,
        'video_info': video_info,
        'tracks': tracks
    }
    
    # Add AU results if available
    if au_results:
        results['action_units'] = [au.to_dict() for au in au_results]
        results['au_analysis_info'] = {
            'method': 'openface' if emotion_service.au_detector.openface_available else 'fallback',
            'total_au_frames': len(au_results),
            'au_types_detected': len(set(au for au_result in au_results for au in au_result.get_active_aus()))
        }
    
    # Save JSON debug output showing the exported data structure
    try:
        import os
        os.makedirs(debug_session_dir, exist_ok=True)
        
        # Create comprehensive debug JSON with sample data and statistics
        debug_json = {
            'video_processing_summary': {
                'video_path': video_path,
                'video_name': video_name,
                'session_id': session_id,
                'processing_time_seconds': processing_time,
                'total_detections': len(all_predictions),
                'frames_processed': processed_frames,
                'sample_rate': sample_rate
            },
            'debug_frame_info': {
                'selected_frame': best_debug_frame['frame_idx'] if best_debug_frame else None,
                'selection_confidence': best_debug_confidence,
                'selection_reason': 'highest_confidence',
                'debug_files_saved': debug_frame_saved,
                'debug_session_dir': debug_session_dir
            },
            'video_info': video_info,
            'detection_statistics': {
                'emotions_detected': {},
                'confidence_distribution': {
                    'high_confidence': 0,  # > 0.8
                    'medium_confidence': 0,  # 0.5-0.8
                    'low_confidence': 0     # < 0.5
                },
                'frame_distribution': {}
            },
            'sample_predictions': all_predictions[:10] if all_predictions else [],  # First 10 predictions as sample
            'tracks_format': tracks[:10] if tracks else [],  # First 10 tracks as sample
            'full_results': results  # Complete results for debugging
        }
        
        # Calculate statistics
        for pred in all_predictions:
            emotion = pred['emotion']
            confidence = pred['confidence']
            frame = pred['frame']
            
            # Emotion distribution
            debug_json['detection_statistics']['emotions_detected'][emotion] = \
                debug_json['detection_statistics']['emotions_detected'].get(emotion, 0) + 1
            
            # Confidence distribution
            if confidence > 0.8:
                debug_json['detection_statistics']['confidence_distribution']['high_confidence'] += 1
            elif confidence > 0.5:
                debug_json['detection_statistics']['confidence_distribution']['medium_confidence'] += 1
            else:
                debug_json['detection_statistics']['confidence_distribution']['low_confidence'] += 1
            
            # Frame distribution
            debug_json['detection_statistics']['frame_distribution'][frame] = \
                debug_json['detection_statistics']['frame_distribution'].get(frame, 0) + 1
        
        # Save debug JSON
        json_path = os.path.join(debug_session_dir, 'inference_results.json')
        with open(json_path, 'w') as f:
            json.dump(debug_json, f, indent=2)
        
        logger.info(f"DEBUG: Saved inference results JSON to {json_path}")
        
    except Exception as e:
        logger.error(f"Failed to save debug JSON: {e}")
    
    # If everything worked and we have results, save the best debug frame
    if debug_frame_saved and best_debug_frame:
        try:
            # Instruct emotion model to save its best debug frame for this session
            emotion_service.emotion_model.save_best_debug_frame(session_dir=debug_session_dir)
            
        except Exception as e:
            logger.error(f"Failed to trigger best debug frame save: {e}")
    
    return results

def get_emotion_service():
    """Get global emotion service instance"""
    return emotion_service

def initialize_emotion_service():
    """Initialize the global emotion service instance
    Called from app.py on startup
    """
    global emotion_service
    
    # Check environment flags for model selection
    use_poster_v2 = os.getenv('USE_POSTER_V2', 'false').lower() == 'true'
    enable_au_detection = os.getenv('ENABLE_AU_DETECTION', 'false').lower() == 'true'
    
    model_name = "POSTER V2" if use_poster_v2 else "EfficientNet-B2"
    au_status = " + Action Units" if enable_au_detection else ""
    
    logger.info("=" * 60)
    logger.info(f"Enhanced Emotion ML Service - InsightFace + {model_name}{au_status}")
    logger.info("=" * 60)
    
    # Initialize the enhanced service
    logger.info(f"Initializing InsightFace + {model_name} service...")
    emotion_service = SimpleTwoStageService(use_poster_v2=use_poster_v2, enable_au_detection=enable_au_detection)
    
    if emotion_service.ok:
        logger.info(f"InsightFace + {model_name} service ready!")
    else:
        logger.error("Failed to initialize service")
    
    return emotion_service