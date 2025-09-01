"""
ML Service Business Logic
Universal ML Service using ONNX-only models
ANY ONNX model can be plugged in via config file!
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

# Import our universal ONNX system
from models.universal_onnx import UniversalONNXModel, ModelOutput, create_emotion_output

# Legacy imports for action unit support
from models.action_unit_detector import ActionUnitDetector, ActionUnitOutput

# Import temporal processing utilities
from utils.video_processing import TemporalProcessor, AdaptiveSampling, detect_emotion_transitions
from insightface.app import FaceAnalysis

logger = logging.getLogger(__name__)

# Global service instance
emotion_service = None

class UniversalMLService:
    """
    Universal ML Service using ONNX-only models
    - Face detection: ANY ONNX face detector
    - Emotion recognition: ANY ONNX emotion model
    - Completely configurable via JSON registry
    """
    
    def __init__(self, registry_path: str = None):
        self.ok = False
        self.face_detector = None
        self.emotion_model = None
        self.conf_threshold = 0.5
        
        # Set registry path - use absolute path to project root
        if registry_path is None:
            project_root = Path(__file__).parent.parent.parent.parent
            self.registry_path = str(project_root / "config" / "model_registry.json")
        else:
            self.registry_path = registry_path
            
        self.registry = None
        self.active_config = None
        
        # Action Unit detection support
        self.enable_au_detection = False
        self.au_detector = None
        self.au_detector_type = None
        
        self._load_registry()
        self._init_models()
    
    def _load_registry(self):
        """Load model registry configuration"""
        try:
            with open(self.registry_path, 'r') as f:
                self.registry = json.load(f)
            
            self.active_config = self.registry.get('active_config', {})
            logger.info(f"✓ Model registry loaded: {self.registry_path}")
            logger.info(f"  Active face detector: {self.active_config.get('face_detector')}")
            logger.info(f"  Active emotion model: {self.active_config.get('emotion_model')}")
            logger.info(f"  Sample rate: {self.active_config.get('sample_rate', 3)}")
            
        except Exception as e:
            logger.error(f"Failed to load registry {self.registry_path}: {e}")
            raise
    
    def _init_models(self):
        """Initialize models from registry"""
        try:
            # Initialize face detector
            face_detector_id = self.active_config.get('face_detector')
            if face_detector_id and face_detector_id in self.registry.get('face_detectors', {}):
                face_config = self.registry['face_detectors'][face_detector_id]
                logger.info(f"Initializing face detector: {face_config['name']}")
                
                # For now, keep using InsightFace until we have ONNX version
                # TODO: Replace with universal ONNX face detector
                if face_detector_id == 'insightface_onnx':
                    # Fallback to old InsightFace until we convert it
                    from insightface.app import FaceAnalysis
                    self.face_detector = FaceAnalysis(
                        providers=['CPUExecutionProvider'],
                        allowed_modules=['detection']
                    )
                    self.face_detector.prepare(ctx_id=-1, det_size=(640, 640))
                else:
                    # Use universal ONNX face detector
                    self.face_detector = UniversalONNXModel(face_config)
                
                logger.info(f"✓ Face detector initialized: {face_config['name']}")
            
            # Initialize emotion model
            emotion_model_id = self.active_config.get('emotion_model')
            if emotion_model_id and emotion_model_id in self.registry.get('emotion_models', {}):
                emotion_config = self.registry['emotion_models'][emotion_model_id]
                logger.info(f"Initializing emotion model: {emotion_config['name']}")
                
                # Use universal ONNX emotion model
                self.emotion_model = UniversalONNXModel(emotion_config)
                
                if self.emotion_model.ok:
                    logger.info(f"✓ Emotion model initialized: {emotion_config['name']}")
                else:
                    raise Exception(f"Failed to initialize emotion model: {emotion_config['name']}")
            
            # Initialize AU detector
            au_detector_id = self.active_config.get('au_detector')
            if au_detector_id and au_detector_id in self.registry.get('au_detectors', {}):
                au_config = self.registry['au_detectors'][au_detector_id]
                if au_config.get('enabled', False):
                    logger.info(f"Initializing AU detector: {au_config['name']}")
                    self.au_detector_type = au_config.get('type')
                    
                    if self.au_detector_type == 'external':
                        # Use existing OpenFace-based detector
                        from models.action_unit_detector import ActionUnitDetector
                        self.au_detector = ActionUnitDetector(au_config.get('command_path'))
                        self.enable_au_detection = True
                    elif self.au_detector_type == 'derived':
                        # Use emotion-based AU estimation
                        self.enable_au_detection = True
                        self.au_detector = None  # Will use emotion predictions
                    elif self.au_detector_type == 'onnx':
                        # Future: Use ONNX AU detector
                        # self.au_detector = UniversalONNXModel(au_config)
                        logger.warning(f"ONNX AU detector not yet implemented: {au_config['name']}")
                    
                    if self.enable_au_detection:
                        logger.info(f"✓ AU detector initialized: {au_config['name']}")
            
            # Service is ready if we have both models
            self.ok = (self.face_detector is not None and 
                      self.emotion_model is not None and 
                      self.emotion_model.ok)
            
            if self.ok:
                face_name = self.registry['face_detectors'][face_detector_id]['name']
                emotion_name = self.registry['emotion_models'][emotion_model_id]['name']
                au_name = "None"
                if au_detector_id and au_detector_id in self.registry.get('au_detectors', {}):
                    au_name = self.registry['au_detectors'][au_detector_id]['name']
                logger.info("Universal ML Service ready!")
                logger.info(f"  Face Detector: {face_name}")
                logger.info(f"  Emotion Model: {emotion_name}")
                logger.info(f"  AU Detector: {au_name}")
                logger.info("  Architecture: Universal ONNX")
            else:
                logger.error("Failed to initialize Universal ML Service")
                
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            self.ok = False
    
    def process_frame(self, frame, debug_info=None):
        """Process a single frame with universal ONNX models"""
        if not self.ok:
            return []
        
        try:
            # Stage 1: Face Detection
            # For now, use InsightFace API (will be replaced with universal ONNX)
            if hasattr(self.face_detector, 'get'):  # InsightFace
                faces = self.face_detector.get(frame)
                results = []
                
                for face_idx, face in enumerate(faces):
                    if face.det_score >= self.conf_threshold:
                        bbox = face.bbox.astype(int)
                        x1, y1, x2, y2 = bbox
                        face_conf = face.det_score
                        
                        # Stage 2: Universal ONNX Emotion Recognition
                        emotion_output = self.emotion_model.predict(frame, [x1, y1, x2, y2])
                        
                        results.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'emotion': emotion_output.label,
                            'confidence': float(emotion_output.confidence),
                            'face_confidence': float(face_conf)
                        })
                        
            else:  # Universal ONNX face detector
                face_detections = self.face_detector.predict(frame)
                results = []
                
                for detection in face_detections:
                    if detection.confidence >= self.conf_threshold:
                        x1, y1, x2, y2 = detection.bbox
                        
                        # Stage 2: Universal ONNX Emotion Recognition  
                        emotion_output = self.emotion_model.predict(frame, [x1, y1, x2, y2])
                        
                        results.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'emotion': emotion_output.label,
                            'confidence': float(emotion_output.confidence),
                            'face_confidence': float(detection.confidence)
                        })
            
            return results
            
        except Exception as e:
            logger.error(f"Frame processing failed: {e}")
            return []
    
    def get_model_info(self) -> Dict[str, str]:
        """Get current model information"""
        if not self.registry or not self.active_config:
            return {"face_detector": "Unknown", "emotion_model": "Unknown", "au_detector": "None"}
        
        face_detector_id = self.active_config.get('face_detector', 'unknown')
        emotion_model_id = self.active_config.get('emotion_model', 'unknown')
        au_detector_id = self.active_config.get('au_detector', 'none')
        
        face_name = "Unknown"
        emotion_name = "Unknown"
        au_name = "None"
        
        if face_detector_id in self.registry.get('face_detectors', {}):
            face_name = self.registry['face_detectors'][face_detector_id]['name']
        
        if emotion_model_id in self.registry.get('emotion_models', {}):
            emotion_name = self.registry['emotion_models'][emotion_model_id]['name']
        
        if au_detector_id in self.registry.get('au_detectors', {}):
            au_name = self.registry['au_detectors'][au_detector_id]['name']
        
        return {
            "face_detector": face_name,
            "emotion_model": emotion_name,
            "au_detector": au_name,
            "architecture": "Universal ONNX"
        }
    
    def switch_models(self, face_detector_id: str = None, emotion_model_id: str = None, sample_rate: int = None):
        """Switch to different models and settings without restarting service"""
        try:
            logger.info(f"Switching models - face: {face_detector_id}, emotion: {emotion_model_id}, sample_rate: {sample_rate}")
            
            # Update active config
            if face_detector_id:
                self.active_config['face_detector'] = face_detector_id
            if emotion_model_id:
                self.active_config['emotion_model'] = emotion_model_id
            if sample_rate is not None:
                self.active_config['sample_rate'] = sample_rate
                logger.info(f"Updated sample rate to: {sample_rate}")
            
            # Save updated config
            self.registry['active_config'] = self.active_config
            with open(self.registry_path, 'w') as f:
                json.dump(self.registry, f, indent=2)
            
            # Reinitialize models
            self._init_models()
            
            logger.info("✓ Models switched successfully!")
            return self.get_model_info()
            
        except Exception as e:
            logger.error(f"Failed to switch models: {e}")
            raise

def process_video(video_path: str, frame_limit: Optional[int] = None, 
                 sample_rate: int = 3) -> Dict[str, Any]:
    """Process video using universal ONNX pipeline"""
    
    if emotion_service is None or not emotion_service.ok:
        return {'error': 'Universal ML service not initialized'}
    
    # Create unique debug session ID for this video
    video_name = Path(video_path).stem
    session_id = f"{video_name}_{int(time.time())}"
    # Use absolute path to project root for consistent debug location
    project_root = Path(__file__).parent.parent.parent.parent.parent / "refactor"
    debug_session_dir = str(project_root / "debug_preprocessed" / session_id)
    
    start_time = time.time()
    logger.info(f"Processing video with Universal ONNX pipeline: {video_path}")
    logger.info(f"Debug session: {session_id}")
    logger.info(f"Sample rate: {sample_rate} (processing every {sample_rate} frames)")
    
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
    
    # Initialize temporal processing
    temporal_processor = TemporalProcessor(window_size=15)  # 0.5s at 30fps
    adaptive_sampler = AdaptiveSampling(min_interval=1, max_interval=10)
    last_processed_idx = -1
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply frame limit
        if frame_limit and processed_frames >= frame_limit:
            break
        
        # Use fixed sample rate if specified, otherwise use adaptive sampling
        if sample_rate > 1:
            # Fixed sampling
            if frame_idx % sample_rate != 0:
                frame_idx += 1
                continue
        else:
            # Adaptive sampling only when sample_rate = 1
            if not adaptive_sampler.should_process_frame(frame, frame_idx, last_processed_idx):
                frame_idx += 1
                continue
        
        last_processed_idx = frame_idx
        
        # Process frame with Universal ONNX pipeline - collect confidence to find best debug frame
        try:
            # Process frame without debug first to get confidence
            frame_results = emotion_service.process_frame(frame)
            
            # Apply temporal smoothing to frame results
            if frame_results:
                frame_results = temporal_processor.add_frame_predictions(
                    frame_results, frame_idx
                )
            
            # Find highest confidence face in this frame
            frame_max_confidence = 0.0
            if frame_results:
                # Use smoothed confidence if available, otherwise regular confidence
                frame_max_confidence = max(
                    result.get('smoothed_confidence', result['confidence']) 
                    for result in frame_results
                )
                
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
    
    # Get video summary from temporal processor
    video_summary = temporal_processor.get_video_summary()
    
    # Detect emotion transitions
    transitions = detect_emotion_transitions(all_predictions, min_duration=5)
    
    # Package results in expected format (include AU data if available)
    results = {
        'predictions': all_predictions,
        'video_info': video_info,
        'tracks': tracks,
        'temporal_analysis': {
            'summary': video_summary,
            'transitions': transitions,
            'smoothing_applied': True,
            'face_tracks': len(temporal_processor.face_tracks)
        }
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
    """Initialize the global Universal ML service instance"""
    global emotion_service
    
    logger.info("=" * 60)
    logger.info("Universal ML Service - ONNX Only Architecture")
    logger.info("ANY model can be plugged in via config!")
    logger.info("=" * 60)
    
    # Initialize the universal service
    emotion_service = UniversalMLService()
    
    if emotion_service.ok:
        model_info = emotion_service.get_model_info()
        logger.info(f"Universal ML Service ready!")
        logger.info(f"Face Detector: {model_info['face_detector']}")
        logger.info(f"Emotion Model: {model_info['emotion_model']}")
    else:
        logger.error("Failed to initialize Universal ML Service")
    
    return emotion_service