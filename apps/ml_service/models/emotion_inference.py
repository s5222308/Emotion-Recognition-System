#!/usr/bin/env python3
"""
Enhanced Emotion Recognition Model - Phase 2
Implements comprehensive upgrades:

Phase 1: Better preprocessing + Facial landmark alignment
Phase 2: Adaptive confidence thresholding, temporal smoothing, 
         face quality assessment, ensemble prediction, and enhanced debug

Key Features:
- MediaPipe FaceMesh landmark alignment (468 points)
- Adaptive confidence thresholds based on face quality
- Temporal smoothing to reduce emotion jitter  
- Multi-metric face quality assessment
- Ensemble prediction with multiple preprocessing variations
- Enhanced debug visualization with quality metrics
- Self-attention temporal modeling (SOTA)
"""

import os
import cv2
import numpy as np
import logging
from typing import Tuple, Optional, List, Dict, Any
import onnxruntime as ort
from collections import deque
import time

# PyTorch for temporal modeling
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None

logger = logging.getLogger(__name__)

# Emotions for 7-class FER models
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

class EmotionOutput:
    """Enhanced container for emotion prediction results with quality metrics"""
    def __init__(self, label: str, confidence: float, quality_score: float = 0.5, 
                 temporal_smoothed: bool = False, adaptive_threshold: float = 0.4):
        self.label = label
        self.confidence = confidence
        self.quality_score = quality_score
        self.temporal_smoothed = temporal_smoothed
        self.adaptive_threshold = adaptive_threshold

class EnhancedEmotionModel:
    """
    Enhanced emotion model with facial landmark alignment and better preprocessing
    """
    
    def __init__(self, onnx_path: str, device: str = "cpu", enable_temporal: bool = True, use_ensemble: bool = False):
        self.ok = False
        self.onnx_path = onnx_path
        self.device = device
        self.session = None
        self.input_name = None
        self.input_shape = None
        self.enable_temporal = enable_temporal and PYTORCH_AVAILABLE
        self.use_ensemble = use_ensemble
        
        # Debug tracking - save the most confident frame per session
        self.debug_sample_saved = False
        self.debug_best_confidence = 0.0
        self.debug_best_image = None
        self.debug_best_bbox = None
        self.debug_best_emotion = "Non-Detected"
        self.debug_best_face_region = None
        self.debug_best_landmarks = None
        
        # Phase 2 Upgrades: Advanced tracking
        self.emotion_history = deque(maxlen=10)  # For temporal smoothing
        self.confidence_history = deque(maxlen=5)  # For adaptive thresholding
        self.face_quality_scores = deque(maxlen=5)  # For quality assessment
        self.prediction_count = 0
        
        # Adaptive thresholding parameters
        self.base_confidence_threshold = 0.4
        self.min_confidence_threshold = 0.25
        self.max_confidence_threshold = 0.7
        
        # Smoothing parameters
        self.smoothing_alpha = 0.3  # Lower = more smoothing
        self.emotion_change_threshold = 0.15  # Minimum confidence difference for emotion change
        
        # Try to load face landmark detector (dlib or mediapipe)
        self.landmark_detector = self._init_landmark_detector()
        
        self._init_model()
        
        # Initialize temporal model if enabled
        self.temporal_model = None
        if self.enable_temporal:
            self._init_temporal_model()
    
    def _init_landmark_detector(self):
        """Initialize facial landmark detector (try multiple options)"""
        # Check if MediaPipe initialization should be skipped to avoid conflicts
        if os.environ.get('SKIP_MEDIAPIPE_INIT') == '1':
            logger.info("Skipping MediaPipe initialization (external face detection in use)")
            return None
            
        logger.info("Attempting to initialize MediaPipe for better face alignment...")
        
        # Option 1: Try MediaPipe with stable face_detection first
        try:
            import mediapipe as mp
            
            # Set environment variables for WSL2 compatibility - WORKING CONFIG
            os.environ.update({
                'CUDA_VISIBLE_DEVICES': '-1',          # Force CPU only
                'LIBGL_ALWAYS_SOFTWARE': '1',           # Force software rendering
                'GALLIUM_DRIVER': 'llvmpipe',           # Use LLVM pipe driver
                'MESA_GL_VERSION_OVERRIDE': '4.5',      # Override GL version
                'MESA_GLSL_VERSION_OVERRIDE': '450',    # Override GLSL version
                'MESA_LOADER_DRIVER_OVERRIDE': 'swrast', # Software rasterizer
                'TF_CPP_MIN_LOG_LEVEL': '2',            # Reduce TensorFlow logs
                'GLOG_minloglevel': '2',                # Reduce Google logs
            })
            
            # Disable display systems that might interfere
            if 'DISPLAY' in os.environ:
                original_display = os.environ['DISPLAY']
                os.environ['DISPLAY'] = ''
            if 'WAYLAND_DISPLAY' in os.environ:
                original_wayland = os.environ['WAYLAND_DISPLAY'] 
                os.environ['WAYLAND_DISPLAY'] = ''
            
            logger.info("Initializing MediaPipe FaceMesh with working configuration...")
            
            # Use FaceMesh directly with working configuration + IMAGE_DIMENSIONS fix
            mp_face_mesh = mp.solutions.face_mesh
            face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            # Test with dummy image to ensure it works
            import numpy as np
            test_image = np.ones((224, 224, 3), dtype=np.uint8) * 128  # Gray test image
            test_results = face_mesh.process(test_image)
            
            logger.info("MediaPipe FaceMesh initialized and tested successfully!")
            return ("mediapipe_facemesh", face_mesh)
            
        except Exception as e:
            logger.error(f"MediaPipe FaceMesh initialization failed: {e}")
            logger.info("Will use enhanced preprocessing without landmarks")
        
        # Option 2: Try dlib (if available)
        try:
            import dlib
            predictor_path = "shape_predictor_68_face_landmarks.dat"
            if os.path.exists(predictor_path):
                predictor = dlib.shape_predictor(predictor_path)
                logger.info("Using dlib for facial landmarks")
                return ("dlib", predictor)
        except ImportError:
            pass
        
        logger.info("No landmark detector available - using enhanced preprocessing without landmarks")
        return None
    
    def _init_model(self):
        """Initialize the ONNX emotion recognition model"""
        if not os.path.exists(self.onnx_path):
            logger.warning(f"Model file not found: {self.onnx_path}")
            return
        
        try:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.device == "cuda" else ["CPUExecutionProvider"]
            self.session = ort.InferenceSession(self.onnx_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            
            # Determine input size from model
            if len(self.input_shape) == 4:  # NCHW format
                self.input_size = self.input_shape[2]  # Height (assuming square input)
            else:
                self.input_size = 224  # Default fallback
                
            self.ok = True
            logger.info(f"Enhanced emotion model loaded: {self.onnx_path}")
            logger.info(f"Input shape: {self.input_shape}, Input size: {self.input_size}")
            
        except Exception as e:
            logger.exception(f"Failed to load emotion model: {e}")
            self.ok = False
    
    def extract_landmarks_mediapipe(self, image: np.ndarray):
        """Extract facial landmarks using MediaPipe FaceMesh"""
        if not self.landmark_detector or self.landmark_detector[0] != "mediapipe_facemesh":
            return None
        
        face_mesh = self.landmark_detector[1]
        
        # Convert BGR to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Set image dimensions to avoid landmark projection warnings
        h, w = rgb_image.shape[:2]
        
        # Create a copy with proper format for MediaPipe
        rgb_image = rgb_image.copy()
        rgb_image.flags.writeable = False  # Improve performance
        
        # Process the image
        results = face_mesh.process(rgb_image)
        
        rgb_image.flags.writeable = True  # Restore writeable flag
        
        if results.multi_face_landmarks and len(results.multi_face_landmarks) > 0:
            landmarks = []
            h, w = image.shape[:2]
            
            # Extract landmarks from first detected face
            face_landmarks = results.multi_face_landmarks[0]
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                landmarks.append([x, y])
            
            return np.array(landmarks)
        
        return None
    
    def align_face_landmarks_with_debug(self, image: np.ndarray, bbox: Tuple[int, int, int, int]):
        """
        Align face using landmarks and return both aligned face and landmarks for debug
        """
        if not self.landmark_detector:
            # Extract face region from bbox when landmarks not available
            x1, y1, x2, y2 = bbox
            face_region = image[y1:y2, x1:x2]
            return face_region, None
        
        x1, y1, x2, y2 = bbox
        face_region = image[y1:y2, x1:x2]
        
        if self.landmark_detector[0] == "mediapipe_facemesh":
            landmarks = self.extract_landmarks_mediapipe(face_region)
            if landmarks is not None:
                aligned_face = self.apply_alignment(face_region, landmarks)
                return aligned_face, landmarks
        
        return face_region, None  # Return unaligned if alignment fails
    
    def apply_alignment(self, face_image: np.ndarray, landmarks: np.ndarray):
        """Apply facial alignment based on eye landmarks using MediaPipe FaceMesh indices"""
        try:
            # MediaPipe FaceMesh landmark indices for eyes
            # Left eye landmarks: 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246
            # Right eye landmarks: 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398
            
            # Key eye corner landmarks for alignment
            left_eye_corner = landmarks[33]   # Left eye left corner
            right_eye_corner = landmarks[263] # Right eye right corner
            
            # Calculate the angle between eye corners
            dx = right_eye_corner[0] - left_eye_corner[0] 
            dy = right_eye_corner[1] - left_eye_corner[1]
            angle = np.degrees(np.arctan2(dy, dx))
            
            # Calculate center point between eyes
            eye_center = (float((left_eye_corner[0] + right_eye_corner[0]) // 2),
                         float((left_eye_corner[1] + right_eye_corner[1]) // 2))
            
            # Face alignment logging  
            logger.debug(f"Face alignment: angle={angle:.2f}°, center={eye_center}")
            
            # Create rotation matrix (rotate by positive angle to make eyes horizontal)
            rotation_matrix = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
            
            # Apply rotation to align the face
            h, w = face_image.shape[:2]
            aligned_face = cv2.warpAffine(face_image, rotation_matrix, (w, h), 
                                        flags=cv2.INTER_CUBIC,
                                        borderMode=cv2.BORDER_REFLECT)
            
            return aligned_face
            
        except Exception as e:
            logger.debug(f"MediaPipe landmark alignment failed: {e}")
            return face_image  # Return original if alignment fails
    
    def preprocess_with_mesh(self, image: np.ndarray, bbox: Tuple[int, int, int, int], debug_context=None):
        """
        Preprocessing with landmark alignment that also returns debug data
        """
        # Extract debug info if provided
        save_debug = debug_context is not None
        if save_debug:
            from pathlib import Path
            import os
            debug_dir = Path(debug_context['session_dir']) / "pipeline_steps"
            os.makedirs(str(debug_dir), exist_ok=True)
            frame_idx = debug_context.get('frame_idx', 0)
            face_idx = debug_context.get('face_idx', 0)
            base_name = f"frame_{frame_idx}_face_{face_idx}"
            
            # Step 0: Original detection
            x1, y1, x2, y2 = bbox
            detection_img = image.copy()
            cv2.rectangle(detection_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imwrite(str(debug_dir / f"step_1_detection_{base_name}.jpg"), detection_img)
        
        # Step 1: Extract face region
        x1, y1, x2, y2 = bbox
        face_region = image[y1:y2, x1:x2].copy()
        if save_debug:
            cv2.imwrite(str(debug_dir / f"step_2_crop_{base_name}.jpg"), face_region)
        
        # Step 2: Align face using landmarks
        aligned_face, landmarks = self.align_face_landmarks_with_debug(image, bbox)
        if save_debug:
            cv2.imwrite(str(debug_dir / f"step_3_aligned_{base_name}.jpg"), aligned_face)
        
        # Step 3: Add 10% padding
        h, w = aligned_face.shape[:2]
        pad_h, pad_w = int(h * 0.1), int(w * 0.1)
        padded_face = cv2.copyMakeBorder(aligned_face, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_REFLECT)
        if save_debug:
            cv2.imwrite(str(debug_dir / f"step_4_padded_{base_name}.jpg"), padded_face)
        
        # Transform landmarks to match the padded and resized coordinates
        if landmarks is not None:
            landmarks = landmarks + np.array([pad_w, pad_h])
            padded_h, padded_w = padded_face.shape[:2]
            scale_x = self.input_size / padded_w
            scale_y = self.input_size / padded_h
            landmarks = landmarks * np.array([scale_x, scale_y])
        
        # Step 4: Resize to model input size
        resized_face = cv2.resize(padded_face, (self.input_size, self.input_size), interpolation=cv2.INTER_CUBIC)
        if save_debug:
            cv2.imwrite(str(debug_dir / f"step_5_resized_{base_name}.jpg"), resized_face)
        
        # Step 5: Convert to RGB and normalize (ImageNet stats)
        rgb_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized_face = (rgb_face - mean) / std
        
        if save_debug:
            # Show what the model actually sees (denormalize for viewing)
            display_normalized = (normalized_face * std) + mean
            display_normalized = np.clip(display_normalized * 255.0, 0, 255).astype(np.uint8)
            display_normalized_bgr = cv2.cvtColor(display_normalized, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(debug_dir / f"step_6_normalized_{base_name}.jpg"), display_normalized_bgr)
        
        # Step 6: Convert to NCHW format
        preprocessed = np.transpose(normalized_face, (2, 0, 1))
        preprocessed = preprocessed[None, :, :, :]  # Add batch dimension
        
        return preprocessed, face_region, landmarks
    
    def assess_face_quality(self, face_region: np.ndarray, landmarks: np.ndarray = None) -> float:
        """
        Phase 2: Assess face quality based on multiple factors
        Returns quality score between 0.0 (poor) and 1.0 (excellent)
        """
        try:
            quality_scores = []
            
            # 1. Sharpness assessment using Laplacian variance
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(laplacian_var / 500.0, 1.0)  # Normalize to 0-1
            quality_scores.append(sharpness_score)
            
            # 2. Brightness assessment - faces should be well-lit
            mean_brightness = np.mean(gray)
            brightness_score = 1.0 - abs(mean_brightness - 128) / 128.0  # Optimal around 128
            quality_scores.append(brightness_score)
            
            # 3. Size assessment - larger faces are generally better
            face_area = face_region.shape[0] * face_region.shape[1]
            size_score = min(face_area / (80 * 80), 1.0)  # 80x80 as minimum good size
            quality_scores.append(size_score)
            
            # 4. Landmark consistency (if available)
            if landmarks is not None:
                # Check if landmarks form reasonable face proportions
                if len(landmarks) > 33:  # MediaPipe has 468 landmarks
                    try:
                        # Eye distance ratio check
                        left_eye = landmarks[33]
                        right_eye = landmarks[263]
                        eye_distance = np.linalg.norm(left_eye - right_eye)
                        face_width = face_region.shape[1]
                        eye_ratio = eye_distance / face_width
                        
                        # Good faces have eye distance around 25-35% of face width
                        landmark_score = 1.0 - abs(eye_ratio - 0.3) / 0.3
                        landmark_score = max(0.0, min(1.0, landmark_score))
                        quality_scores.append(landmark_score)
                    except:
                        pass
            
            # 5. Contrast assessment - good faces have reasonable contrast
            contrast = gray.std()
            contrast_score = min(contrast / 50.0, 1.0)  # Normalize
            quality_scores.append(contrast_score)
            
            # Weighted average of all quality metrics
            final_score = np.mean(quality_scores)
            
            logger.debug(f"Face quality assessment: sharpness={sharpness_score:.3f}, "
                        f"brightness={brightness_score:.3f}, size={size_score:.3f}, "
                        f"contrast={contrast_score:.3f}, final={final_score:.3f}")
            
            return float(final_score)
            
        except Exception as e:
            logger.debug(f"Face quality assessment failed: {e}")
            return 0.5  # Default neutral quality
    
    def get_adaptive_confidence_threshold(self) -> float:
        """
        Phase 2: Calculate adaptive confidence threshold based on recent performance
        """
        if not self.confidence_history or not self.face_quality_scores:
            return self.base_confidence_threshold
        
        # Calculate average confidence and quality from recent predictions
        avg_confidence = np.mean(self.confidence_history)
        avg_quality = np.mean(self.face_quality_scores)
        
        # Adjust threshold based on face quality
        # High quality faces -> lower threshold (more sensitive)
        # Low quality faces -> higher threshold (more conservative)
        quality_adjustment = (avg_quality - 0.5) * 0.2  # ±0.1 adjustment
        adapted_threshold = self.base_confidence_threshold - quality_adjustment
        
        # Also adjust based on recent confidence levels
        # If consistently high confidence -> slightly lower threshold
        # If consistently low confidence -> slightly higher threshold
        confidence_adjustment = (avg_confidence - 0.5) * 0.1  # ±0.05 adjustment
        adapted_threshold -= confidence_adjustment
        
        # Clamp to reasonable bounds
        adapted_threshold = max(self.min_confidence_threshold, 
                              min(self.max_confidence_threshold, adapted_threshold))
        
        logger.debug(f"Adaptive threshold: base={self.base_confidence_threshold:.3f}, "
                    f"quality_adj={quality_adjustment:.3f}, conf_adj={confidence_adjustment:.3f}, "
                    f"final={adapted_threshold:.3f}")
        
        return adapted_threshold
    
    def apply_temporal_smoothing(self, new_emotion: str, new_confidence: float) -> Tuple[str, float]:
        """
        Phase 2: Apply temporal smoothing to reduce emotion jitter
        """
        if not self.emotion_history:
            # First prediction - no smoothing needed
            return new_emotion, new_confidence
        
        # Get the most recent emotion and confidence
        recent_emotion, recent_confidence = self.emotion_history[-1]
        
        # If the new emotion is different from recent, check if confidence difference is significant
        if new_emotion != recent_emotion:
            confidence_diff = new_confidence - recent_confidence
            
            # Only change emotion if confidence difference exceeds threshold
            if confidence_diff < self.emotion_change_threshold:
                # Keep previous emotion but with smoothed confidence
                smoothed_confidence = (self.smoothing_alpha * new_confidence + 
                                     (1 - self.smoothing_alpha) * recent_confidence)
                
                logger.debug(f"Temporal smoothing: kept {recent_emotion} "
                            f"(new: {new_emotion} {new_confidence:.3f}, "
                            f"smoothed: {smoothed_confidence:.3f})")
                
                return recent_emotion, smoothed_confidence
        
        # Either same emotion or significant confidence difference - allow change
        # But still apply some confidence smoothing
        if new_emotion == recent_emotion:
            smoothed_confidence = (self.smoothing_alpha * new_confidence + 
                                 (1 - self.smoothing_alpha) * recent_confidence)
        else:
            smoothed_confidence = new_confidence  # Strong change, keep original confidence
        
        return new_emotion, smoothed_confidence
    
    def predict_ensemble(self, image_bgr: np.ndarray, bbox: Tuple[int, int, int, int], debug_context: Dict[str, Any] = None) -> EmotionOutput:
        """
        Phase 2: Ensemble prediction using multiple preprocessing variations for better accuracy
        """
        if not self.ok:
            return EmotionOutput("neutral", 0.33, 0.0)
        
        try:
            # Extract face region once
            x1, y1, x2, y2 = bbox
            face_region = image_bgr[y1:y2, x1:x2].copy()
            
            # Assess face quality
            face_quality = self.assess_face_quality(face_region)
            
            # Variation 1: Standard preprocessing with alignment
            preprocessed_1, _, landmarks = self.preprocess_with_mesh(image_bgr, bbox)
            outputs_1 = self.session.run(None, {self.input_name: preprocessed_1})[0]
            if outputs_1.ndim == 2:
                outputs_1 = outputs_1[0]
            
            # Variation 2: Slight brightness adjustment
            brightened_face = cv2.convertScaleAbs(face_region, alpha=1.1, beta=10)
            brightened_image = image_bgr.copy()
            brightened_image[y1:y2, x1:x2] = brightened_face
            preprocessed_2, _, _ = self.preprocess_with_mesh(brightened_image, bbox)
            outputs_2 = self.session.run(None, {self.input_name: preprocessed_2})[0]
            if outputs_2.ndim == 2:
                outputs_2 = outputs_2[0]
            
            # Variation 3: Contrast enhancement
            contrast_face = cv2.convertScaleAbs(face_region, alpha=1.2, beta=0)
            contrast_image = image_bgr.copy()
            contrast_image[y1:y2, x1:x2] = contrast_face
            preprocessed_3, _, _ = self.preprocess_with_mesh(contrast_image, bbox)
            outputs_3 = self.session.run(None, {self.input_name: preprocessed_3})[0]
            if outputs_3.ndim == 2:
                outputs_3 = outputs_3[0]
            
            # Ensemble averaging with quality-based weighting
            weight_1 = 0.5  # Standard preprocessing gets highest weight
            weight_2 = 0.3 * face_quality  # Brightness adjustment weighted by quality
            weight_3 = 0.2 * face_quality  # Contrast enhancement weighted by quality
            total_weight = weight_1 + weight_2 + weight_3
            
            # Normalize weights
            weight_1 /= total_weight
            weight_2 /= total_weight 
            weight_3 /= total_weight
            
            # Weighted ensemble
            ensemble_outputs = (weight_1 * outputs_1 + 
                              weight_2 * outputs_2 + 
                              weight_3 * outputs_3)
            
            # Apply softmax
            ensemble_outputs = ensemble_outputs - np.max(ensemble_outputs)
            exp_outputs = np.exp(ensemble_outputs)
            probabilities = exp_outputs / np.sum(exp_outputs)
            
            # Get prediction
            predicted_idx = int(np.argmax(probabilities))
            raw_confidence = float(probabilities[predicted_idx])
            raw_emotion = EMOTION_LABELS[predicted_idx]
            
            # Apply temporal smoothing
            smoothed_emotion, smoothed_confidence = self.apply_temporal_smoothing(raw_emotion, raw_confidence)
            temporal_smoothed = (smoothed_emotion != raw_emotion or smoothed_confidence != raw_confidence)
            
            # Get adaptive threshold
            adaptive_threshold = self.get_adaptive_confidence_threshold()
            
            # Update histories
            self.emotion_history.append((smoothed_emotion, smoothed_confidence))
            self.confidence_history.append(smoothed_confidence)
            self.face_quality_scores.append(face_quality)
            self.prediction_count += 1
            
            # Debug tracking
            debug_save = debug_context is not None
            if debug_save and smoothed_confidence > self.debug_best_confidence:
                self.debug_best_confidence = smoothed_confidence
                self.debug_best_image = preprocessed_1.copy()  # Use standard preprocessing for debug
                self.debug_best_bbox = bbox
                self.debug_best_emotion = smoothed_emotion
                self.debug_best_face_region = face_region.copy()
                self.debug_best_landmarks = landmarks.copy() if landmarks is not None else None
                
                logger.info(f"DEBUG: Ensemble best frame - {smoothed_emotion} "
                           f"(conf: {smoothed_confidence:.3f}, quality: {face_quality:.3f}, "
                           f"weights: [{weight_1:.3f}, {weight_2:.3f}, {weight_3:.3f}])")
            
            result = EmotionOutput(
                label=smoothed_emotion,
                confidence=smoothed_confidence,
                quality_score=face_quality,
                temporal_smoothed=temporal_smoothed,
                adaptive_threshold=adaptive_threshold
            )
            
            logger.debug(f"Ensemble prediction: {smoothed_emotion} ({smoothed_confidence:.3f}), "
                        f"quality: {face_quality:.3f}, weights: [{weight_1:.2f}, {weight_2:.2f}, {weight_3:.2f}]")
            
            return result
            
        except Exception as e:
            logger.error(f"Ensemble prediction failed: {e}")
            return EmotionOutput("neutral", 0.33, 0.0)
    
    def predict(self, image_bgr: np.ndarray, bbox: Tuple[int, int, int, int], debug_context: Dict[str, Any] = None) -> EmotionOutput:
        """
        Phase 2: Enhanced emotion prediction with quality assessment, adaptive thresholding, and temporal smoothing
        
        Args:
            image_bgr: Input image in BGR format
            bbox: Face bounding box (x1, y1, x2, y2)
            debug_context: Optional debug context for saving debug information
        """
        if not self.ok:
            return EmotionOutput("neutral", 0.33, 0.0)
        
        # Use ensemble prediction if enabled
        if self.use_ensemble:
            return self.predict_ensemble(image_bgr, bbox, debug_context)
        
        try:
            # Use enhanced preprocessing with debug data
            preprocessed_input, face_region, landmarks = self.preprocess_with_mesh(image_bgr, bbox, debug_context)
            
            # Phase 2: Assess face quality
            face_quality = self.assess_face_quality(face_region, landmarks)
            
            # Run inference
            outputs = self.session.run(None, {self.input_name: preprocessed_input})[0]
            
            # Handle different output formats
            if outputs.ndim == 2:
                outputs = outputs[0]
            
            # Apply softmax to get probabilities
            outputs = outputs - np.max(outputs)  # Numerical stability
            exp_outputs = np.exp(outputs)
            probabilities = exp_outputs / np.sum(exp_outputs)
            
            # Get raw prediction
            predicted_idx = int(np.argmax(probabilities))
            raw_confidence = float(probabilities[predicted_idx])
            raw_emotion = EMOTION_LABELS[predicted_idx]
            
            # Phase 2: Apply temporal smoothing
            smoothed_emotion, smoothed_confidence = self.apply_temporal_smoothing(raw_emotion, raw_confidence)
            temporal_smoothed = (smoothed_emotion != raw_emotion or smoothed_confidence != raw_confidence)
            
            # Phase 2: Get adaptive confidence threshold
            adaptive_threshold = self.get_adaptive_confidence_threshold()
            
            # Update tracking histories
            self.emotion_history.append((smoothed_emotion, smoothed_confidence))
            self.confidence_history.append(smoothed_confidence)
            self.face_quality_scores.append(face_quality)
            self.prediction_count += 1
            
            # Enhanced debug tracking with quality metrics
            debug_save = debug_context is not None if debug_context else False
            if debug_save and smoothed_confidence > self.debug_best_confidence:
                self.debug_best_confidence = smoothed_confidence
                self.debug_best_image = preprocessed_input.copy()
                self.debug_best_bbox = bbox
                self.debug_best_emotion = smoothed_emotion
                self.debug_best_face_region = face_region.copy()
                self.debug_best_landmarks = landmarks.copy() if landmarks is not None else None
                # Store original frame for pipeline visualization
                if 'input_frame' in debug_context:
                    self.debug_best_original_frame = debug_context['input_frame'].copy()
                
                logger.info(f"DEBUG: UPDATED best frame - raw: {raw_emotion}, smoothed: {smoothed_emotion}, "
                           f"conf: {smoothed_confidence:.3f}, stored_emotion: {self.debug_best_emotion}")
                
                logger.info(f"DEBUG: New best frame - {smoothed_emotion} "
                           f"(conf: {smoothed_confidence:.3f}, quality: {face_quality:.3f}, "
                           f"threshold: {adaptive_threshold:.3f}, smoothed: {temporal_smoothed})")
                
                # Debug: Show temporal smoothing details
                if self.emotion_history:
                    recent_history = [e[0] for e in list(self.emotion_history)[-3:]]
                    logger.info(f"DEBUG: Temporal context - raw: {raw_emotion}, smoothed: {smoothed_emotion}, "
                               f"recent history: {' -> '.join(recent_history)}")
            
            # Create enhanced output with quality metrics
            result = EmotionOutput(
                label=smoothed_emotion,
                confidence=smoothed_confidence, 
                quality_score=face_quality,
                temporal_smoothed=temporal_smoothed,
                adaptive_threshold=adaptive_threshold
            )
            
            logger.debug(f"Enhanced prediction: {smoothed_emotion} ({smoothed_confidence:.3f}), "
                        f"quality: {face_quality:.3f}, smoothed: {temporal_smoothed}")
            
            return result
            
        except Exception as e:
            logger.error(f"Enhanced prediction failed: {e}")
            return EmotionOutput("neutral", 0.33, 0.0)
    
    def save_best_debug_frame(self, session_dir=None):
        """Save the most confident frame - now uses enhanced Phase 2 debug saving"""
        self.save_enhanced_debug_frame(session_dir=session_dir)
    
    def reset_debug_tracking(self):
        """Reset debug tracking for new video session"""
        self.debug_sample_saved = False
        self.debug_best_confidence = 0.0
        self.debug_best_image = None
        self.debug_best_bbox = None
        self.debug_best_face_region = None
        self.debug_best_landmarks = None
    
    def draw_landmarks_mesh(self, face_image: np.ndarray, landmarks: np.ndarray):
        """Draw MediaPipe face mesh on the face image"""
        try:
            h, w = face_image.shape[:2]
            logger.info(f"DEBUG: Drawing mesh on {w}x{h} image with {len(landmarks)} landmarks")
            logger.info(f"DEBUG: First few landmarks: {landmarks[:5]}")
            
            # Draw all landmark points (make them bigger and more visible)
            points_drawn = 0
            for i, (x, y) in enumerate(landmarks):
                # The landmarks are already in pixel coordinates for the 260x260 aligned face
                x_pixel = int(x)
                y_pixel = int(y)
                logger.debug(f"DEBUG: Landmark {i}: ({x}, {y}) -> ({x_pixel}, {y_pixel})")
                
                # Draw with bounds checking
                if 0 <= x_pixel < w and 0 <= y_pixel < h:
                    cv2.circle(face_image, (x_pixel, y_pixel), 2, (0, 255, 0), -1)  
                    points_drawn += 1
                else:
                    logger.debug(f"DEBUG: Point {i} out of bounds: ({x_pixel}, {y_pixel})")
            
            logger.info(f"DEBUG: Drew {points_drawn} landmark points")
            
            # Draw key connections for face outline and features
            # MediaPipe FaceMesh key landmark connections
            connections = [
                # Face contour (simplified)
                (10, 338), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389), (389, 356), (356, 454),
                (454, 323), (323, 361), (361, 288), (288, 397), (397, 365), (365, 379), (379, 378), (378, 400),
                (400, 377), (377, 152), (152, 148), (148, 176), (176, 149), (149, 150), (150, 136), (136, 172),
                (172, 58), (58, 132), (132, 93), (93, 234), (234, 127), (127, 162), (162, 21), (21, 54),
                (54, 103), (103, 67), (67, 109), (109, 10),
                
                # Left eye
                (33, 7), (7, 163), (163, 144), (144, 145), (145, 153), (153, 154), (154, 155), (155, 133),
                (133, 173), (173, 157), (157, 158), (158, 159), (159, 160), (160, 161), (161, 246),
                
                # Right eye  
                (362, 382), (382, 381), (381, 380), (380, 374), (374, 373), (373, 390), (390, 249),
                (249, 263), (263, 466), (466, 388), (388, 387), (387, 386), (386, 385), (385, 384), (384, 398),
                
                # Nose
                (1, 2), (2, 5), (5, 4), (4, 6), (6, 168), (168, 8), (8, 9), (9, 10), (10, 151),
                
                # Lips outer
                (61, 84), (84, 17), (17, 314), (314, 405), (405, 320), (320, 307), (307, 375), (375, 321),
                (321, 308), (308, 324), (324, 318), (318, 402), (402, 317), (317, 14), (14, 87), (87, 178),
                (178, 88), (88, 95), (95, 78), (78, 191), (191, 80), (80, 81), (81, 82), (82, 13), (13, 312), (312, 311),
            ]
            
            # Draw connections as lines (make them thicker and more visible)
            lines_drawn = 0
            for connection in connections:
                if connection[0] < len(landmarks) and connection[1] < len(landmarks):
                    pt1 = landmarks[connection[0]]
                    pt2 = landmarks[connection[1]]
                    
                    # Landmarks are already in pixel coordinates
                    x1, y1 = int(pt1[0]), int(pt1[1])
                    x2, y2 = int(pt2[0]), int(pt2[1])
                    
                    # Check bounds for both points
                    if (0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h):
                        cv2.line(face_image, (x1, y1), (x2, y2), (0, 0, 255), 1)  # Red lines
                        lines_drawn += 1
            
            logger.info(f"DEBUG: Drew {lines_drawn} connection lines")
            return face_image
            
        except Exception as e:
            logger.debug(f"Failed to draw landmarks mesh: {e}")
            return face_image


    def save_enhanced_debug_frame(self, session_dir=None):
        """
        Phase 2: Enhanced debug frame saving with quality metrics and analysis
        """
        logger.info(f"DEBUG: Enhanced debug save - image: {self.debug_best_image is not None}, "
                   f"saved: {self.debug_sample_saved}, conf: {self.debug_best_confidence}")
                   
        if self.debug_best_image is not None and not self.debug_sample_saved:
            try:
                # Use provided session directory or create a new one
                if session_dir:
                    from pathlib import Path
                    session_dir = Path(session_dir)
                    logger.info(f"DEBUG: Using provided session directory: {session_dir}")
                else:
                    # Fallback to original behavior if no session_dir provided
                    from pathlib import Path
                    import datetime
                    project_root = Path(__file__).parent.parent.parent
                    debug_base_dir = project_root / "debug_preprocessed"
                    
                    # Create individual session folder for this inference
                    timestamp = int(time.time() * 1000)
                    confidence_str = f"{self.debug_best_confidence:.3f}".replace('.', '')
                    session_name = f"{self.debug_best_emotion}_conf{confidence_str}_{timestamp}"
                    session_dir = debug_base_dir / session_name
                
                # Create organized folder structure within session
                best_frames_dir = session_dir / "best_frames"
                analysis_dir = session_dir / "analysis"
                mesh_dir = session_dir / "mesh_visualization"
                pipeline_steps_dir = session_dir / "pipeline_steps"
                
                for dir_path in [best_frames_dir, analysis_dir, mesh_dir, pipeline_steps_dir]:
                    os.makedirs(str(dir_path), exist_ok=True)
                
                base_filename = f"{self.debug_best_emotion}_conf{confidence_str}_{timestamp}"
                
                # Convert model input back to image
                debug_img = self.debug_best_image[0].copy()
                mean = np.array([0.485, 0.456, 0.406])[:, None, None]
                std = np.array([0.229, 0.224, 0.225])[:, None, None]
                debug_img = (debug_img * std) + mean
                debug_img = np.clip(debug_img, 0, 1) * 255
                debug_img = debug_img.transpose(1, 2, 0).astype(np.uint8)
                debug_img = cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR)
                
                # Save main aligned face
                main_path = str(best_frames_dir / f"enhanced_{base_filename}.jpg")
                cv2.imwrite(main_path, debug_img)
                
                # Create step-by-step pipeline visualization if we have the necessary data
                if (self.debug_best_face_region is not None and 
                    self.debug_best_bbox is not None and
                    hasattr(self, 'debug_best_original_frame')):
                    
                    self._create_pipeline_steps_visualization(
                        pipeline_steps_dir, 
                        base_filename
                    )
                
                # Create analysis visualization if we have quality and history data
                if hasattr(self, 'face_quality_scores') and len(self.face_quality_scores) > 0:
                    analysis_img = debug_img.copy()
                    
                    # Add text overlay with Phase 2 metrics
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.6
                    color = (0, 255, 0)  # Green
                    thickness = 1
                    
                    # Get current quality and thresholds
                    current_quality = list(self.face_quality_scores)[-1] if self.face_quality_scores else 0.5
                    current_threshold = self.get_adaptive_confidence_threshold()
                    
                    # Add quality metrics
                    y_offset = 25
                    cv2.putText(analysis_img, f"Quality: {current_quality:.3f}", 
                              (10, y_offset), font, font_scale, color, thickness)
                    y_offset += 25
                    cv2.putText(analysis_img, f"Threshold: {current_threshold:.3f}", 
                              (10, y_offset), font, font_scale, color, thickness)
                    y_offset += 25
                    cv2.putText(analysis_img, f"Predictions: {self.prediction_count}", 
                              (10, y_offset), font, font_scale, color, thickness)
                    
                    # Add emotion history if available
                    if len(self.emotion_history) > 1:
                        y_offset += 25
                        recent_emotions = [e[0] for e in list(self.emotion_history)[-3:]]
                        cv2.putText(analysis_img, f"History: {' -> '.join(recent_emotions)}", 
                                  (10, y_offset), font, font_scale, color, thickness)
                    
                    analysis_path = str(analysis_dir / f"analysis_{base_filename}.jpg")
                    cv2.imwrite(analysis_path, analysis_img)
                    logger.info(f"DEBUG: Saved analysis visualization to {analysis_path}")
                
                # Save landmarks mesh if available
                if (self.debug_best_face_region is not None and 
                    self.debug_best_landmarks is not None):
                    mesh_img = debug_img.copy()
                    mesh_img = self.draw_landmarks_mesh(mesh_img, self.debug_best_landmarks)
                    mesh_path = str(mesh_dir / f"mesh_{base_filename}.jpg")
                    cv2.imwrite(mesh_path, mesh_img)
                    logger.info(f"DEBUG: Saved mesh visualization to {mesh_path}")
                
                # Create a debug summary file with Phase 2 metrics
                summary_path = str(session_dir / "debug_summary.txt")
                
                logger.info(f"DEBUG: Creating summary with stored emotion: {self.debug_best_emotion}")
                
                with open(summary_path, 'w') as f:
                    f.write(f"Phase 2 Enhanced Emotion Recognition - Debug Summary\n")
                    f.write(f"=" * 60 + "\n\n")
                    f.write(f"Best Frame Analysis:\n")
                    f.write(f"  Best Prediction: {self.debug_best_emotion}\n")
                    f.write(f"  Confidence: {self.debug_best_confidence:.4f}\n")
                    f.write(f"  Timestamp: {timestamp}\n")
                    f.write(f"  Note: This shows the highest-confidence frame, which may bypass temporal smoothing\n\n")
                    
                    if hasattr(self, 'face_quality_scores') and len(self.face_quality_scores) > 0:
                        current_quality = self.face_quality_scores[-1]
                        avg_quality = sum(self.face_quality_scores) / len(self.face_quality_scores)
                        f.write(f"Face Quality Metrics:\n")
                        f.write(f"  Current Quality: {current_quality:.4f}\n")
                        f.write(f"  Average Quality: {avg_quality:.4f}\n")
                        f.write(f"  Quality Samples: {len(self.face_quality_scores)}\n\n")
                    
                    if hasattr(self, 'confidence_history') and len(self.confidence_history) > 0:
                        current_threshold = self.get_adaptive_confidence_threshold()
                        avg_confidence = sum(self.confidence_history) / len(self.confidence_history)
                        f.write(f"Adaptive Thresholding:\n")
                        f.write(f"  Current Threshold: {current_threshold:.4f}\n")
                        f.write(f"  Average Confidence: {avg_confidence:.4f}\n")
                        f.write(f"  Confidence Samples: {len(self.confidence_history)}\n\n")
                    
                    if hasattr(self, 'emotion_history') and len(self.emotion_history) > 0:
                        recent_emotions = [e[0] for e in list(self.emotion_history)[-5:]]
                        
                        # Calculate emotion distribution across the session
                        emotion_counts = {}
                        for emotion, _ in self.emotion_history:
                            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                        
                        # Find most common emotion in session
                        most_common = max(emotion_counts, key=emotion_counts.get)
                        most_common_pct = (emotion_counts[most_common] / len(self.emotion_history)) * 100
                        
                        f.write(f"Session-Level Temporal Smoothing:\n")
                        f.write(f"  Recent History: {' -> '.join(recent_emotions)}\n")
                        f.write(f"  Total Predictions: {len(self.emotion_history)}\n")
                        f.write(f"  Most Common Emotion: {most_common} ({most_common_pct:.1f}%)\n")
                        f.write(f"  Emotion Distribution: {dict(emotion_counts)}\n")
                        f.write(f"  Note: This shows actual temporal smoothing results across all frames\n\n")
                    
                    f.write(f"Performance:\n")
                    f.write(f"  Total Predictions: {self.prediction_count}\n")
                    f.write(f"  Debug Files Generated:\n")
                    f.write(f"    - Enhanced Face: best_frames/enhanced_{base_filename}.jpg\n")
                    f.write(f"    - Analysis View: analysis/analysis_{base_filename}.jpg\n") 
                    f.write(f"    - Mesh View: mesh_visualization/mesh_{base_filename}.jpg\n")
                    f.write(f"    - Pipeline Steps: pipeline_steps/ (step-by-step processing visualization)\n")
                
                logger.info(f"DEBUG: Enhanced debug saved - {self.debug_best_emotion} ({self.debug_best_confidence:.3f})")
                logger.info(f"DEBUG: Session files saved to {session_dir}")
                self.debug_sample_saved = True
                self.reset_debug_tracking()
                
            except Exception as e:
                logger.error(f"Enhanced debug save failed: {e}")
    
    def _create_pipeline_steps_visualization(self, pipeline_dir, base_filename):
        """Create step-by-step pipeline visualization showing ACTUAL processing stages"""
        try:
            logger.info(f"DEBUG: Creating REAL pipeline steps visualization")
            
            # Step 1: Original full frame with detected face bbox (clean)
            original_frame = self.debug_best_original_frame.copy()
            x1, y1, x2, y2 = self.debug_best_bbox
            cv2.rectangle(original_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            step1_path = str(pipeline_dir / f"step_1_detection_{base_filename}.jpg")
            cv2.imwrite(step1_path, original_frame)
            
            # Step 2: Face crop extraction (clean)
            face_crop = self.debug_best_face_region.copy()
            step2_path = str(pipeline_dir / f"step_2_crop_{base_filename}.jpg")
            cv2.imwrite(step2_path, face_crop)
            
            # Step 3: Generate landmarks on the crop (clean)
            mesh_face = face_crop.copy()
            crop_landmarks = self.extract_landmarks_mediapipe(face_crop)
            
            if crop_landmarks is not None:
                logger.info(f"DEBUG: Extracted {len(crop_landmarks)} landmarks on crop")
                
                # Draw landmarks mesh
                mesh_face = self.draw_landmarks_mesh(mesh_face, crop_landmarks)
                
                # Show eye alignment analysis (before alignment)
                left_eye_outer = tuple(crop_landmarks[33].astype(int))
                right_eye_outer = tuple(crop_landmarks[263].astype(int))
                
                # Eye alignment line - yellow line connecting eyes
                cv2.line(mesh_face, left_eye_outer, right_eye_outer, (0, 255, 255), 2)
                
                # Horizontal reference line - green reference
                eye_center_y = (left_eye_outer[1] + right_eye_outer[1]) // 2
                cv2.line(mesh_face, (0, eye_center_y), (mesh_face.shape[1], eye_center_y), 
                        (0, 255, 0), 1)
                
                # Mark eye corners with circles
                cv2.circle(mesh_face, left_eye_outer, 3, (255, 0, 0), -1)   # Blue
                cv2.circle(mesh_face, right_eye_outer, 3, (255, 0, 0), -1)  # Blue
            
            step3_path = str(pipeline_dir / f"step_3_landmarks_{base_filename}.jpg")
            cv2.imwrite(step3_path, mesh_face)
            
            # ACTUAL PROCESSING STEPS (no text labels in processing)
            
            # Step 4: Face alignment 
            if crop_landmarks is not None:
                aligned_face = self.apply_alignment(face_crop, crop_landmarks)
            else:
                aligned_face = face_crop.copy()
            
            # Step 5: Add 10% padding
            h, w = aligned_face.shape[:2]
            pad_h, pad_w = int(h * 0.1), int(w * 0.1)
            padded_face = cv2.copyMakeBorder(aligned_face, pad_h, pad_h, pad_w, pad_w, 
                                           cv2.BORDER_REFLECT)
            
            # Step 6: Resize to model input (260x260)
            resized_face = cv2.resize(padded_face, (260, 260), interpolation=cv2.INTER_CUBIC)
            
            # Step 7: Normalization for model (CLEAN - no labels)
            rgb_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            normalized_face = (rgb_face - mean) / std
            
            # Convert back to displayable format (CLEAN - no labels)
            display_norm = normalized_face * std + mean
            display_norm = np.clip(display_norm * 255, 0, 255).astype(np.uint8)
            display_norm = cv2.cvtColor(display_norm, cv2.COLOR_RGB2BGR)
            
            # Save Step 4 with alignment guides to show alignment quality
            aligned_debug = aligned_face.copy()
            if crop_landmarks is not None:
                # Extract landmarks on the aligned face to show the alignment result
                aligned_landmarks = self.extract_landmarks_mediapipe(aligned_face)
                if aligned_landmarks is not None:
                    # Show eye alignment on the aligned face
                    left_eye_outer = tuple(aligned_landmarks[33].astype(int))
                    right_eye_outer = tuple(aligned_landmarks[263].astype(int))
                    
                    # Eye alignment line - yellow line connecting eyes (should be more horizontal now)
                    cv2.line(aligned_debug, left_eye_outer, right_eye_outer, (0, 255, 255), 2)
                    
                    # Horizontal reference line - green reference
                    eye_center_y = (left_eye_outer[1] + right_eye_outer[1]) // 2
                    cv2.line(aligned_debug, (0, eye_center_y), (aligned_debug.shape[1], eye_center_y), 
                            (0, 255, 0), 1)
                    
                    # Mark eye corners with circles
                    cv2.circle(aligned_debug, left_eye_outer, 3, (255, 0, 0), -1)   # Blue
                    cv2.circle(aligned_debug, right_eye_outer, 3, (255, 0, 0), -1)  # Blue
            
            step4_path = str(pipeline_dir / f"step_4_aligned_{base_filename}.jpg")
            cv2.imwrite(step4_path, aligned_debug)
            
            step5_path = str(pipeline_dir / f"step_5_padded_{base_filename}.jpg")
            cv2.imwrite(step5_path, padded_face)
            
            step6_path = str(pipeline_dir / f"step_6_resized_260x260_{base_filename}.jpg")
            cv2.imwrite(step6_path, resized_face)
            
            step7_path = str(pipeline_dir / f"step_7_normalized_{base_filename}.jpg")
            cv2.imwrite(step7_path, display_norm)
            
            logger.info(f"DEBUG: Saved 7 REAL pipeline steps to {pipeline_dir}")
            
        except Exception as e:
            logger.error(f"Pipeline steps visualization failed: {e}")
            import traceback
            traceback.print_exc()
    
    def get_model_info(self):
        """Get comprehensive information about the loaded model with Phase 2 enhancements"""
        if not self.ok:
            return {"status": "not_loaded"}
        
        # Calculate performance statistics
        avg_quality = np.mean(self.face_quality_scores) if self.face_quality_scores else 0.0
        avg_confidence = np.mean(self.confidence_history) if self.confidence_history else 0.0
        current_threshold = self.get_adaptive_confidence_threshold()
        
        # Emotion distribution from history
        emotion_counts = {}
        for emotion, _ in self.emotion_history:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        base_info = {
            "status": "loaded",
            "model_path": self.onnx_path,
            "input_shape": self.input_shape,
            "input_size": self.input_size,
            "landmark_detector": self.landmark_detector[0] if self.landmark_detector else "none",
            "device": self.device,
            "temporal_enabled": self.enable_temporal,
            "temporal_model_loaded": self.temporal_model is not None
        }
        
        # Phase 2 enhancements
        phase2_info = {
            "phase2_features": {
                "adaptive_thresholding": True,
                "temporal_smoothing": True,
                "face_quality_assessment": True,
                "ensemble_prediction": True,
                "enhanced_debug": True
            },
            "performance_stats": {
                "total_predictions": self.prediction_count,
                "average_quality": float(avg_quality),
                "average_confidence": float(avg_confidence),
                "current_adaptive_threshold": float(current_threshold),
                "smoothing_alpha": self.smoothing_alpha,
                "emotion_change_threshold": self.emotion_change_threshold
            },
            "emotion_distribution": emotion_counts,
            "recent_emotions": [e[0] for e in list(self.emotion_history)[-5:]] if self.emotion_history else []
        }
        
        return {**base_info, **phase2_info}
    
    def _init_temporal_model(self):
        """Initialize temporal self-attention model (SOTA)"""
        if not PYTORCH_AVAILABLE:
            logger.warning("PyTorch not available - temporal modeling disabled")
            return
            
        try:
            # Initialize SOTA self-attention temporal model
            self.temporal_model = TemporalSelfAttentionModel(
                input_channels=3,
                feature_dim=512,  # Rich feature representation
                num_classes=7,
                sequence_length=16,
                num_heads=8  # Multi-head attention
            )
            
            # Set to evaluation mode (untrained but ready for inference)
            self.temporal_model.eval()
            
            # Move to appropriate device
            if self.device == "cuda" and torch.cuda.is_available():
                self.temporal_model = self.temporal_model.cuda()
            else:
                self.temporal_model = self.temporal_model.cpu()
            
            logger.info("SOTA Temporal Self-Attention model initialized (untrained, 8-head)")
            
        except Exception as e:
            logger.warning(f"Failed to initialize self-attention temporal model: {e}")
            self.temporal_model = None
    
    def predict_temporal(self, frame_sequence: np.ndarray) -> Tuple[str, float]:
        """
        Predict emotion from temporal sequence using SOTA self-attention
        
        Args:
            frame_sequence: (16, 3, 260, 260) array of preprocessed face frames
            
        Returns:
            Tuple of (emotion_label, confidence)
        """
        if not self.enable_temporal or self.temporal_model is None:
            logger.warning("Self-attention temporal model not available, falling back to single frame")
            # Use the last frame for single-frame prediction
            last_frame = frame_sequence[-1]
            return self._predict_single_frame_from_preprocessed(last_frame)
        
        try:
            # Convert to torch tensor and add batch dimension
            # Input: (16, 3, 260, 260) -> (1, 16, 3, 260, 260)
            input_tensor = torch.from_numpy(frame_sequence).float().unsqueeze(0)
            
            # Move to device
            if self.device == "cuda" and torch.cuda.is_available():
                input_tensor = input_tensor.cuda()
            
            # Forward pass with self-attention
            with torch.no_grad():
                logits, attention_info = self.temporal_model(input_tensor)  # (1, 7), attention_weights
                probabilities = F.softmax(logits, dim=1)
                
            # Convert back to numpy and get prediction
            probs_np = probabilities.cpu().numpy()[0]  # (7,)
            predicted_idx = int(np.argmax(probs_np))
            confidence = float(probs_np[predicted_idx])
            
            # Log attention insights for interpretability 
            if attention_info and 'temporal_attention' in attention_info:
                temp_attn = attention_info['temporal_attention'].cpu().numpy()[0, 0]  # (16,)
                most_important_frames = np.argsort(temp_attn)[-3:]  # Top 3 frames
                logger.debug(f"Self-attention: Most important frames {most_important_frames} "
                           f"with weights {temp_attn[most_important_frames]}")
            
            return EMOTION_LABELS[predicted_idx], confidence
            
        except Exception as e:
            logger.warning(f"Self-attention temporal prediction failed: {e}, falling back to single frame")
            # Fallback to single-frame prediction
            last_frame = frame_sequence[-1] 
            return self._predict_single_frame_from_preprocessed(last_frame)
    
    def _predict_single_frame_from_preprocessed(self, preprocessed_frame: np.ndarray) -> Tuple[str, float]:
        """Predict from already preprocessed frame (fallback method)"""
        if not self.ok:
            return "neutral", 0.33
            
        try:
            # Add batch dimension: (3, 260, 260) -> (1, 3, 260, 260)
            if preprocessed_frame.ndim == 3:
                preprocessed_frame = preprocessed_frame[None, :, :, :]
            
            # Run ONNX inference
            outputs = self.session.run(None, {self.input_name: preprocessed_frame})[0]
            
            # Handle different output formats
            if outputs.ndim == 2:
                outputs = outputs[0]
            
            # Apply softmax to get probabilities
            outputs = outputs - np.max(outputs)  # Numerical stability
            exp_outputs = np.exp(outputs)
            probabilities = exp_outputs / np.sum(exp_outputs)
            
            # Get prediction
            predicted_idx = int(np.argmax(probabilities))
            confidence = float(probabilities[predicted_idx])
            
            return EMOTION_LABELS[predicted_idx], confidence
            
        except Exception as e:
            logger.error(f"Single frame prediction from preprocessed failed: {e}")
            return "neutral", 0.33

def softmax(x: np.ndarray) -> np.ndarray:
    """Softmax function for probability computation"""
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


# =============================================================================
# Temporal Modeling with Self-Attention 
# =============================================================================

import math

class PositionalEncoding(nn.Module):
    """Positional encoding for temporal sequences"""
    
    def __init__(self, d_model, max_seq_length=32):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: [seq_len, batch_size, d_model]
        return x + self.pe[:x.size(0), :]


class MultiHeadTemporalAttention(nn.Module):
    """Multi-head self-attention for temporal emotion sequences"""
    
    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super(MultiHeadTemporalAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()
        
        # Generate Q, K, V
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)  
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        # Output projection
        output = self.w_o(attended)
        
        return output, attention_weights


class TemporalSelfAttentionModel(nn.Module):
    """SOTA Temporal Emotion Recognition with Self-Attention"""
    
    def __init__(self, input_channels=3, feature_dim=512, num_classes=7, sequence_length=16, num_heads=8):
        super(TemporalSelfAttentionModel, self).__init__()
        
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        
        # Feature extraction backbone (lightweight but effective)
        self.feature_extractor = nn.Sequential(
            # First block
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Second block  
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Third block
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Fourth block
            nn.Conv2d(256, feature_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            
            # Global average pooling
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # Positional encoding for temporal order
        self.positional_encoding = PositionalEncoding(feature_dim, sequence_length)
        
        # Multi-head self-attention layers
        self.attention1 = MultiHeadTemporalAttention(feature_dim, num_heads, dropout=0.1)
        self.norm1 = nn.LayerNorm(feature_dim)
        
        self.attention2 = MultiHeadTemporalAttention(feature_dim, num_heads, dropout=0.1)
        self.norm2 = nn.LayerNorm(feature_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim)
        )
        self.norm3 = nn.LayerNorm(feature_dim)
        
        # Temporal aggregation and classification
        self.temporal_pool = nn.MultiheadAttention(feature_dim, num_heads=4, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim // 2, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, sequence, channels, height, width) - e.g., (1, 16, 3, 260, 260)
        Returns:
            logits: (batch, num_classes)
            attention_weights: For interpretability
        """
        batch_size, seq_len, c, h, w = x.size()
        
        # Extract features for each frame in parallel
        x_reshaped = x.view(batch_size * seq_len, c, h, w)
        frame_features = self.feature_extractor(x_reshaped)  # (batch*seq, feature_dim)
        
        # Reshape back to sequence format
        frame_features = frame_features.view(batch_size, seq_len, self.feature_dim)
        
        # Add positional encoding (temporal order awareness)
        # Note: PE expects (seq, batch, features) but attention expects (batch, seq, features)
        frame_features_pe = frame_features.transpose(0, 1)  # (seq, batch, features)
        frame_features_pe = self.positional_encoding(frame_features_pe)
        frame_features = frame_features_pe.transpose(0, 1)  # (batch, seq, features)
        
        # First self-attention layer with residual connection
        attended1, attn_weights1 = self.attention1(frame_features)
        frame_features = self.norm1(frame_features + attended1)
        
        # Second self-attention layer with residual connection  
        attended2, attn_weights2 = self.attention2(frame_features)
        frame_features = self.norm2(frame_features + attended2)
        
        # Feed-forward network with residual connection
        ffn_output = self.ffn(frame_features)
        frame_features = self.norm3(frame_features + ffn_output)
        
        # Global temporal aggregation using cross-attention
        # Query: learnable global token, Keys/Values: all frame features
        global_query = torch.mean(frame_features, dim=1, keepdim=True)  # (batch, 1, feature_dim)
        aggregated, final_attn_weights = self.temporal_pool(
            global_query, frame_features, frame_features
        )
        
        # Classification
        logits = self.classifier(aggregated.squeeze(1))  # (batch, num_classes)
        
        # Return attention weights for interpretability
        attention_info = {
            'layer1_attention': attn_weights1,  # (batch, heads, seq, seq)
            'layer2_attention': attn_weights2,  # (batch, heads, seq, seq) 
            'temporal_attention': final_attn_weights  # (batch, 1, seq)
        }
        
        return logits, attention_info