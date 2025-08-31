#!/usr/bin/env python3
"""
POSTER V2 Emotion Recognition Model Wrapper
Integrates POSTER V2 with the existing emotion recognition pipeline
"""

import sys
import os
import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import logging
from torchvision import transforms


# Add POSTER_V2 to path - Fixed namespace collision by adding parent directory
THIS_FILE = Path(__file__).resolve()
# poster_v2_model.py -> models -> ml_service -> apps -> refactor  
REF_ROOT = THIS_FILE.parents[3]
THIRD_PARTY = REF_ROOT / "third_party"

# Make POSTER_V2 a proper top-level package (add parent, not POSTER_V2 itself)
sys.path.insert(0, str(THIRD_PARTY))

# Now import is unambiguous - no collision with local 'models' package
try:
    from POSTER_V2.models.PosterV2_7cls import pyramid_trans_expr2
except Exception as e:
    raise ImportError(f"Failed to import POSTER_V2 from {THIRD_PARTY}: {e}")

logger = logging.getLogger(__name__)

# Emotion labels for 7-class classification (RAF-DB format)
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

class EmotionOutput:
    """Output structure for emotion predictions"""
    def __init__(self, label: str, confidence: float, quality_score: float = 1.0, temporal_smoothed: bool = False):
        self.label = label
        self.confidence = confidence
        self.quality_score = quality_score
        self.temporal_smoothed = temporal_smoothed

class PosterV2EmotionModel:
    """
    POSTER V2 Emotion Recognition Model
    Drop-in replacement for EnhancedEmotionModel with better accuracy
    """
    
    def __init__(self, device: str = "cpu", enable_temporal: bool = False):
        self.device = device
        self.enable_temporal = enable_temporal
        self.ok = False
        self.model = None
        
        # Preprocessing parameters
        self.input_size = 224  # POSTER V2 uses 224x224 input
        
        # Emotion history for temporal smoothing
        self.emotion_history = []
        self.max_history = 5
        
        # Initialize model
        self._init_model()
        
    def _init_model(self):
        """Initialize POSTER V2 model"""
        try:
            logger.info("Initializing POSTER V2 emotion model...")
            
            # Create model
            self.model = pyramid_trans_expr2(img_size=224, num_classes=7)
            self.model.to(self.device)
            
            # Load trained POSTER V2 weights
            self._load_trained_weights()
            
            self.model.eval()
            
            # Image preprocessing pipeline
            self.preprocess = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            self.ok = True
            logger.info(f"✓ POSTER V2 model initialized successfully on {self.device}")
            logger.info(f"✓ Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            
        except Exception as e:
            logger.error(f"Failed to initialize POSTER V2 model: {e}")
            self.ok = False
    
    def _load_trained_weights(self):
        """Load trained POSTER V2 emotion recognition weights"""
        try:
            # Path to clean trained model weights using consistent REF_ROOT
            checkpoint_path = REF_ROOT / "third_party" / "POSTER_V2" / "models" / "pretrain" / "poster_v2_clean_weights.pth"
            
            if not checkpoint_path.exists():
                logger.warning(f"Clean POSTER V2 weights not found at {checkpoint_path}")
                logger.warning("Using untrained model - predictions may not be accurate")
                logger.warning("Run extract_poster_weights.py to create clean weights from research checkpoint")
                return
                
            logger.info(f"Loading clean POSTER V2 weights from {checkpoint_path}")
            
            # Load clean weights (no RecorderMeter classes needed)
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            
            # Load weights into model
            self.model.load_state_dict(state_dict, strict=False)
            
            logger.info("✓ Successfully loaded trained POSTER V2 emotion weights")
            
        except Exception as e:
            logger.error(f"Failed to load trained POSTER V2 weights: {e}")
            logger.warning("Using untrained model - predictions may not be accurate")
    
    def predict(self, image_bgr: np.ndarray, bbox: Tuple[int, int, int, int], 
                debug_context: Dict[str, Any] = None) -> EmotionOutput:
        """
        Predict emotion using POSTER V2
        
        Args:
            image_bgr: Input image in BGR format  
            bbox: Face bounding box (x1, y1, x2, y2)
            debug_context: Optional debug context for saving debug information
            
        Returns:
            EmotionOutput with predicted emotion and confidence
        """
        if not self.ok:
            return EmotionOutput("neutral", 0.33, 0.0)
            
        try:
            # Extract and preprocess face
            x1, y1, x2, y2 = bbox
            face_bgr = image_bgr[y1:y2, x1:x2].copy()
            
            # Save debug steps if requested
            if debug_context and debug_context.get('session_dir'):
                self._save_debug_steps(image_bgr, bbox, face_bgr, debug_context)
            
            # Convert BGR to RGB for preprocessing
            face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            
            # Apply preprocessing
            input_tensor = self.preprocess(face_rgb).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
                
                # Get emotion and confidence
                emotion = EMOTION_LABELS[predicted_idx.item()]
                conf = confidence.item()
            
            # Apply temporal smoothing if enabled
            if self.enable_temporal:
                emotion, conf = self._apply_temporal_smoothing(emotion, conf)
            
            # Assess face quality (simple version)
            quality_score = self._assess_face_quality(face_bgr)
            
            logger.debug(f"POSTER V2 prediction: {emotion} (conf={conf:.3f})")
            
            return EmotionOutput(
                label=emotion, 
                confidence=conf, 
                quality_score=quality_score,
                temporal_smoothed=self.enable_temporal
            )
            
        except Exception as e:
            logger.error(f"POSTER V2 prediction failed: {e}")
            return EmotionOutput("neutral", 0.33, 0.0)
    
    def _save_debug_steps(self, image_bgr, bbox, face_bgr, debug_context):
        """Save debug preprocessing steps"""
        try:
            from pathlib import Path
            
            session_dir = Path(debug_context['session_dir'])
            frame_idx = debug_context.get('frame_idx', 0)
            face_idx = debug_context.get('face_idx', 0)
            
            # Create pipeline_steps directory
            pipeline_dir = session_dir / "pipeline_steps"
            pipeline_dir.mkdir(parents=True, exist_ok=True)
            
            base_name = f"frame_{frame_idx}_face_{face_idx}"
            
            # Step 1: Original detection
            x1, y1, x2, y2 = bbox
            detection_img = image_bgr.copy()
            cv2.rectangle(detection_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(detection_img, "POSTER V2", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imwrite(str(pipeline_dir / f"step_1_detection_{base_name}.jpg"), detection_img)
            
            # Step 2: Face crop
            cv2.imwrite(str(pipeline_dir / f"step_2_crop_{base_name}.jpg"), face_bgr)
            
            # Step 3: Resized to 224x224 (POSTER V2 input size)
            resized_face = cv2.resize(face_bgr, (224, 224))
            cv2.imwrite(str(pipeline_dir / f"step_3_resized_224x224_{base_name}.jpg"), resized_face)
            
            logger.info(f"DEBUG: Saved POSTER V2 pipeline steps to {pipeline_dir}")
            
        except Exception as e:
            logger.warning(f"Failed to save POSTER V2 debug steps: {e}")
    
    def _apply_temporal_smoothing(self, emotion: str, confidence: float) -> Tuple[str, float]:
        """Apply simple temporal smoothing"""
        # Add current prediction to history
        self.emotion_history.append((emotion, confidence))
        
        # Keep only recent history
        if len(self.emotion_history) > self.max_history:
            self.emotion_history.pop(0)
        
        # If we have enough history, apply smoothing
        if len(self.emotion_history) >= 3:
            # Simple majority voting for emotion
            recent_emotions = [e[0] for e in self.emotion_history[-3:]]
            emotion_counts = {}
            for e in recent_emotions:
                emotion_counts[e] = emotion_counts.get(e, 0) + 1
            
            # Get most common emotion
            smoothed_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]
            
            # Average confidence for the smoothed emotion
            emotion_confidences = [conf for e, conf in self.emotion_history if e == smoothed_emotion]
            smoothed_confidence = sum(emotion_confidences) / len(emotion_confidences)
            
            return smoothed_emotion, smoothed_confidence
        
        return emotion, confidence
    
    def _assess_face_quality(self, face_bgr: np.ndarray) -> float:
        """Simple face quality assessment"""
        try:
            # Convert to grayscale for quality metrics
            gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
            
            # Sharpness (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(laplacian_var / 500.0, 1.0)  # Normalize
            
            # Size score (larger faces are generally better)
            h, w = face_bgr.shape[:2]
            size_score = min((h * w) / (112 * 112), 1.0)  # Normalize to 112x112 baseline
            
            # Brightness score (avoid too dark or too bright)
            brightness = np.mean(gray)
            brightness_score = 1.0 - abs(brightness - 127.5) / 127.5
            
            # Combined quality score
            quality_score = (sharpness_score + size_score + brightness_score) / 3.0
            
            return max(0.1, min(1.0, quality_score))  # Clamp between 0.1 and 1.0
            
        except Exception as e:
            logger.warning(f"Face quality assessment failed: {e}")
            return 0.5
    
    def reset_temporal_history(self):
        """Reset temporal smoothing history (call at start of new video)"""
        self.emotion_history.clear()
        logger.debug("Temporal history reset for new video")
    
    def save_best_debug_frame(self, session_dir):
        """Save debug information for the best frame (compatibility method)"""
        logger.info(f"POSTER V2 debug session completed: {session_dir}")
        # POSTER V2 already saves debug steps during prediction, so this is just for logging
        pass