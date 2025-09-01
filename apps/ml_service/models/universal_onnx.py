#!/usr/bin/env python3
"""
Universal ONNX Model System
ONE loader for ALL models - faces, emotions, anything!
Replaces all the complex model-specific wrapper classes
"""

import onnxruntime as ort
import numpy as np
import cv2
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import json

logger = logging.getLogger(__name__)

class ModelOutput:
    """Universal output structure"""
    def __init__(self, label: str, confidence: float, bbox: Optional[List] = None, 
                 raw_outputs: Optional[np.ndarray] = None):
        self.label = label
        self.confidence = confidence
        self.bbox = bbox  # For face detectors: [x1, y1, x2, y2]
        self.raw_outputs = raw_outputs

class UniversalONNXModel:
    """
    Universal ONNX model that can handle ANY ONNX model
    - Face detectors (YOLO, SCRFD, etc.)
    - Emotion models (EfficientNet, ResNet, Transformers, etc.)
    - Any other vision model
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_path = config['onnx_path']
        self.model_type = config.get('type', 'classifier')  # 'classifier', 'detector', 'custom'
        self.input_size = tuple(config.get('input_size', [224, 224]))
        self.labels = config.get('labels', [])
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        
        # ONNX session setup
        providers = config.get('providers', ['CPUExecutionProvider'])
        self.session = None
        self.input_name = None
        self.output_names = None
        self.ok = False
        
        self._initialize_session(providers)
    
    def _initialize_session(self, providers: List[str]):
        """Initialize ONNX runtime session"""
        try:
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            # Create ONNX session
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            
            # Get input/output info
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            input_shape = self.session.get_inputs()[0].shape
            logger.info(f"✓ Universal ONNX model loaded: {Path(self.model_path).name}")
            logger.info(f"  - Input shape: {input_shape}")
            logger.info(f"  - Input name: {self.input_name}")
            logger.info(f"  - Output names: {self.output_names}")
            logger.info(f"  - Model type: {self.model_type}")
            logger.info(f"  - Providers: {providers}")
            
            self.ok = True
            
        except Exception as e:
            logger.error(f"Failed to initialize ONNX model {self.model_path}: {e}")
            self.ok = False
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Universal image preprocessing"""
        # Handle different input formats
        if len(image.shape) == 3 and image.shape[2] == 3:  # BGR
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        h, w = self.input_size
        image = cv2.resize(image, (w, h))
        
        # Normalize based on config
        normalization = self.config.get('normalization', 'imagenet')
        if normalization == 'imagenet':
            # Standard ImageNet normalization
            image = image.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = (image - mean) / std
        elif normalization == 'zero_one':
            # Simple 0-1 normalization
            image = image.astype(np.float32) / 255.0
        elif normalization == 'neg_one_one':
            # -1 to 1 normalization
            image = (image.astype(np.float32) / 255.0 - 0.5) * 2.0
        
        # Convert to NCHW format (batch, channels, height, width)
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        
        # Ensure float32 (ONNX models expect float32, not float64)
        image = image.astype(np.float32)
        
        return image
    
    def _postprocess_classifier(self, outputs: List[np.ndarray]) -> ModelOutput:
        """Post-process classification model outputs"""
        logits = outputs[0][0]  # Remove batch dimension
        
        # Apply softmax if needed
        if self.config.get('apply_softmax', True):
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)
        else:
            probs = logits
        
        # Get prediction
        pred_idx = np.argmax(probs)
        confidence = float(probs[pred_idx])
        
        if pred_idx < len(self.labels):
            label = self.labels[pred_idx]
        else:
            label = f"class_{pred_idx}"
        
        return ModelOutput(label=label, confidence=confidence, raw_outputs=outputs)
    
    def _postprocess_detector(self, outputs: List[np.ndarray], 
                            original_shape: Tuple[int, int]) -> List[ModelOutput]:
        """Post-process object detection model outputs (YOLO format)"""
        detections = []
        
        # Handle YOLO format: [batch, num_detections, 6] where 6 = [x1, y1, x2, y2, conf, class]
        predictions = outputs[0][0]  # Remove batch dimension
        
        orig_h, orig_w = original_shape[:2]
        input_h, input_w = self.input_size
        
        for detection in predictions:
            if len(detection) >= 6:
                x1, y1, x2, y2, conf, class_id = detection[:6]
                
                if conf >= self.confidence_threshold:
                    # Convert to original image coordinates
                    x1 = int(x1 * orig_w / input_w)
                    y1 = int(y1 * orig_h / input_h)
                    x2 = int(x2 * orig_w / input_w)
                    y2 = int(y2 * orig_h / input_h)
                    
                    # Get class label
                    class_idx = int(class_id)
                    if class_idx < len(self.labels):
                        label = self.labels[class_idx]
                    else:
                        label = f"class_{class_idx}"
                    
                    detections.append(ModelOutput(
                        label=label,
                        confidence=float(conf),
                        bbox=[x1, y1, x2, y2]
                    ))
        
        return detections
    
    def predict(self, image: np.ndarray, bbox: Optional[List] = None) -> Any:
        """
        Universal prediction method
        - For classifiers: returns single ModelOutput  
        - For detectors: returns List[ModelOutput]
        """
        if not self.ok:
            logger.error("Model not initialized")
            if self.model_type == 'detector':
                return []
            else:
                return ModelOutput("error", 0.0)
        
        try:
            # If bbox provided, crop the image (for emotion classification)
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                image = image[y1:y2, x1:x2]
            
            original_shape = image.shape
            
            # Preprocess
            input_tensor = self._preprocess_image(image)
            
            # Run inference
            outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
            
            # Post-process based on model type
            if self.model_type == 'detector':
                return self._postprocess_detector(outputs, original_shape)
            else:  # classifier or custom
                return self._postprocess_classifier(outputs)
                
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            if self.model_type == 'detector':
                return []
            else:
                return ModelOutput("error", 0.0)

def load_model_from_registry(model_id: str, registry_path: str = "config/model_registry.json") -> UniversalONNXModel:
    """Load a model from the registry by ID"""
    try:
        with open(registry_path, 'r') as f:
            registry = json.load(f)
        
        # Check in both face_detectors and emotion_models
        model_config = None
        if model_id in registry.get('face_detectors', {}):
            model_config = registry['face_detectors'][model_id]
        elif model_id in registry.get('emotion_models', {}):
            model_config = registry['emotion_models'][model_id]
        
        if model_config is None:
            raise ValueError(f"Model {model_id} not found in registry")
        
        return UniversalONNXModel(model_config)
        
    except Exception as e:
        logger.error(f"Failed to load model {model_id}: {e}")
        raise

# Backwards compatibility classes for easy migration
class EmotionOutput:
    """Compatibility class - maps to ModelOutput"""
    def __init__(self, label: str, confidence: float, quality_score: float = 1.0):
        self.label = label
        self.confidence = confidence
        self.quality_score = quality_score

def create_emotion_output(model_output: ModelOutput) -> EmotionOutput:
    """Convert ModelOutput to EmotionOutput for backwards compatibility"""
    return EmotionOutput(
        label=model_output.label,
        confidence=model_output.confidence,
        quality_score=1.0
    )