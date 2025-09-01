#!/usr/bin/env python3
"""
Enhanced Video Processing Utilities
Adds temporal smoothing, face tracking, and advanced video analysis
Works with Universal ONNX models
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from collections import deque, Counter
import logging

logger = logging.getLogger(__name__)

class TemporalProcessor:
    """
    Handles temporal processing for video emotion recognition
    - Sliding window smoothing
    - Outlier filtering
    - Temporal consistency
    - Face tracking
    """
    
    def __init__(self, window_size: int = 15, outlier_threshold: float = 2.0):
        """
        Initialize temporal processor
        
        Args:
            window_size: Number of frames for sliding window (default 15 = 0.5s at 30fps)
            outlier_threshold: Standard deviations for outlier detection
        """
        self.window_size = window_size
        self.outlier_threshold = outlier_threshold
        self.emotion_buffer = deque(maxlen=window_size)
        self.confidence_buffer = deque(maxlen=window_size)
        self.face_tracks = {}  # Track faces across frames
        self.next_track_id = 0
        
    def add_frame_predictions(self, predictions: List[Dict], frame_idx: int) -> List[Dict]:
        """
        Process predictions for a single frame with temporal context
        
        Args:
            predictions: List of face/emotion predictions for current frame
            frame_idx: Current frame index
            
        Returns:
            Enhanced predictions with temporal smoothing
        """
        enhanced_predictions = []
        
        for pred in predictions:
            # Track faces across frames
            track_id = self._match_or_create_track(pred['bbox'], frame_idx)
            pred['track_id'] = track_id
            
            # Add to temporal buffer
            self.emotion_buffer.append(pred['emotion'])
            self.confidence_buffer.append(pred['confidence'])
            
            # Apply temporal smoothing
            smoothed_emotion = self._get_smoothed_emotion()
            smoothed_confidence = self._get_smoothed_confidence()
            
            # Detect outliers
            is_outlier = self._is_outlier(pred['confidence'])
            
            # Create enhanced prediction
            enhanced_pred = pred.copy()
            enhanced_pred.update({
                'smoothed_emotion': smoothed_emotion,
                'smoothed_confidence': float(smoothed_confidence),
                'temporal_consistency': float(self._calculate_consistency()),
                'is_outlier': bool(is_outlier),
                'window_size': len(self.emotion_buffer)
            })
            
            enhanced_predictions.append(enhanced_pred)
            
        return enhanced_predictions
    
    def _match_or_create_track(self, bbox: List[int], frame_idx: int) -> int:
        """
        Match face to existing track or create new one
        Simple IoU-based tracking
        """
        best_track_id = None
        best_iou = 0.3  # Lower IoU threshold for single-person tracking
        
        # Check existing tracks
        for track_id, track_info in self.face_tracks.items():
            # More lenient for single-person videos - allow up to 30 frames gap
            if frame_idx - track_info['last_frame'] > 30:  # Track lost if missing for 30 frames
                continue
                
            iou = self._calculate_iou(bbox, track_info['last_bbox'])
            if iou > best_iou:
                best_iou = iou
                best_track_id = track_id
        
        if best_track_id is not None:
            # Update existing track
            self.face_tracks[best_track_id]['last_bbox'] = bbox
            self.face_tracks[best_track_id]['last_frame'] = frame_idx
            return best_track_id
        else:
            # Create new track
            track_id = self.next_track_id
            self.next_track_id += 1
            self.face_tracks[track_id] = {
                'last_bbox': bbox,
                'last_frame': frame_idx,
                'start_frame': frame_idx
            }
            return track_id
    
    def _calculate_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """Calculate Intersection over Union for two bboxes"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # Calculate intersection
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
            return 0.0
        
        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        
        # Calculate union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def _get_smoothed_emotion(self) -> str:
        """
        Get smoothed emotion using majority voting over window
        """
        if not self.emotion_buffer:
            return "neutral"
        
        # Weighted voting based on recency and confidence
        emotion_votes = {}
        for i, (emotion, conf) in enumerate(zip(self.emotion_buffer, self.confidence_buffer)):
            # More recent frames get higher weight
            weight = (i + 1) / len(self.emotion_buffer) * conf
            emotion_votes[emotion] = emotion_votes.get(emotion, 0) + weight
        
        # Return emotion with highest weighted votes
        return max(emotion_votes, key=emotion_votes.get)
    
    def _get_smoothed_confidence(self) -> float:
        """
        Get smoothed confidence using weighted average
        """
        if not self.confidence_buffer:
            return 0.0
        
        # Weighted average with more weight on recent frames
        weights = np.linspace(0.5, 1.0, len(self.confidence_buffer))
        weighted_sum = sum(c * w for c, w in zip(self.confidence_buffer, weights))
        return weighted_sum / weights.sum()
    
    def _calculate_consistency(self) -> float:
        """
        Calculate temporal consistency score (0-1)
        Higher score means more consistent emotions across window
        """
        if len(self.emotion_buffer) < 2:
            return 1.0
        
        # Count emotion changes
        changes = sum(1 for i in range(1, len(self.emotion_buffer)) 
                     if self.emotion_buffer[i] != self.emotion_buffer[i-1])
        
        # Normalize to 0-1 (fewer changes = higher consistency)
        max_changes = len(self.emotion_buffer) - 1
        consistency = 1.0 - (changes / max_changes) if max_changes > 0 else 1.0
        
        return consistency
    
    def _is_outlier(self, confidence: float) -> bool:
        """
        Detect if current confidence is an outlier
        """
        if len(self.confidence_buffer) < 3:
            return False
        
        mean = np.mean(self.confidence_buffer)
        std = np.std(self.confidence_buffer)
        
        if std == 0:
            return False
        
        z_score = abs((confidence - mean) / std)
        return z_score > self.outlier_threshold
    
    def get_video_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for the entire video
        """
        if not self.emotion_buffer:
            return {}
        
        # Get emotion distribution
        emotion_counts = Counter(self.emotion_buffer)
        total = len(self.emotion_buffer)
        emotion_distribution = {
            emotion: count / total 
            for emotion, count in emotion_counts.items()
        }
        
        # Find dominant emotion
        dominant_emotion = max(emotion_counts, key=emotion_counts.get)
        
        # Calculate average confidence
        avg_confidence = np.mean(self.confidence_buffer) if self.confidence_buffer else 0.0
        
        return {
            'dominant_emotion': dominant_emotion,
            'emotion_distribution': emotion_distribution,
            'average_confidence': float(avg_confidence),
            'temporal_consistency': float(self._calculate_consistency()),
            'num_tracks': len(self.face_tracks),
            'frames_processed': len(self.emotion_buffer)
        }


class AdaptiveSampling:
    """
    Adaptive frame sampling based on scene changes and motion
    """
    
    def __init__(self, min_interval: int = 1, max_interval: int = 10):
        """
        Initialize adaptive sampling
        
        Args:
            min_interval: Minimum frames between samples
            max_interval: Maximum frames between samples
        """
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.last_frame = None
        self.last_hist = None
        
    def should_process_frame(self, frame: np.ndarray, frame_idx: int, 
                            last_processed_idx: int) -> bool:
        """
        Determine if frame should be processed based on content changes
        
        Args:
            frame: Current frame
            frame_idx: Current frame index
            last_processed_idx: Index of last processed frame
            
        Returns:
            True if frame should be processed
        """
        # Always process if we've hit max interval
        if frame_idx - last_processed_idx >= self.max_interval:
            return True
        
        # Don't process if under min interval
        if frame_idx - last_processed_idx < self.min_interval:
            return False
        
        # Calculate scene change metric
        if self.last_frame is not None:
            change_score = self._calculate_change_score(frame)
            
            # Adaptive threshold based on interval
            interval_ratio = (frame_idx - last_processed_idx) / self.max_interval
            threshold = 0.1 + (0.3 * (1 - interval_ratio))  # Lower threshold as time passes
            
            if change_score > threshold:
                self.last_frame = frame.copy()
                return True
        else:
            self.last_frame = frame.copy()
            return True
        
        return False
    
    def _calculate_change_score(self, frame: np.ndarray) -> float:
        """
        Calculate change score between current and last frame
        Uses histogram difference and edge detection
        """
        # Convert to grayscale
        gray = np.mean(frame, axis=2) if len(frame.shape) == 3 else frame
        
        # Calculate histogram
        hist = np.histogram(gray, bins=32, range=(0, 255))[0]
        hist = hist / hist.sum()  # Normalize
        
        if self.last_hist is not None:
            # Calculate histogram difference
            hist_diff = np.sum(np.abs(hist - self.last_hist))
            
            # Calculate structural difference (simplified)
            if self.last_frame is not None:
                last_gray = np.mean(self.last_frame, axis=2) if len(self.last_frame.shape) == 3 else self.last_frame
                struct_diff = np.mean(np.abs(gray - last_gray)) / 255.0
                
                # Combine metrics
                change_score = 0.7 * hist_diff + 0.3 * struct_diff
            else:
                change_score = hist_diff
        else:
            change_score = 1.0  # First frame
        
        self.last_hist = hist
        return change_score


def apply_temporal_filtering(predictions: List[Dict], 
                            filter_type: str = 'gaussian',
                            window_size: int = 5) -> List[Dict]:
    """
    Apply temporal filtering to smooth predictions
    
    Args:
        predictions: List of predictions with confidence scores
        filter_type: Type of filter ('gaussian', 'median', 'bilateral')
        window_size: Size of filter window
        
    Returns:
        Filtered predictions
    """
    if len(predictions) < window_size:
        return predictions
    
    # Extract confidence scores
    confidences = np.array([p['confidence'] for p in predictions])
    
    if filter_type == 'gaussian':
        # Gaussian smoothing
        from scipy.ndimage import gaussian_filter1d
        smoothed = gaussian_filter1d(confidences, sigma=window_size/4)
    elif filter_type == 'median':
        # Median filtering (good for outlier removal)
        from scipy.signal import medfilt
        smoothed = medfilt(confidences, kernel_size=min(window_size, len(confidences)))
    elif filter_type == 'bilateral':
        # Bilateral filtering (preserves edges/transitions)
        smoothed = bilateral_filter_1d(confidences, window_size)
    else:
        smoothed = confidences
    
    # Update predictions with smoothed confidences
    filtered_predictions = []
    for i, pred in enumerate(predictions):
        filtered_pred = pred.copy()
        filtered_pred['original_confidence'] = pred['confidence']
        filtered_pred['confidence'] = float(smoothed[i])
        filtered_predictions.append(filtered_pred)
    
    return filtered_predictions


def bilateral_filter_1d(data: np.ndarray, window_size: int, 
                        sigma_space: float = None, 
                        sigma_range: float = None) -> np.ndarray:
    """
    1D bilateral filter for preserving transitions
    """
    if sigma_space is None:
        sigma_space = window_size / 4
    if sigma_range is None:
        sigma_range = np.std(data)
    
    filtered = np.zeros_like(data)
    half_window = window_size // 2
    
    for i in range(len(data)):
        start = max(0, i - half_window)
        end = min(len(data), i + half_window + 1)
        
        # Spatial weights
        positions = np.arange(start, end)
        spatial_weights = np.exp(-0.5 * ((positions - i) / sigma_space) ** 2)
        
        # Range weights
        values = data[start:end]
        range_weights = np.exp(-0.5 * ((values - data[i]) / sigma_range) ** 2)
        
        # Combined weights
        weights = spatial_weights * range_weights
        weights /= weights.sum()
        
        # Apply filter
        filtered[i] = np.sum(values * weights)
    
    return filtered


def detect_emotion_transitions(predictions: List[Dict], 
                              min_duration: int = 5) -> List[Dict]:
    """
    Detect significant emotion transitions in video
    
    Args:
        predictions: List of predictions
        min_duration: Minimum frames for stable emotion
        
    Returns:
        List of transition events
    """
    transitions = []
    current_emotion = None
    current_start = 0
    current_frames = []
    
    for i, pred in enumerate(predictions):
        emotion = pred.get('smoothed_emotion', pred.get('emotion'))
        
        if emotion != current_emotion:
            if current_emotion is not None and len(current_frames) >= min_duration:
                # Record transition
                transitions.append({
                    'from_emotion': current_emotion,
                    'to_emotion': emotion,
                    'frame': i,
                    'duration': len(current_frames),
                    'confidence': np.mean([p['confidence'] for p in current_frames])
                })
            
            current_emotion = emotion
            current_start = i
            current_frames = [pred]
        else:
            current_frames.append(pred)
    
    return transitions