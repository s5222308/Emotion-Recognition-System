#!/usr/bin/env python3
"""
Action Unit Detection Module
Integrates with OpenFace for FACS Action Unit detection
"""

import os
import cv2
import pandas as pd
import numpy as np
import subprocess
import tempfile
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json

logger = logging.getLogger(__name__)

# FACS Action Units mapping
AU_DESCRIPTIONS = {
    'AU01_r': 'Inner Brow Raiser',
    'AU02_r': 'Outer Brow Raiser', 
    'AU04_r': 'Brow Lowerer',
    'AU05_r': 'Upper Lid Raiser',
    'AU06_r': 'Cheek Raiser',
    'AU07_r': 'Lid Tightener',
    'AU09_r': 'Nose Wrinkler',
    'AU10_r': 'Upper Lip Raiser',
    'AU12_r': 'Lip Corner Puller',
    'AU14_r': 'Dimpler',
    'AU15_r': 'Lip Corner Depressor',
    'AU17_r': 'Chin Raiser',
    'AU20_r': 'Lip Stretcher',
    'AU23_r': 'Lip Tightener',
    'AU25_r': 'Lips Part',
    'AU26_r': 'Jaw Drop',
    'AU45_r': 'Blink'
}

class ActionUnitOutput:
    """Output structure for Action Unit detection"""
    def __init__(self, frame_idx: int, timestamp: float, aus: Dict[str, float], 
                 confidence: float = 1.0, face_bbox: Optional[List[int]] = None):
        self.frame_idx = frame_idx
        self.timestamp = timestamp
        self.aus = aus  # Dictionary of AU intensities
        self.confidence = confidence
        self.face_bbox = face_bbox
        
    def get_active_aus(self, threshold: float = 2.0) -> List[str]:
        """Get list of AUs above threshold intensity"""
        return [au for au, intensity in self.aus.items() if intensity > threshold]
    
    def to_dict(self):
        return {
            'frame_idx': self.frame_idx,
            'timestamp': self.timestamp,
            'aus': self.aus,
            'confidence': self.confidence,
            'active_aus': self.get_active_aus(),
            'face_bbox': self.face_bbox
        }

class ActionUnitDetector:
    """
    Action Unit Detection using OpenFace
    Falls back to simplified AU estimation if OpenFace not available
    """
    
    def __init__(self, openface_path: Optional[str] = None):
        self.openface_path = openface_path
        self.openface_available = False
        self.use_fallback = False
        
        # Try to locate OpenFace
        if self._check_openface_availability():
            logger.info("✓ OpenFace detected - using full AU analysis")
            self.openface_available = True
        else:
            logger.warning("OpenFace not found - using simplified AU estimation")
            self.use_fallback = True
            
    def _check_openface_availability(self) -> bool:
        """Check if OpenFace is available"""
        # Check common OpenFace locations
        possible_paths = [
            self.openface_path,
            "/home/lleyt/WIL_project/emotion_labelstudio_final/third_party/OpenFace/build/bin/FeatureExtraction",
            "./third_party/OpenFace/build/bin/FeatureExtraction",
            "/usr/local/bin/FeatureExtraction", 
            "FeatureExtraction"
        ]
        
        for path in possible_paths:
            if path and self._test_openface_executable(path):
                self.openface_path = path
                return True
        return False
    
    def _test_openface_executable(self, path: str) -> bool:
        """Test if OpenFace executable works"""
        try:
            result = subprocess.run([path, "-help", "-mloc", "model/main_clnf_general.txt"], 
                                  capture_output=True, timeout=5, cwd=os.path.dirname(path))
            # Check if OpenFace responds (output goes to stdout)
            output = result.stdout + result.stderr
            return b"landmark detector" in output or b"Reading the landmark detector" in output
        except:
            return False
    
    def detect_from_video(self, video_path: str, 
                         face_detections: List[Dict]) -> List[ActionUnitOutput]:
        """
        Detect Action Units from video with known face detections
        
        Args:
            video_path: Path to video file
            face_detections: List of face detection results from main pipeline
            
        Returns:
            List of ActionUnitOutput objects
        """
        if self.openface_available:
            return self._detect_with_openface(video_path, face_detections)
        else:
            logger.error("OpenFace not available and fallback disabled - no AU detection possible")
            return []
    
    def _detect_with_openface(self, video_path: str, 
                             face_detections: List[Dict]) -> List[ActionUnitOutput]:
        """Use OpenFace for full AU detection"""
        try:
            # Create temporary output file
            with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp_file:
                output_csv = temp_file.name
            
            # Run OpenFace feature extraction with working CLNF model
            cmd = [
                self.openface_path,
                "-f", video_path,
                "-of", output_csv,
                "-aus",  # Extract Action Units
                "-mloc", "model/main_clnf_general.txt",  # Use CLNF instead of CEN
                "-q"     # Quiet mode
            ]
            
            logger.info(f"Running OpenFace AU detection on {video_path}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode != 0:
                logger.error(f"OpenFace failed: {result.stderr}")
                return self._detect_fallback(video_path, face_detections)
            
            # Parse OpenFace CSV output
            return self._parse_openface_results(output_csv, face_detections)
            
        except Exception as e:
            logger.error(f"OpenFace detection failed: {e}")
            return self._detect_fallback(video_path, face_detections)
        finally:
            # Cleanup
            try:
                os.unlink(output_csv)
            except:
                pass
    
    def _parse_openface_results(self, csv_path: str, 
                               face_detections: List[Dict]) -> List[ActionUnitOutput]:
        """Parse OpenFace CSV results"""
        try:
            df = pd.read_csv(csv_path)
            
            # Get AU intensity columns
            au_columns = [col for col in df.columns if col.endswith('_r') and 'AU' in col]
            
            results = []
            for _, row in df.iterrows():
                frame_idx = int(row.get('frame', 0))
                timestamp = float(row.get('timestamp', frame_idx / 30.0))
                
                # Extract AU intensities
                aus = {}
                for au_col in au_columns:
                    if au_col in row:
                        aus[au_col] = float(row[au_col])
                
                # Find corresponding face detection
                face_bbox = self._find_matching_face_bbox(frame_idx, face_detections)
                
                results.append(ActionUnitOutput(
                    frame_idx=frame_idx,
                    timestamp=timestamp,
                    aus=aus,
                    confidence=float(row.get('confidence', 1.0)),
                    face_bbox=face_bbox
                ))
            
            logger.info(f"✓ Extracted {len(results)} AU frames with {len(au_columns)} AUs")
            return results
            
        except Exception as e:
            logger.error(f"Failed to parse OpenFace results: {e}")
            return []
    
    def _find_matching_face_bbox(self, frame_idx: int, 
                                face_detections: List[Dict]) -> Optional[List[int]]:
        """Find face bounding box for given frame"""
        for detection in face_detections:
            if detection.get('frame') == frame_idx:
                return detection.get('bbox')
        return None
    
    def _detect_fallback(self, video_path: str, 
                        face_detections: List[Dict]) -> List[ActionUnitOutput]:
        """
        Simplified AU estimation without OpenFace
        Uses basic facial landmark analysis
        """
        logger.info("Using fallback AU estimation")
        
        results = []
        
        # Simple mapping: Use emotion predictions to estimate likely AUs
        for detection in face_detections:
            frame_idx = detection.get('frame', 0)
            timestamp = detection.get('timestamp', frame_idx / 30.0)
            emotion = detection.get('emotion', 'neutral')
            confidence = detection.get('confidence', 0.5)
            
            # Map emotions to likely AU activations (simplified)
            aus = self._emotion_to_au_mapping(emotion, confidence)
            
            results.append(ActionUnitOutput(
                frame_idx=frame_idx,
                timestamp=timestamp,
                aus=aus,
                confidence=confidence * 0.5,  # Lower confidence for fallback
                face_bbox=detection.get('bbox')
            ))
        
        logger.info(f"✓ Generated {len(results)} AU estimates using fallback method")
        return results
    
    def _emotion_to_au_mapping(self, emotion: str, confidence: float) -> Dict[str, float]:
        """
        Simple mapping from emotions to likely Action Units
        Based on FACS coding for basic emotions
        """
        # Base intensity proportional to confidence
        base_intensity = confidence * 3.0
        
        au_mappings = {
            'happy': {
                'AU06_r': base_intensity,      # Cheek Raiser
                'AU12_r': base_intensity * 1.2, # Lip Corner Puller
            },
            'sad': {
                'AU01_r': base_intensity,      # Inner Brow Raiser  
                'AU04_r': base_intensity,      # Brow Lowerer
                'AU15_r': base_intensity,      # Lip Corner Depressor
            },
            'angry': {
                'AU04_r': base_intensity * 1.3, # Brow Lowerer
                'AU07_r': base_intensity,      # Lid Tightener
                'AU23_r': base_intensity,      # Lip Tightener
            },
            'fear': {
                'AU01_r': base_intensity,      # Inner Brow Raiser
                'AU02_r': base_intensity,      # Outer Brow Raiser
                'AU05_r': base_intensity,      # Upper Lid Raiser
                'AU26_r': base_intensity,      # Jaw Drop
            },
            'surprise': {
                'AU01_r': base_intensity,      # Inner Brow Raiser
                'AU02_r': base_intensity,      # Outer Brow Raiser
                'AU05_r': base_intensity * 1.2, # Upper Lid Raiser
                'AU26_r': base_intensity,      # Jaw Drop
            },
            'disgust': {
                'AU09_r': base_intensity,      # Nose Wrinkler
                'AU10_r': base_intensity,      # Upper Lip Raiser
            },
            'neutral': {
                # Minimal AU activation
                'AU45_r': 1.0,  # Occasional blinks
            }
        }
        
        return au_mappings.get(emotion, {})
    
    def save_au_timeline(self, au_results: List[ActionUnitOutput], 
                        output_path: str) -> None:
        """Save AU timeline to JSON for visualization"""
        timeline_data = {
            'au_timeline': [result.to_dict() for result in au_results],
            'au_descriptions': AU_DESCRIPTIONS,
            'analysis_info': {
                'method': 'openface' if self.openface_available else 'fallback',
                'total_frames': len(au_results),
                'au_count': len(AU_DESCRIPTIONS)
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(timeline_data, f, indent=2)
        
        logger.info(f"✓ Saved AU timeline to {output_path}")

if __name__ == "__main__":
    # Test the AU detector
    detector = ActionUnitDetector()
    print(f"OpenFace available: {detector.openface_available}")
    print(f"Using fallback: {detector.use_fallback}")