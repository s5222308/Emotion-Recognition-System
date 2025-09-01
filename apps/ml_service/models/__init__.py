"""
ML Models
Core machine learning models for emotion recognition and face analysis
Using Universal ONNX system now
"""
import os
import logging

logger = logging.getLogger(__name__)

# Import only the models that still exist
from .action_unit_detector import ActionUnitDetector, ActionUnitOutput
from .universal_onnx import UniversalONNXModel, ModelOutput

__all__ = ['ActionUnitDetector', 'ActionUnitOutput', 'UniversalONNXModel', 'ModelOutput']

logger.info("Models module loaded with Universal ONNX system")