"""
ML Models
Core machine learning models for emotion recognition and face analysis
"""
import os
import logging

logger = logging.getLogger(__name__)

from .emotion_inference import EnhancedEmotionModel
from .action_unit_detector import ActionUnitDetector, ActionUnitOutput

# Honest import detection for POSTER_V2
USE_POSTER_V2 = os.getenv("USE_POSTER_V2", "false").lower() == "true"
POSTER_V2_AVAILABLE = False
PosterV2EmotionModel = None

if USE_POSTER_V2:
    try:
        # This import will run the fixed logic and really test availability
        from .poster_v2_model import PosterV2EmotionModel
        POSTER_V2_AVAILABLE = True
        __all__ = ['EnhancedEmotionModel', 'PosterV2EmotionModel', 'ActionUnitDetector', 'ActionUnitOutput']
    except Exception as e:
        logger.exception("POSTER_V2 not available: %s", e)
        POSTER_V2_AVAILABLE = False
        __all__ = ['EnhancedEmotionModel', 'ActionUnitDetector', 'ActionUnitOutput']
else:
    __all__ = ['EnhancedEmotionModel', 'ActionUnitDetector', 'ActionUnitOutput']

model_choice = "POSTER_V2" if (USE_POSTER_V2 and POSTER_V2_AVAILABLE) else "EfficientNet-B2"
logger.info("Emotion model selected: %s (USE_POSTER_V2=%s, available=%s)",
            model_choice, USE_POSTER_V2, POSTER_V2_AVAILABLE)