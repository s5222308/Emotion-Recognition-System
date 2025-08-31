#!/usr/bin/env python3
"""
Modular ML Visual Inference Service
Clean, maintainable Flask application using modular architecture
PRESERVES exact functionality from original app.py
"""

import os
import sys
import logging
from flask import Flask

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import modular components
from routes.emotion_routes import emotion_bp
from routes.health_routes import health_bp
from services.ml_service import initialize_emotion_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLServiceApp:
    """Main ML service application class"""
    
    def __init__(self):
        self.app = None
        self.emotion_service = None
        
    def create_app(self):
        """Create and configure Flask application"""
        self.app = Flask(__name__)
        
        # Initialize ML service first
        self.emotion_service = initialize_emotion_service()
        
        # Register modular blueprints
        self.app.register_blueprint(emotion_bp)
        self.app.register_blueprint(health_bp)
        
        return self.app
    
    def run(self):
        """Start the ML service application"""
        # Display startup info
        model_name = "POSTER V2" if hasattr(self.emotion_service, 'use_poster_v2') and self.emotion_service.use_poster_v2 else "EfficientNet-B2"
        au_status = " + Action Units" if hasattr(self.emotion_service, 'enable_au_detection') and self.emotion_service.enable_au_detection else ""
        
        logger.info("=" * 60)
        logger.info(f"ML Visual Inference Service - {model_name}{au_status}")
        logger.info("Starting on: http://localhost:5003")
        logger.info("Endpoints:")
        logger.info("   - Health:    http://localhost:5003/health")
        logger.info("   - Status:    http://localhost:5003/status")  
        logger.info("   - API:       http://localhost:5003/prelabel")
        logger.info("   - Shutdown:  POST http://localhost:5003/shutdown")
        logger.info("Architecture: Component-based routes")
        logger.info("=" * 60)
        
        # Run the application
        self.app.run(host='0.0.0.0', port=5003, debug=False)

def main():
    """Main entry point"""
    ml_service = MLServiceApp()
    app = ml_service.create_app()
    ml_service.run()

if __name__ == '__main__':
    main()