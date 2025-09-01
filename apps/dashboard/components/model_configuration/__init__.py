"""
Model Configuration Component
Handles ML model configuration for Label Studio
"""

from flask import Blueprint
from .routes import model_config_bp

# Create blueprint with template support
model_configuration_bp = Blueprint(
    'model_configuration', 
    __name__,
    template_folder='templates'
)

def init_model_configuration(app):
    """Initialize model configuration component with Flask app"""
    # Register main blueprint
    app.register_blueprint(model_configuration_bp)
    # Register API routes
    app.register_blueprint(model_config_bp)
    return model_configuration_bp

__all__ = ['model_configuration_bp', 'init_model_configuration']