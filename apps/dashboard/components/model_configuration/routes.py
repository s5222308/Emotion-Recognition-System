"""
Model Configuration Routes
API endpoints for model configuration and switching
"""

from flask import Blueprint, jsonify, request
import requests
import json
import os
from pathlib import Path

model_config_bp = Blueprint('model_config', __name__, url_prefix='/api/model-config')

@model_config_bp.route('/current', methods=['GET'])
def get_current_models():
    """Get currently active models from ML service"""
    try:
        # Get ML service health
        ml_response = requests.get('http://localhost:5003/health', timeout=5)
        
        if ml_response.status_code == 200:
            ml_data = ml_response.json()
            
            # Get model registry
            registry_path = Path(__file__).parent.parent.parent.parent.parent / "config" / "model_registry.json"
            registry_data = {}
            
            if registry_path.exists():
                with open(registry_path, 'r') as f:
                    registry_data = json.load(f)
            
            return jsonify({
                'current': {
                    'face_detector': ml_data.get('face_detector', 'Unknown'),
                    'emotion_model': ml_data.get('emotion_model', 'Unknown'),
                    'au_detector': ml_data.get('au_detector', 'None'),
                    'sample_rate': registry_data.get('active_config', {}).get('sample_rate', 3),
                    'service': ml_data.get('service', 'Unknown'),
                    'status': ml_data.get('status', 'unknown')
                },
                'registry': {
                    'face_detectors_count': len(registry_data.get('face_detectors', {})),
                    'emotion_models_count': len(registry_data.get('emotion_models', {})),
                    'au_detectors_count': len(registry_data.get('au_detectors', {})),
                    'active_config': registry_data.get('active_config', {}),
                    'architecture': registry_data.get('metadata', {}).get('architecture', 'Universal ONNX'),
                    'face_detectors': registry_data.get('face_detectors', {}),
                    'emotion_models': registry_data.get('emotion_models', {}),
                    'au_detectors': registry_data.get('au_detectors', {})
                }
            })
        else:
            # ML service unavailable - still provide registry data for dropdowns
            registry_path = Path(__file__).parent.parent.parent.parent.parent / "config" / "model_registry.json"
            registry_data = {}
            
            if registry_path.exists():
                with open(registry_path, 'r') as f:
                    registry_data = json.load(f)
            
            return jsonify({
                'current': {
                    'face_detector': 'Service Unavailable',
                    'emotion_model': 'Service Unavailable',
                    'service': 'ML Service Down',
                    'status': 'error'
                },
                'registry': {
                    'face_detectors_count': len(registry_data.get('face_detectors', {})),
                    'emotion_models_count': len(registry_data.get('emotion_models', {})),
                    'active_config': registry_data.get('active_config', {}),
                    'architecture': registry_data.get('metadata', {}).get('architecture', 'Universal ONNX'),
                    'face_detectors': registry_data.get('face_detectors', {}),
                    'emotion_models': registry_data.get('emotion_models', {})
                }
            }), 200  # Return 200 so frontend gets the registry data
            
    except requests.exceptions.RequestException as e:
        # Connection failed - still provide registry data for dropdowns
        registry_path = Path(__file__).parent.parent.parent.parent.parent / "config" / "model_registry.json"
        registry_data = {}
        
        if registry_path.exists():
            with open(registry_path, 'r') as f:
                registry_data = json.load(f)
        
        return jsonify({
            'current': {
                'face_detector': 'Connection Failed',
                'emotion_model': 'Connection Failed', 
                'service': 'ML Service Unreachable',
                'status': 'error'
            },
            'registry': {
                'face_detectors_count': len(registry_data.get('face_detectors', {})),
                'emotion_models_count': len(registry_data.get('emotion_models', {})),
                'active_config': registry_data.get('active_config', {}),
                'architecture': registry_data.get('metadata', {}).get('architecture', 'Universal ONNX'),
                'face_detectors': registry_data.get('face_detectors', {}),
                'emotion_models': registry_data.get('emotion_models', {})
            }
        }), 200
    except Exception as e:
        return jsonify({
            'error': 'Internal error',
            'details': str(e)
        }), 500

@model_config_bp.route('/registry', methods=['GET'])  
def get_model_registry():
    """Get full model registry"""
    try:
        registry_path = Path(__file__).parent.parent.parent.parent.parent / "config" / "model_registry.json"
        
        if registry_path.exists():
            with open(registry_path, 'r') as f:
                registry_data = json.load(f)
            return jsonify(registry_data)
        else:
            return jsonify({
                'error': 'Model registry not found',
                'path': str(registry_path)
            }), 404
            
    except Exception as e:
        return jsonify({
            'error': 'Failed to load model registry',
            'details': str(e)
        }), 500

@model_config_bp.route('/switch', methods=['POST'])
def switch_models():
    """Switch active models via ML service"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Forward request to ML service
        ml_response = requests.post(
            'http://localhost:5003/models/switch',
            json=data,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        
        if ml_response.status_code == 200:
            result = ml_response.json()
            return jsonify(result), 200
        else:
            error_data = ml_response.json() if ml_response.content else {'error': 'Unknown error'}
            return jsonify(error_data), ml_response.status_code
            
    except requests.exceptions.RequestException as e:
        return jsonify({
            'error': 'Failed to connect to ML service',
            'details': str(e)
        }), 503
    except Exception as e:
        return jsonify({
            'error': 'Internal error',
            'details': str(e)
        }), 500