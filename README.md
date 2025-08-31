# Clinical Emotion Recognition System

Emotion recognition system designed for research applications.

## Features

- **Multi-Model Support**: EfficientNet-B2 and POSTER V2 emotion recognition models
- **Action Unit Detection**: Facial action unit analysis for detailed expression tracking
- **Real-time Processing**: Efficient video processing with face tracking
- **Clinical Dashboard**: Web-based monitoring and control interface
- **Label Studio Integration**: Seamless annotation workflow for data labeling
- **Temporal Smoothing**: Advanced temporal analysis for stable predictions

## Architecture

The system uses a microservices architecture with three main components:

1. **ML Service** (Port 5003): Core emotion recognition engine
2. **Dashboard** (Port 8081): Web-based control and monitoring interface
3. **Label Studio Connector** (Port 9091): Integration service for annotation

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, for accelerated inference)
- 8GB+ RAM recommended
- Ubuntu 20.04+ or macOS 12+

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/emotion-recognition-system.git
cd emotion-recognition-system
```

### 2. Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download dependencies
Due to file size limitations, large dependencies are not included in the repository:

**Model weights:**
- Download model weights and extract to `models/` directory
- Required: EfficientNet-B2 ONNX model, POSTER V2 weights

**Third-party libraries:**
- OpenFace (for action unit detection)
- dlib (face processing utilities)  
- POSTER_V2 (transformer-based emotion model)
- Extract to `third_party/` directory

### 5. Set up environment variables
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Quick Start

### Using the simplified restart script:
```bash
./restart.sh
```

Then select option 1 to start all services.

### Manual start:
```bash
# Start ML Service
cd apps/ml_service
python ml_service_app.py

# Start Dashboard (in new terminal)
cd apps/dashboard
python dashboard_app.py

# Start Label Studio Connector (in new terminal)
cd apps/label_studio_connector
python app.py
```

## Usage

1. **Access the Dashboard**: Open http://localhost:8081 in your browser
2. **Process Videos**: Upload videos through the dashboard interface
3. **View Results**: Real-time emotion detection results with confidence scores
4. **Export Data**: Download annotations in various formats

## Project Structure

```
├── apps/
│   ├── ml_service/         # ML inference service
│   ├── dashboard/           # Web dashboard
│   └── label_studio_connector/  # Annotation integration
├── models/                  # Model weights and configs
├── scripts/                 # Utility scripts
├── storage/                 # Data storage
├── third_party/            # External dependencies
└── web/                    # Frontend assets
```

## Models

### EfficientNet-B2
- Lightweight and fast
- 7 emotion classes
- Optimized for real-time processing

### POSTER V2
- State-of-the-art transformer architecture
- Superior accuracy on RAF-DB dataset
- Better neutral/sad distinction

## API Endpoints

### ML Service
- `GET /health` - Service health check
- `GET /status` - Detailed status page
- `POST /prelabel` - Process video for emotion recognition

### Dashboard
- `GET /` - Main dashboard interface
- `GET /api/status` - System status
- `POST /api/annotation/setup` - Configure annotation project

## Configuration

Edit `config/production.env` for production settings:
```env
USE_POSTER_V2=true
ENABLE_AU_DETECTION=true
BATCH_SIZE=32
CONFIDENCE_THRESHOLD=0.5
```

## Development

### Running tests
```bash
pytest tests/
```

### Code style
```bash
black apps/
flake8 apps/
```

## Troubleshooting

### Port already in use
```bash
# Kill processes on specific ports
lsof -ti:8081 | xargs kill -9
lsof -ti:5003 | xargs kill -9
lsof -ti:9091 | xargs kill -9
```

### GPU not detected
Ensure CUDA and cuDNN are properly installed:
```bash
nvidia-smi  # Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- POSTER V2 model from [POSTER Paper](https://arxiv.org/abs/2301.12149)
- InsightFace for face detection
- MediaPipe for face mesh alignment

## Support

For issues and questions:
- Open an issue on GitHub

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

**Note**: This system is designed for research use.
