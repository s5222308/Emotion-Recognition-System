# Models Directory

This directory contains all the neural network models used by the SOTA Emotion Recognition System.

## Directory Structure

```
models/
├── emotieff/                 # EfficientNet-B2 emotion models (current working)
│   └── fer_enet_b2_7.onnx   # 7-class emotion classifier
├── efficientvit/            # EfficientViT-M5 models (SOTA Phase 3)
│   ├── efficientvit_m5_backbone.onnx
│   └── efficientvit_m5_emotion_head.onnx
├── scrfd/                   # SCRFD face detection models
│   └── scrfd_10g_bnkps.onnx # SCRFD-10GF face detector
├── insightface/             # InsightFace models (legacy face detection)
│   └── models/
├── affectnet_fer/           # Alternative emotion models
│   └── affectnet_mobilenet_7.onnx
└── ravdess_finetuned/       # RAVDESS dataset finetuned models
```

## Model Details

### Current Working Models (SOTA Phase 3)

#### Face Detection: SCRFD-10GF
- **File**: `scrfd/scrfd_10g_bnkps.onnx`
- **Accuracy**: 96.06% mAP
- **Speed**: 3x faster than RetinaFace
- **Input Size**: 640x640
- **Output**: Face bounding boxes and landmarks

#### Emotion Recognition: EfficientViT-M5
- **Files**: 
  - `efficientvit/efficientvit_m5_backbone.onnx` (feature extractor)
  - `efficientvit/efficientvit_m5_emotion_head.onnx` (classifier)
- **Accuracy**: 94.28% on FER2013
- **Speed**: 1.9x faster than EfficientNet-B2
- **Input Size**: 224x224
- **Classes**: 8 emotions (neutral, happy, sad, angry, fearful, disgusted, surprised, contemptuous)

#### Fallback: Enhanced EfficientNet-B2
- **File**: `emotieff/fer_enet_b2_7.onnx`
- **Classes**: 7 emotions (angry, disgust, fear, happy, neutral, sad, surprise)
- **Input Size**: 260x260
- **Features**: MediaPipe alignment, temporal processing

### Legacy/Alternative Models

#### InsightFace (Legacy)
- **Purpose**: Original face detection and recognition
- **Status**: Replaced by SCRFD for better performance
- **Location**: `insightface/models/`

#### AffectNet Model
- **File**: `affectnet_fer/affectnet_mobilenet_7.onnx`
- **Purpose**: Alternative emotion classifier
- **Status**: Available but not actively used

#### RAVDESS Finetuned
- **Location**: `ravdess_finetuned/`
- **Purpose**: Models finetuned on RAVDESS dataset
- **Status**: Experimental/research

## Model Usage by Configuration

### SOTA_PHASE3 (Recommended)
- Face Detection: `scrfd/scrfd_10g_bnkps.onnx`
- Emotion: `efficientvit/efficientvit_m5_*.onnx` → fallback to `emotieff/fer_enet_b2_7.onnx`

### SOTA_PHASE2
- Face Detection: `scrfd/scrfd_10g_bnkps.onnx`
- Emotion: `emotieff/fer_enet_b2_7.onnx`

### SOTA_PHASE1
- Face Detection: `scrfd/scrfd_10g_bnkps.onnx`
- Emotion: `emotieff/fer_enet_b2_7.onnx`

### CURRENT (Legacy)
- Face Detection: InsightFace RetinaFace
- Emotion: `emotieff/fer_enet_b2_7.onnx`

## Setup Instructions

### EfficientViT Models
Run the setup script to create placeholder models:
```bash
python setup_efficientvit.py
```

### SCRFD Models
SCRFD models are automatically downloaded by InsightFace.

### Manual Model Replacement
To use real trained models:
1. Replace dummy ONNX files with actual trained models
2. Ensure input/output shapes match expected format
3. Test with health endpoint: `curl http://localhost:5003/health`

## Performance Benchmarks

| Model | Accuracy | Speed | Memory | Notes |
|-------|----------|-------|---------|-------|
| SCRFD-10GF | 96.06% | 3x faster | 30MB | vs RetinaFace |
| EfficientViT-M5 | 94.28% | 1.9x faster | 0.9GB | vs EfficientNet-B2 |
| EfficientNet-B2 | 92.1% | Baseline | 0.8GB | Current working |
| RetinaFace | 94.2% | Baseline | 40MB | Legacy |

## Troubleshooting

### Missing Models
- Check if ONNX files exist in expected locations
- Run setup scripts to create placeholder models
- Verify file permissions and accessibility

### GPU/CPU Providers
- Models automatically fall back to CPU if CUDA unavailable
- Check logs for provider selection messages
- Ensure ONNX Runtime GPU version installed for CUDA support

### Model Format Issues
- Ensure ONNX models are compatible with ONNX Runtime version
- Check input/output tensor shapes match expected format
- Verify model was exported correctly from training framework