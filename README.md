# InvigilAI

**AI-Based Real-Time Exam Cheating Detection System for Physical Examination Halls**

An intelligent multi-modal fusion framework combining VideoMAE, YOLO-based detectors, and pose reasoning for robust, explainable detection of multiple cheating behaviors in examination environments.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Team](#project-team)
- [Citation](#citation)
- [License](#license)

---

## Overview

### Problem Statement

Offline examination cheating is a critical challenge in educational institutions. Traditional human invigilation is insufficient due to:

- **Fatigue and distraction** in manual monitoring
- **Limitation to single behaviors** in AI-based solutions
- **Image-based approaches** lacking temporal understanding
- **Poor generalization** to real examination environments

Existing AI solutions are limited to binary classification, single-behavior detection, and lack explainability.

### Solution: InvigilAI

**InvigilAI** proposes an intelligent multi-modal framework that:

✅ Detects **5 cheating classes** (Normal, Copying, Mobile Usage, Gesture Sharing, Hidden Notes)  
✅ Combines **action recognition**, **object detection**, and **pose reasoning**  
✅ Provides **explainable decisions** with evidence generation  
✅ Achieves **~90% accuracy** on custom examination dataset  
✅ Enables **real-time deployment** on commodity GPUs  

---

## Key Features

| Feature | Capability |
|---------|-----------|
| **Multi-Class Detection** | 5 cheating behavior classes |
| **Temporal Modeling** | Transformer-based action recognition (VideoMAE) |
| **Object Detection** | Mobile phones & hidden notes via YOLO |
| **Pose Reasoning** | Behavioral analysis via body keypoints |
| **Student Tracking** | Automatic ID assignment & temporal continuity |
| **Evidence Generation** | Annotated videos + CSV event logs |
| **Real-Time Processing** | 12–30 FPS on NVIDIA T4 GPU |
| **Explainability** | Pose-based behavioral insights |

---

## System Architecture

### Overall Pipeline

```
Input Video
    ↓
Student Detection & Tracking
    ↓
Student Clip Extraction
    ↓
┌─────────────────────────────┐
│   Parallel Processing       │
├─────────────────────────────┤
│ • VideoMAE (Action)         │
│ • YOLO Mobile (Detection)   │
│ • YOLO Notes (Detection)    │
│ • YOLO Pose (Reasoning)     │
└─────────────────────────────┘
    ↓
Weighted Fusion & Temporal Smoothing
    ↓
Final Prediction
    ↓
┌──────────────────┬──────────────────┐
│ Annotated Video  │  CSV Event Logs  │
└──────────────────┴──────────────────┘
```

### Module Descriptions

#### 1. **VideoMAE (Action Recognition)**
- **Model**: MCG-NJU/videomae-base-finetuned-kinetics
- **Input**: 16 frames @ 224×224 pixels
- **Output**: Class probability + confidence score
- **Strength**: Long-range temporal dependency modeling
- **Weakness**: Struggles with small objects (phones, notes)

#### 2. **YOLO-Mobile Detector**
- **Model**: YOLOv8m
- **Task**: Localize mobile phones in student frames
- **Output**: Bounding box + confidence
- **Strength**: Accurate small object detection
- **Limitation**: No temporal understanding

#### 3. **YOLO-Notes Detector**
- **Model**: YOLOv8m
- **Task**: Detect hidden notes and unauthorized papers
- **Output**: Bounding box + confidence
- **Strength**: Precise paper localization
- **Limitation**: Limited contextual reasoning

#### 4. **YOLO Pose Module**
- **Model**: YOLOv8n-pose
- **Output**: 17 body keypoints
- **Rules Implemented**:
  - Side glance (head orientation)
  - Leaning (shoulder asymmetry)
  - Reaching (hand-torso distance)
  - Looking toward neighbor (face direction)
- **Strength**: Explainable behavioral reasoning
- **Limitation**: Hand-crafted thresholds

#### 5. **Fusion & Temporal Smoothing**

```
P_fusion = w₁·P_VideoMAE + w₂·P_Mobile + w₃·P_Notes + w₄·P_Pose

where w₁=0.50, w₂=0.20, w₃=0.15, w₄=0.15
```

- **Priority Rules**: Mobile detection (>0.85) and Notes detection (>0.80) override others
- **Temporal Smoothing**: Majority voting over 10-frame window to reduce flickering

---

## Dataset

### Overview

A novel **5-class video-based examination cheating dataset** collected under realistic conditions from ~65 participants.

### Dataset Statistics

| Metric | Value |
|--------|-------|
| **Total Videos** | 637 raw videos |
| **Total Clips** | 3,100 processed clips |
| **Participants** | ~65 diverse individuals |
| **Format** | MP4/MOV @ 480p–1080p |
| **Environment** | Theater-style examination setting |

### Class Distribution

| Class | Count | Examples |
|-------|-------|----------|
| **Normal** | 964 | Writing, reading, looking forward |
| **Copying** | 807 | Side glances, head turning, object exchange |
| **Mobile Usage** | 569 | Holding phone, looking downward, hidden interaction |
| **Gesture Sharing** | 322 | Hand signals, finger gestures, communication |
| **Hidden Notes** | 438 | Paper notes, concealed documents, cheat sheets |

### Dataset Evolution

The dataset was created in 5 versions, progressively increasing realism:

| Version | Environment | Characteristics | Purpose |
|---------|-------------|-----------------|---------|
| **V1** | Controlled | Single student, simple background | Feasibility testing |
| **V2** | Multi-angle | Multiple viewpoints, varied lighting | Robustness improvement |
| **V3** | Multi-student | Theater environment, interactions | Realistic scenarios |
| **V4** | Classroom | Natural behavior, long recordings | Temporal diversity |
| **Final** | Theater-style | Multiple students, varied perspectives | Real examination simulation |

### Recording Conditions

- **Lighting**: Bright, Medium, Dim
- **Camera Views**: Front, Side, Oblique, Diagonal, Long-distance
- **Resolutions**: 480p, 720p, 1080p
- **Challenges**: Motion blur, compression artifacts, occlusions, scale variations

### Data Preprocessing

```
Raw Video
  ↓
Person Detection (YOLO)
  ↓
Multi-Object Tracking
  ↓
Student ID Assignment
  ↓
Individual Student Cropping
  ↓
Clip Generation (16-frame windows)
  ↓
Manual Annotation & Quality Assurance
  ↓
CSV Labeling
  ↓
Final Dataset
```

**Dataset Available**: [Kaggle](https://kaggle.com) (link to be added)

---

## Installation

### Requirements

- Python 3.10+
- CUDA 12.x (recommended) or CPU-only mode
- 15GB RAM (Google Colab Pro) or equivalent GPU with 8GB+ VRAM

### Clone Repository

```bash
git clone https://github.com/Rimsha002/InvigilAI-AI-Based-Exam-Cheating-Detection-System-for-Physical-Examination-Halls-.git
cd InvigilAI
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Verify CUDA (Optional)

```python
import torch
print(torch.cuda.is_available())  # Should output: True
```

### Repository Structure

```
InvigilAI/
├── dataset/                  # Processed clips & annotations
├── models/                   # Pre-trained model weights
│   ├── videomae_model.pt
│   ├── yolo_mobile.pt
│   ├── yolo_notes.pt
│   └── yolo_pose.pt
├── notebooks/               # Training & inference scripts
├── sample_outputs/          # Example results
├── requirements.txt
├── train_videomae.py       # VideoMAE fine-tuning
├── train_yolo_mobile.py    # Mobile detector training
├── train_yolo_notes.py     # Notes detector training
├── fusion_inference.py     # Main inference pipeline
└── README.md
```

---

## Usage

### 1. Training VideoMAE

```bash
python train_videomae.py \
  --data_path ./dataset \
  --epochs 20 \
  --batch_size 4 \
  --learning_rate 1e-5
```

**Hyperparameters**:

| Parameter | Value |
|-----------|-------|
| Frames | 16 |
| Resolution | 224×224 |
| Patch Size | 16×16 |
| Optimizer | AdamW |
| Weight Decay | 0.05 |
| Warmup Ratio | 0.1 |

### 2. Training YOLO Mobile Detector

```bash
yolo task=detect mode=train \
  model=yolov8m.pt \
  data=mobile.yaml \
  epochs=100 \
  imgsz=640
```

### 3. Training YOLO Notes Detector

```bash
yolo task=detect mode=train \
  model=yolov8m.pt \
  data=notes.yaml \
  epochs=100 \
  imgsz=640
```

### 4. Running Inference (Fusion Framework)

```bash
python fusion_inference.py \
  --input_video ./sample.mp4 \
  --output_dir ./outputs \
  --gpu 0
```

**Output**:
- `output_annotated.mp4` – Video with bounding boxes & labels
- `events.csv` – Frame-by-frame predictions with confidence scores

### CSV Output Format

```csv
Timestamp,Frame,Student_ID,VideoMAE_Label,VideoMAE_Conf,Mobile_Conf,Notes_Conf,Pose_Score,Final_Label,Final_Confidence
00:01:22,560,3,Copying,0.81,0.00,0.00,0.73,Copying,0.88
00:02:35,845,7,Mobile,0.75,0.94,0.00,0.42,Mobile,0.91
```

---

## Results

### Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| R3D-18 | 63.0% | 61.8% | 60.4% | 61.1% |
| VideoMAE | 78.25% | 77.9% | 77.3% | 77.6% |
| VideoMAE + Pose | 82.7% | 80.2% | 80.8% | 80.5% |
| VideoMAE + Mobile + Notes | 87.1% | 86.5% | 86.2% | 86.3% |
| **Final Fusion Framework** | **89.8%** | **89.2%** | **88.9%** | **89.0%** |

### Ablation Study

| Configuration | Accuracy | Improvement |
|---------------|----------|-------------|
| VideoMAE Only | 78.25% | Baseline |
| + Pose Module | 82.7% | +4.45% |
| + Mobile Detector | 84.5% | +6.25% |
| + Notes Detector | 87.1% | +8.85% |
| Complete Framework | 89.8% | +11.55% |

### Class-wise Performance

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Normal | 92.4% | 91.8% | 92.1% |
| Copying | 88.6% | 87.9% | 88.2% |
| Mobile Usage | 91.3% | 90.7% | 91.0% |
| Gesture Sharing | 84.7% | 83.9% | 84.3% |
| Hidden Notes | 87.4% | 86.6% | 87.0% |

### Computational Efficiency

| Model | Parameters | FPS | Real-Time |
|-------|-----------|-----|-----------|
| R3D-18 | 33M | 38 | ✓ |
| VideoMAE | 86M | 16 | ✓ |
| YOLOv8 Mobile | 11M | 55 | ✓ |
| Fusion Framework | ~110M | 12–16 | ✓ |

**Hardware**: NVIDIA T4 GPU (Google Colab Pro)

---

## Model Development Journey

The system evolved through 4 stages:

### Stage 1: R3D-18 (CNN-based)
- **Performance**: 63% accuracy
- **Issue**: Weak long-range temporal modeling
- **Limitation**: Struggles with periodic cheating actions

### Stage 2: YOLO Pose + Rules
- **Performance**: 67.8% accuracy
- **Advantage**: Explainable behavioral rules
- **Limitation**: Only 2 classes, hand-crafted thresholds

### Stage 3: VideoMAE (Transformer)
- **Performance**: 78.25% accuracy
- **Advantage**: Strong temporal understanding via self-attention
- **Limitation**: Misses small objects (phones, notes)

### Stage 4: Multi-Modal Fusion
- **Performance**: 89.8% accuracy
- **Innovation**: Combines strengths of all previous approaches
- **Result**: Robust, explainable, real-world deployable system

---

## Key Contributions

1. **Custom Video-Based Dataset**: 5-class examination cheating dataset with 3,100 clips
2. **Student-Centric Preprocessing**: Automatic tracking and clip generation pipeline
3. **Model Comparison**: Benchmark R3D-18, YOLO Pose, and VideoMAE approaches
4. **Multi-Modal Architecture**: Fusion of action recognition, object detection, and pose reasoning
5. **Explainability**: Pose-based rules for interpretable decisions
6. **Real-World Deployment**: Framework suitable for practical examination monitoring
7. **Evidence Generation**: Annotated videos + CSV logs for post-exam analysis

---

## Technical Specifications

### Software Stack

| Component | Technology |
|-----------|-----------|
| **Deep Learning** | PyTorch, Transformers, TorchVision |
| **Object Detection** | Ultralytics YOLOv8 |
| **Pose Estimation** | YOLO Pose, MediaPipe |
| **Video Processing** | OpenCV |
| **Data Processing** | NumPy, Pandas |
| **Visualization** | Matplotlib, Seaborn |
| **Metrics** | Scikit-Learn |

### Hardware (Tested)

| Component | Specification |
|-----------|---------------|
| **GPU** | NVIDIA T4 (Google Colab Pro) |
| **RAM** | 15GB |
| **OS** | Ubuntu 22.04 |
| **CUDA** | 12.x |

---

## Project Team

### 👥 Team Members

| Name | Roll No | Role | Department |
|------|---------|------|-----------|
| **Rimsha Majeed** | BITF22M029 | Team Lead | Information Technology |
| **Alina Idrees** | BITF22M003 | Member | Information Technology |
| **Khadija Tul Kubra** | BITF22M025 | Member | Information Technology |

### 🎓 Supervisor

**Dr. Muhammad Farooq**  
Department of Information Technology  
Punjab University College of Information Technology (PUCIT)  
Email: [mfarooq@pucit.edu.pk](mailto:mfarooq@pucit.edu.pk)

### 🏛️ Institution

**Punjab University College of Information Technology (PUCIT)**  
University of the Punjab, Lahore, Pakistan

---

## Citation

If you use InvigilAI in your research, please cite:

```bibtex
@misc{invigilai2025,
  title={InvigilAI: AI-Based Real-Time Exam Cheating Detection System 
         for Physical Examination Halls Using VideoMAE, YOLO-Based Detectors 
         and Human Pose Reasoning},
  author={Idrees, Alina and Kubra, Khadija Tul and Majeed, Rimsha},
  year={2025},
  institution={Punjab University College of Information Technology (PUCIT)},
  type={Final Year Project},
  url={https://github.com/Rimsha002/InvigilAI-AI-Based-Exam-Cheating-Detection-System-for-Physical-Examination-Halls-}
}
```

---

## References

### Key Papers

- **VideoMAE**: Tong et al. (2022). "VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training." NeurIPS.
- **Vision Transformer**: Dosovitskiy et al. (2021). "An Image is Worth 16×16 Words." ICLR.
- **Attention Mechanism**: Vaswani et al. (2017). "Attention Is All You Need." NeurIPS.
- **R3D**: Tran et al. (2018). "A Closer Look at Spatiotemporal Convolutions for Action Recognition." CVPR.
- **YOLOv8**: Jocher et al. (2023). "Ultralytics YOLOv8."
- **Pose Estimation**: Bazarevsky et al. (2020). "BlazePose: On-device Real-time Body Pose Tracking." CVPR Workshops.

---

## Future Work

### Short-Term Improvements
- Hyperparameter optimization for higher accuracy
- Fine-grained gesture recognition via MediaPipe Hands
- Robustness against blur and occlusion via super-resolution
- Improved tracking stability for high-movement students

### Medium-Term Enhancements
- Multi-camera fusion for large examination halls
- Advanced fusion strategies (transformer-based, graph neural networks)
- Gaze estimation and eye tracking
- Edge AI optimization for real-time deployment on edge devices

### Long-Term Vision
- Audio-visual cheating detection (speech, whispers)
- LLM-based reasoning for automated report generation
- Larger public benchmark dataset
- Fully intelligent examination surveillance platform

---

## License

This project is released under the **MIT License**.

**Copyright (c) 2025 InvigilAI Research Team**  
Punjab University College of Information Technology (PUCIT)

---

## Acknowledgments

We thank:
- All 65 participants who contributed to data collection
- Dr. Muhammad Farooq for research supervision and guidance
- PUCIT for institutional support
- Google Colab for GPU access
- Open-source communities behind PyTorch, YOLOv8, and MediaPipe

---

## Contact

For questions or collaboration inquiries, please contact:

📧 **Email**: mfarooq@pucit.edu.pk  
🔗 **GitHub**: [Rimsha002](https://github.com/Rimsha002)

---

**Last Updated**: June 2025  
**Status**: Final Release
