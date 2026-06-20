# InvigilAI: AI-Based Exam Cheating Detection System for Physical Examination Halls


**A Research-based Project**

---

## Abstract

This paper presents **InvigilAI**, an intelligent multi-modal fusion framework designed for automatic detection and classification of multiple cheating behaviors from examination surveillance videos. Academic integrity in traditional examination environments faces significant challenges due to the limitations of human invigilation. Existing AI-based solutions suffer from binary classification constraints, image-based dependencies, and poor generalization to real examination scenarios. 

To address these limitations, we propose a comprehensive framework that integrates:
- **VideoMAE**: Transformer-based temporal action recognition
- **YOLO-Mobile**: Custom mobile phone detector
- **YOLO-Notes**: Custom hidden notes detector  
- **YOLO Pose**: Explainable pose reasoning engine
- **Weighted fusion mechanism**: Multi-modal decision aggregation

The system achieves **89.8% overall accuracy** on a custom 5-class video dataset containing 3,100 clips across diverse examination scenarios. This represents a **26.8 percentage point improvement** over the baseline R3D-18 CNN approach (63.0%).

---

## 1. Introduction

### 1.1 Problem Statement

Offline examination cheating remains a persistent challenge in educational institutions worldwide. Traditional invigilation depends heavily on human supervision, which is susceptible to fatigue, distraction, and cognitive limitations (Jahangir et al., 2024). While humans can monitor 20-30 students simultaneously with reasonable effectiveness, large-scale examinations involving 100+ students present significant practical constraints.

**Key limitations of existing AI-based solutions:**
- Binary classification (cheating vs. normal) rather than multi-class behavior recognition
- Focus on single cheating behaviors (e.g., mobile phones only)
- Dependence on static image-based datasets
- Insufficient temporal understanding
- Heavy reliance on manually designed rules with poor generalization
- Limited explainability regarding detection decisions
- Absence of publicly available video-based datasets with diverse cheating classes

### 1.2 Motivation for Video-Based Multi-Class Approach

Cheating is fundamentally a **sequence of actions rather than a static image**. Critical behaviors include:
- **Copying**: Side glances, head turning toward neighbors
- **Mobile Phone Usage**: Hidden phone interaction, downward gaze coordination
- **Gesture Sharing**: Hand signals, finger communication
- **Hidden Notes Consultation**: Paper material exchange, concealed documents
- **Normal Behavior**: Legitimate writing, reading question papers

Individual frames cannot reliably distinguish these temporal patterns. For example, a student's downward glance for 0.5 seconds is normal writing behavior, but a 5-second focused gaze combined with pen inactivity indicates potential mobile phone usage or note consultation.

### 1.3 Contributions

1. **Novel 5-class video-based examination cheating dataset** with 3,100 annotated clips from 65 diverse participants
2. **Student-centric preprocessing pipeline** with automatic tracking and clip extraction
3. **Comprehensive comparative analysis** of CNN-based, rule-based, and transformer-based approaches
4. **Multi-modal fusion architecture** combining action recognition, object detection, and pose reasoning
5. **Explainable reasoning module** with pose-based behavioral analysis
6. **Deployment-ready system** with real-time inference capability (~12 FPS on GPU)

---

## 2. Related Work and Research Gaps

### 2.1 Existing Datasets
| Dataset | Type | Classes | Scale | Real-World |
|---------|------|---------|-------|-----------|
| Kinetics-400 | Video | 400 (generic actions) | 306K videos | Limited |
| UCF101 | Video | 101 (generic actions) | 13K videos | Limited |
| HMDB51 | Video | 51 (generic actions) | 7K videos | Limited |
| **InvigilAI** | **Video** | **5 (cheating classes)** | **3.1K clips** | **✓ High** |

### 2.2 Research Gaps Addressed
1. **Lack of realistic examination video datasets**: Most existing datasets focus on generic actions (sports, daily activities) rather than cheating-specific behaviors in examination environments
2. **Binary vs. multi-class limitation**: Previous works (Jahangir et al., 2024) achieve only binary classification
3. **Single-modality constraints**: Existing systems use either CNNs or pose-based rules independently
4. **Small object detection problem**: Mobile phones and notes occupy minimal image regions, challenging standard CNN approaches
5. **Explainability-performance tradeoff**: Rule-based systems are interpretable but inflexible; deep learning models are powerful but opaque

---

## 3. Dataset Description and Construction

### 3.1 Dataset Specifications

**Kaggle Link**: [InvigilAI Examination Cheating Dataset]( https://www.kaggle.com/datasets/rimmajeed/examcheating-multiv-video-based-dataset)

| Property | Value |
|----------|-------|
| **Total Clips** | 3,100 || **Total Raw Videos** | 637 MP4/MOV files |
| **Participants** | ~65 (diverse demographics) |
| **Recording Environments** | 5 versions (V1-V5) + final theater-style setup |
| **Frame Rate** | 24-30 FPS |
| **Resolution** | 480p, 720p, 1080p (mixed) |
| **Duration per Clip** | 16 frames (0.5-0.7 seconds) |
| **Annotation Method** | Manual + quality assurance |

### 3.2 Class Distribution

| Class | Clip Count | Percentage | Examples |
|-------|-----------|-----------|----------|
| **Normal** | 964 | 31.1% | Writing, reading question paper, forward gaze |
| **Copying** | 807 | 26.0% | Side glances, head turning, object exchange |
| **Mobile Phone Usage** | 569 | 18.4% | Hidden phone interaction, downward focus |
| **Gesture Sharing** | 322 | 10.4% | Hand signals, finger communication |
| **Hidden Notes** | 438 | 14.1% | Paper consultation, concealed materials |

### 3.3 Dataset Evolution Through Iterative Versions

#### Version 1 (V1): Controlled Single-Student
- **Purpose**: Proof-of-concept feasibility
- **Characteristics**: Single student, uniform lighting, minimal occlusion
- **Limitation**: Highly artificial; no student interaction

#### Version 2 (V2): Multi-Angle Recording
- **Purpose**: Viewpoint invariance
- **Characteristics**: Front, side, oblique angles; varying lighting; multiple resolutions
- **Challenge**: Motion blur and lighting variations introduced

#### Version 3 (V3): Multi-Student Theater Environment
- **Purpose**: Student interaction capture
- **Characteristics**: Multiple simultaneous students, complex backgrounds, partial occlusions
- **Challenge**: Overlapping students and small object visibility

#### Version 4 (V4): Classroom Environment
- **Purpose**: Temporal diversity and realistic patterns
- **Characteristics**: 20-30 students, natural behavior, long-duration recordings
- **Challenge**: Perspective distortion and heavy occlusion

#### Version 5 (Final): Theater-Style Examination Hall
- **Purpose**: Closest approximation to real examination conditions
- **Characteristics**: Theater seating arrangement, diverse viewpoints, varied distances, realistic inter-class similarity
- **Outcome**: Foundation dataset for the proposed framework

### 3.4 Participant Demographics

To ensure generalization and fairness:
- **Gender**: 52% male, 48% female
- **Physical diversity**: Heights 5'2" to 6'3"; various body structures and sitting postures
- **Behavioral diversity**: 45% left-handed, 55% right-handed; different writing styles
- **Appearance diversity**: Various clothing, accessories, and hairstyles

### 3.5 Recording Conditions

| Factor | Variation |
|--------|-----------|
| **Lighting** | Bright (≥500 lux), Medium (200-500 lux), Dim (≤200 lux) |
| **Camera Angles** | Front, side, oblique, diagonal, long-distance |
| **Resolutions** | 480p, 720p, 1080p |
| **Environmental Artifacts** | Motion blur, compression artifacts, occlusions, scale variations |

### 3.6 Data Preprocessing Pipeline

```
Raw Classroom Video (637 total)
        ↓
Student Detection (YOLO person class)
        ↓
Multi-Object Tracking (ByteTrack)
        ↓
Student ID Assignment (unique per student)
        ↓
Individual Student Cropping (128×128 → 224×224)
        ↓
16-Frame Clip Generation (overlap=8 frames)
        ↓
Manual Verification (quality check)
        ↓
Ambiguous Clip Removal (uncertain samples)
        ↓
CSV Annotation (label, timestamp, student_id)
        ↓
Final Dataset (3,100 clips)
```

**Rationale**: Student-centric cropping reduces irrelevant background noise and focuses model attention exclusively on behavior-relevant regions.

### 3.7 Quality Assurance Mechanisms

1. **Manual Cross-Checking**: Each clip reviewed by 2+ annotators
2. **Ambiguous Clip Removal**: 15% of generated clips discarded due to label uncertainty
3. **Duplicate Removal**: Redundant clips filtered using frame hashing
4. **Inter-Annotator Agreement**: Cohen's kappa ≥ 0.87 across all classes
5. **Noise Inspection**: Severely blurred or compressed clips excluded

### 3.8 Dataset Challenges and Characteristics

| Challenge | Description | Impact |
|-----------|-------------|--------|
| **Occlusion** | Hands hide phones/notes/faces | Low phone detection recall |
| **Motion Blur** | Rapid movements introduce blur | Temporal feature degradation |
| **Inter-Class Similarity** | Writing ≈ note usage; glance ≈ normal look | Confusion matrix asymmetry |
| **Small Object Detection** | Phones occupy 1-3% of image pixels | CNN receptive field inadequacy |
| **Scale Variation** | Student size varies by seating position | Inconsistent detector responses |
| **Perspective Variation** | Different viewing angles alter appearance | Pose estimation variance |
| **Long-Term Dependencies** | Periodic cheating patterns (Write→Look→Write) | Insufficient temporal context |
| **Multi-Person Overlap** | Students partially block each other | Tracker ID switches |

---

## 4. Methodology: Model Development Journey

### 4.1 Progression and Rationale

```
Baseline 1: R3D-18 (63.0% accuracy)
    ↓ [Limitation: Weak temporal understanding]
Baseline 2: YOLO Pose + Rules (69.4% accuracy)
    ↓ [Limitation: Only binary classification, manual thresholds]
Baseline 3: VideoMAE Transformer (78.25% accuracy)
    ↓ [Limitation: Misses small objects like phones/notes]
Final: Multi-Modal Fusion (89.8% accuracy)
    ✓ [Solution: Combines all modalities]
```

Each progression was motivated by specific limitations identified in prior stages.

---

## 5. Baseline Models: Analysis and Results

### 5.1 Baseline 1: R3D-18 (3D Residual Network-18)

#### Architecture Overview
```
Input: 16 frames × 224×224×3
    ↓
Conv3D (64 filters, 3×3×3 kernel)
    ↓
Residual Blocks (×4 with skip connections)
    ↓
Global Average Pooling
    ↓
Fully Connected (1024 → 512 → 5 classes)
    ↓
Softmax Output
```

#### Key Characteristics
- **3D Convolutions**: Process temporal + spatial dimensions jointly
- **Skip Connections**: Enable deep network training (18 layers)
- **Pretrained Weights**: Available from Kinetics-400 dataset
- **Computational Efficiency**: ~33M parameters

#### Hyperparameters
| Parameter | Value |
|-----------|-------|
| Batch Size | 8 |
| Epochs | 30 |
| Learning Rate | 0.001 (Adam optimizer) |
| Frame Input | 16 frames at 112×112 |
| Weight Decay | 5e-4 |

#### Performance Results

| Metric | Value |
|--------|-------|
| **Training Accuracy** | 84.3% |
| **Validation Accuracy** | 63.0% |
| **Precision** | 62.4% |
| **Recall** | 61.8% |
| **F1-Score** | 62.1% |
| **Inference Speed** | 38 FPS (T4 GPU) |

#### Class-Wise Performance
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Normal | 68.2% | 71.3% | 69.7% |
| Copying | 59.4% | 58.1% | 58.7% |
| Mobile Phone | 61.1% | 59.8% | 60.4% |
| Gesture Sharing | 58.3% | 55.2% | 56.7% |
| Hidden Notes | 60.7% | 62.4% | 61.5% |

#### Limitations Identified

1. **Local Receptive Fields**: 3D convolutions observe only neighboring spatial regions, limiting long-range temporal dependency capture
2. **Weak Periodic Pattern Recognition**: Cannot model behaviors like "Write→Look→Write→Look" that repeat over longer sequences
3. **Small Object Sensitivity**: Phones and notes occupy 1-3% of image pixels; lost after multiple pooling layers
4. **Inter-Class Confusion**: Similar appearances (writing vs. note usage) create high confusion
5. **Significant Overfitting**: 84.3% → 63.0% accuracy gap indicates poor generalization

#### Why R3D-18 Saturated at 63%

Cheating behaviors are characterized by:
- **Long-range temporal dependencies** (look pattern emerges over seconds)
- **Subtle motion cues** (small head turns, finger signals)
- **Small object interactions** (hidden phones, notes)

3D CNNs fundamentally struggle because:
- Local convolution kernels cannot model patterns >10-15 frames apart
- Max pooling destroys fine-grained details needed for small object detection
- Feature maps become spatially compressed, losing object localization capability

---

### 5.2 Baseline 2: YOLO Pose + Rule-Based System

#### Motivation

Copying behavior involves observable body posture changes. Human pose estimation provides an **interpretable alternative** to opaque deep learning.

#### Architecture

```
Input Frame
    ↓
YOLOv8n-Pose (17 keypoint extraction)
    ↓
Pose Feature Computation:
├─ Side Glance: |nose_x - face_center_x|
├─ Leaning: |LS_y - RS_y|
├─ Reaching: ||wrist - torso||
└─ Head Turning: Δnose_x
    ↓
Rule-Based Thresholds
    ↓
Temporal Score Accumulation: Score_t = 0.9·Score_{t-1} + CurrentBehavior
    ↓
Binary Output: {Normal, Copying}
```

#### Keypoints Used (17 total)
Nose, Eyes, Ears, Shoulders, Elbows, Wrists, Hips, Knees, Ankles

#### Rule Definitions

| Rule | Computation | Threshold | Interpretation |
|------|-----------|-----------|-----------------|
| **Side Glance** | \|nose_x - face_center_x\| | > 15px | Head turned toward neighbor |
| **Leaning** | \|LS_y - RS_y\| | > 20px | Body asymmetry |
| **Reaching** | \|\|wrist - torso\|\| | > 60px | Arm extended beyond body |
| **Head Turning** | Δnose_x | > 10px/frame | Rapid head movement |

#### Temporal Scoring

Each frame contributes to a cumulative behavior score:
$$Score_t = 0.9 \cdot Score_{t-1} + \begin{cases} 1 & \text{if rule triggered} \\ 0 & \text{otherwise} \end{cases}$$

Maximum score = 10.0; triggers detection at score > 5.0

#### Performance Results

| Metric | Value |
|--------|-------|
| **Accuracy** | 69.4% |
| **Precision** | 67.2% |
| **Recall** | 68.8% |
| **F1-Score** | 68.0% |
| **Inference Speed** | 48 FPS (CPU capable) |

#### Critical Limitations

1. **Binary Classification Only**: Cannot distinguish copying, mobile usage, notes, or gestures—only "Normal" vs. "Copying"
2. **Manual Threshold Design**: Thresholds (15px, 20px, 60px, etc.) tuned on training set; fail with viewpoint changes
3. **No Temporal Semantics**: Rules operate on individual frame features; cannot understand action sequences
4. **Cannot Detect**: Mobile phones, hidden notes, or gesture communication
5. **Poor Generalization**: 30% accuracy drop when camera angle changes by >15°

#### Why Rule-Based Approach Failed for Multi-Class

Pose-based reasoning alone cannot distinguish:
- **Writing vs. Note Consultation**: Both involve downward gaze and hand occlusion
- **Gesture Sharing vs. Normal Fidgeting**: Similar hand positions and trajectories
- **Mobile Usage vs. Side Glance**: Both correlate with head turning

---

### 5.3 Baseline 3: VideoMAE (Transformer-Based Temporal Modeling)

#### Motivation

Transformers overcome CNN limitations through **self-attention mechanisms**, enabling models to relate distant frames directly without intermediate layers.

#### Architecture Overview

```
Input: 16 frames × 224×224×3
    ↓
Patch Embedding: 196 patches (14×14, 16×16 patch size)
    ↓
Patch Masking: 75% patches randomly masked
    ↓
Transformer Encoder (12 layers, 8 attention heads):
├─ Multi-Head Attention (global context)
├─ LayerNorm
├─ Feed-Forward Network (2048 hidden)
└─ Residual Connections
    ↓
Transformer Decoder (lightweight reconstruction)
    ↓
Patch Reconstruction Loss
    ↓
(During fine-tuning) Classification Head
    ↓
Softmax (5 classes)
```

#### Key Technical Details

| Component | Specification |
|-----------|---------------|
| **Model** | MCG-NJU/videomae-base-finetuned-kinetics |
| **Pretraining** | Kinetics-400 (self-supervised masked autoencoding) |
| **Patch Size** | 16×16 pixels |
| **Temporal Patches** | 16 frames |
| **Total Tokens** | 196 spatial × 1 temporal = 196 tokens |
| **Parameters** | 86M |
| **Attention Heads** | 8 |
| **Layers** | 12 |

#### Self-Attention Mechanism

$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$$

Each patch attends to all other patches globally, enabling:
- **Spatial Attention**: Phone ↔ Hand, Face ↔ Eyes relationships
- **Temporal Attention**: Frame t ↔ Frame t+5 dependencies (captures periodic patterns)

#### Hyperparameters for Fine-Tuning

| Parameter | Value |
|-----------|-------|
| Batch Size | 4 (GPU memory constraints) |
| Epochs | 20 |
| Learning Rate | 1e-5 (AdamW) |
| Weight Decay | 0.05 |
| Warmup Steps | 10% of total |
| Dropout | 0.2 |
| Frame Input | 16 at 224×224 |

#### Performance Results

| Metric | Value |
|--------|-------|
| **Training Accuracy** | 99.0% |
| **Validation Accuracy** | 78.25% |
| **Precision** | 77.9% |
| **Recall** | 77.3% |
| **F1-Score** | 77.6% |
| **Inference Speed** | 16 FPS (T4 GPU) |

#### Class-Wise Performance
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Normal | 84.2% | 83.1% | 83.6% |
| Copying | 76.8% | 75.9% | 76.3% |
| Mobile Phone | 78.3% | 77.1% | 77.7% |
| Gesture Sharing | 71.4% | 70.2% | 70.8% |
| Hidden Notes | 76.5% | 75.8% | 76.1% |

#### Why VideoMAE Significantly Outperforms R3D-18

| Capability | R3D-18 | VideoMAE |
|-----------|--------|----------|
| Long-range temporal modeling | Weak (local convolution) | **Strong (self-attention)** |
| Global spatial context | Local (neighborhoods) | **Global (all patches)** |
| Self-supervised pretraining | ✗ | **✓ (Kinetics-400)** |
| Handling periodic patterns | Poor | **Excellent** |
| Transfer learning benefit | Moderate | **High** |
| **Overall Accuracy** | **63.0%** | **78.25% (+15.25 pp)** |

#### Remaining Limitations

Despite significant improvement, VideoMAE exhibits critical weaknesses:

1. **Hidden Mobile Phone Detection**: Phones may be completely occluded; VideoMAE classifies as "Normal" (84% of false negatives)
2. **Hidden Notes Recognition**: Small paper objects challenging; 18% false negative rate
3. **Gesture Sharing Confusion**: Hand movements resemble normal writing; 28.6% false negatives
4. **Explainability Gap**: Attention maps show which patches are important but not "why this decision was made"
5. **Overfitting Signal**: 99.0% → 78.25% accuracy gap (20.75 pp) indicates room for architectural improvements

#### Analysis: Why VideoMAE Alone Insufficient

VideoMAE is fundamentally a **sequence classification model**:
- Outputs: P(class | sequence)
- Does NOT provide: "Where is the cheating object?" or "What body parts matter?"

For high-confidence mobile detection with occlusion, the model needs:
- Direct object localization (not just class probability)
- Pose-based behavioral reasoning
- Temporal decision stabilization

---

## 6. Proposed: Multi-Modal Fusion Framework

### 6.1 System Architecture

```
Input Examination Video
        ↓
────────────────────────────────────
Student Detection & Multi-Object Tracking
├─ YOLO v8 Person Detection
└─ BoT-SORT Tracking (assigns unique IDs)
────────────────────────────────────
        ↓
Student-Centric Clip Extraction
(Each student independently processed)
        ↓
═══════ PARALLEL PROCESSING ═════════
│                                   │
│  Module 1: VideoMAE              │
│  (Temporal Action Recognition)    │
│  → P_VideoMAE (5 classes)         │
│                                   │
│  Module 2: YOLO-Mobile           │
│  (Phone Localization)             │
│  → Confidence_Mobile              │
│                                   │
│  Module 3: YOLO-Notes            │
│  (Notes Localization)             │
│  → Confidence_Notes               │
│                                   │
│  Module 4: YOLO Pose             │
│  (Behavioral Reasoning)           │
│  → Pose_Score                     │
│                                   │
═══════════════════════════════════
        ↓
Weighted Fusion Layer
P_fusion = w₁·P_V + w₂·P_M + w₃·P_N + w₄·P_P
        ↓
Temporal Smoothing (Majority voting over 10-frame buffer)
        ↓
Final Decision & Confidence Score
        ↓
Annotated Video + CSV Event Logs
```

### 6.2 Module Specifications

#### Module 1: VideoMAE Action Recognition

| Parameter | Value |
|-----------|-------|
| **Role** | Primary temporal classifier |
| **Input** | 16 frames (224×224×3) |
| **Output** | P_VideoMAE ∈ ℝ^5 (softmax probabilities) |
| **Classes** | Normal, Copying, Mobile, Gesture, Notes |
| **Contribution Weight** | w₁ = 0.50 |

**Rationale**: VideoMAE captures long-range temporal dependencies, primary cheating indicator

#### Module 2: YOLO-Mobile Detector

| Parameter | Value |
|-----------|-------|
| **Role** | Phone-specific evidence |
| **Model** | YOLOv8m (custom trained) |
| **Input** | Single frame (640×640) |
| **Output** | Bounding box + confidence |
| **Training Data** | 2000+ phone images (various occlusion levels) |
| **mAP@50** | 0.91 |
| **Contribution Weight** | w₂ = 0.20 |

**Priority Rule**: If Confidence_Mobile > 0.85 AND gaze directed toward phone → Immediately elevate Mobile classification

#### Module 3: YOLO-Notes Detector

| Parameter | Value |
|-----------|-------|
| **Role** | Paper material localization |
| **Model** | YOLOv8m (custom trained) |
| **Input** | Single frame (640×640) |
| **Output** | Bounding box + confidence |
| **Training Data** | 1800+ notes/paper images |
| **mAP@50** | 0.88 |
| **Contribution Weight** | w₃ = 0.15 |

**Priority Rule**: If Confidence_Notes > 0.80 → Notes classification dominates

#### Module 4: YOLO Pose Reasoning

| Parameter | Value |
|-----------|-------|
| **Role** | Behavioral pattern analysis |
| **Model** | YOLOv8n-pose |
| **Keypoints** | 17 body landmarks |
| **Features Computed** | Side glance, leaning, reaching, head turn, gaze direction |
| **Temporal Window** | 20-frame buffer with exponential weighting |
| **Score Accumulation** | Score_t = 0.9·Score_{t-1} + CurrentBehavior |
| **Contribution Weight** | w₄ = 0.15 |

**Pose Features Detailed**:
| Feature | Computation | Indicates |
|---------|-----------|-----------|
| **Side Glance** | \|nose_x - image_center_x\| | Copying attempt |
| **Leaning** | \|LS_y - RS_y\| (shoulder asymmetry) | Reaching for neighbor's paper |
| **Reaching** | \|\|wrist - torso\|\| | Suspicious object exchange |
| **Head Turning** | Δnose_x/Δt (rate of change) | Quick glance detection |
| **Gaze Direction** | angle(nose → eyes, direction_vector) | Where student is looking |

### 6.3 Fusion Mechanism

#### Score-Level Fusion

$$P_{fusion} = w_1 \cdot P_{VideoMAE} + w_2 \cdot P_{Mobile} + w_3 \cdot P_{Notes} + w_4 \cdot P_{Pose}$$

Where:
- $w_1 = 0.50$ (VideoMAE dominates due to proven temporal strength)
- $w_2 = 0.20$ (Mobile detector provides strong evidence for specific class)
- $w_3 = 0.15$ (Notes detector less frequent than mobile cases)
- $w_4 = 0.15$ (Pose reasoning provides behavioral context)
- $\sum w_i = 1.0$

#### Class Assignment

$$\hat{y} = \arg\max(P_{fusion})$$

#### Confidence Estimation

$$\text{Confidence} = \max(P_{fusion})$$

#### Decision Priority Rules

```python
# Priority 1: Mobile phone detected with high confidence
if P_Mobile > 0.85 and gaze_toward_phone:
    confidence *= 1.2  # Amplify weight
    class = "Mobile_Usage"

# Priority 2: Notes detected with high confidence
if P_Notes > 0.80:
    confidence *= 1.15
    class = "Hidden_Notes"

# Priority 3: Standard fusion
else:
    class = argmax(P_fusion)
```

### 6.4 Temporal Smoothing

Frame-level predictions fluctuate due to noise and motion blur. Temporal smoothing stabilizes predictions:

```python
prediction_history = deque(maxlen=10)  # 10-frame buffer
prediction_history.append(class_t)
stable_prediction = mode(prediction_history)  # Majority voting
```

**Effect**: Removes false positives from single anomalous frames; ensures predictions are stable over ≥ 5 consecutive frames (≥0.17 seconds)

### 6.5 Event Logging

For each detected cheating instance:

```csv
Timestamp,Student_ID,Frame_Number,VideoMAE_Label,VideoMAE_Conf,Mobile_Conf,Notes_Conf,Pose_Score,Final_Label,Final_Confidence
00:01:23,5,560,Copying,0.81,0.00,0.00,0.73,Copying,0.88
00:02:07,9,1204,Mobile_Usage,0.76,0.94,0.05,0.42,Mobile_Usage,0.92
```

---

## 7. Experimental Results and Comparative Analysis

### 7.1 Overall Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | Inference Speed |
|-------|----------|-----------|--------|----------|-----------------|
| **R3D-18** | 63.0% | 62.4% | 61.8% | 62.1% | 38 FPS |
| **YOLO Pose + Rules** | 69.4% | 67.2% | 68.8% | 68.0% | 48 FPS |
| **VideoMAE** | 78.25% | 77.9% | 77.3% | 77.6% | 16 FPS |
| **VideoMAE + Pose** | 82.7% | 81.8% | 81.2% | 81.5% | 14 FPS |
| **VideoMAE + Mobile** | 84.5% | 83.9% | 83.1% | 83.5% | 13 FPS |
| **VideoMAE + Notes** | 83.8% | 83.2% | 82.4% | 82.8% | 13 FPS |
| **VideoMAE + Mobile + Notes** | 87.1% | 86.5% | 85.9% | 86.2% | 13 FPS |
| **Full Fusion Framework** | **89.8%** | **89.2%** | **88.9%** | **89.0%** | **12 FPS** |

**Key Observation**: +26.8 pp improvement over R3D-18 baseline; +11.5 pp improvement over VideoMAE alone

### 7.2 Ablation Study: Module Contribution Analysis

| Configuration | Accuracy | Δ vs Baseline |
|---------------|----------|--------------|
| VideoMAE only | 78.25% | — |
| + Pose Module | 82.7% | +4.45 pp |
| + Mobile Detector | 84.5% | +6.25 pp |
| + Notes Detector | 83.8% | +5.55 pp |
| + Mobile + Notes | 87.1% | +8.85 pp |
| + Mobile + Notes + Pose | 89.8% | +11.53 pp |

**Insights**:
- Pose contributes incremental gains (+4.45 pp) through behavioral pattern recognition
- Mobile detector provides largest individual contribution (+6.25 pp) due to high class imbalance
- Synergistic effect: Full combination > sum of individual improvements (11.53 pp > 4.45+6.25+5.55 = 16.25 pp suggests non-additive interactions)

### 7.3 Class-Wise Performance

| Class | Precision | Recall | F1-Score | False Positive Rate |
|-------|-----------|--------|----------|-------------------|
| **Normal** | 92.4% | 91.8% | 92.1% | 7.6% |
| **Copying** | 88.6% | 87.9% | 88.2% | 11.4% |
| **Mobile Usage** | 91.3% | 90.7% | 91.0% | 8.7% |
| **Gesture Sharing** | 84.7% | 83.9% | 84.3% | **15.3%** |
| **Hidden Notes** | 87.4% | 86.6% | 87.0% | 12.6% |

**Notable**: Gesture Sharing remains most challenging class (15.3% false positive rate). Primary cause: Hand movements during normal writing resemble cheating gestures; requires additional context cues.

### 7.4 Confusion Matrix Analysis

**Most Common Confusions**:

| Source | Misclassified As | Frequency | Reason |
|--------|------------------|-----------|--------|
| **Gesture** | Normal | 8.2% | Similar hand trajectories |
| **Notes** | Copying | 5.4% | Downward gaze resembles side-looking |
| **Mobile** | Notes | 3.1% | Both involve occlusion and gaze downward |
| **Copying** | Normal | 2.1% | Subtle head turns not always detected |

**Resolution**: Multi-modal fusion reduces "Notes ↔ Mobile" confusion through dedicated detectors; "Gesture ↔ Normal" remains challenging due to visual similarity.

### 7.5 Performance Under Challenging Conditions

| Condition | Accuracy Drop | Primary Factor |
|-----------|---------------|-----------------|
| **Motion Blur** | -3.2% | Temporal feature degradation |
| **Occlusion (hands)** | -4.8% | Pose keypoint loss |
| **Lighting Change** | -2.1% | Minimal impact (transformer robust) |
| **Camera Angle Change (±15°)** | -5.6% | Small object detector sensitivity |
| **Student Far from Camera** | -6.3% | Resolution reduction + scale variation |

---

## 8. Research Contributions Summary

### 8.1 Novelty and Significance

1. **First comprehensive multi-class examination cheating dataset**: 5 classes in realistic video settings (vs. prior binary image-based approaches)
2. **Comparison of CNN, rule-based, and transformer approaches**: Systematic analysis of architectural tradeoffs
3. **Multi-modal fusion without explicit knowledge transfer**: Novel integration of action recognition, object detection, and pose reasoning
4. **Explainability through pose reasoning**: Provides behavioral justification alongside deep learning predictions
5. **Practical deployment framework**: Real-time inference (~12 FPS) on affordable GPUs

### 8.2 Key Findings

- **Transformers substantially outperform CNNs for temporal modeling** (78.25% vs 63.0%): Self-attention enables long-range dependency capture essential for periodic cheating patterns
- **Multi-modality is essential for robust detection**: Single-modality approaches miss critical evidence (small objects, behavioral cues)
- **Explainability-performance tradeoff is not necessary**: Fusion framework achieves state-of-the-art accuracy while maintaining pose-based interpretability
- **Real-time deployment feasible**: 12 FPS sufficient for surveillance system integration

---

## 9. Limitations and Future Work

### 9.1 Current Limitations

1. **Severe Occlusion Handling**: Phones completely hidden inside clothing remain undetected
2. **Extreme Camera Angles**: Side views (>45° from frontal) reduce pose accuracy
3. **Small Dataset Size**: 3,100 clips may not capture all cheating variations
4. **Single Camera**: No multi-view occlusion resolution
5. **Gesture-Normal Ambiguity**: Hand movement similarity creates ~15% class confusion
6. **Computational Cost**: Requires GPU for real-time inference (though 12 FPS achievable on consumer GPUs)

### 9.2 Research Gaps and Future Directions

| Gap | Proposed Solution | Expected Impact |
|-----|------------------|-----------------|
| **Small object robustness** | Anchor-based attention mechanisms; feature pyramid networks | +3-5% accuracy |
| **Gesture disambiguation** | MediaPipe Hands with finger-level reasoning | +5-8% accuracy |
| **Multi-camera integration** | 3D pose estimation + cross-view tracking | +4-6% accuracy |
| **Gaze estimation** | Eye landmark tracking + gaze direction inference | +3-4% accuracy |
| **Audio-visual fusion** | Speech activity + whisper detection | +2-3% accuracy |
| **Edge deployment** | Model quantization (INT8) + distillation | Reduced latency, slightly lower accuracy |
| **Automated thresholding** | Meta-learning for weight optimization | +1-2% accuracy |
| **Larger datasets** | Multi-institutional data collection | +5-10% accuracy |

### 9.3 Recommended Dataset Extensions

1. **Additional Environments**: 10+ distinct examination halls
2. **More Participants**: 500+ for demographic representativeness
3. **Rare Events**: Explicit collection of challenging cases (severe occlusion, unusual gestures)
4. **Multi-Angle Recording**: Synchronized multi-camera views
5. **Long-Duration Videos**: Extended examination sessions (2+ hours) to capture rare behaviors

---

## 10. Software Implementation

### 10.1 Requirements

```
Python 3.10+
PyTorch 2.0+
torchvision
transformers (HuggingFace)
Ultralytics YOLOv8
OpenCV
MediaPipe
NumPy, Pandas, scikit-learn
```

### 10.2 Deployment Specifications

| Component | Configuration |
|-----------|----------------|
| **GPU** | NVIDIA T4 (8GB) or RTX 3060 (12GB) minimum |
| **CPU Fallback** | Supported (inference ~2 FPS) |
| **Memory** | 16GB RAM recommended |
| **Storage** | 20GB for models + data |

### 10.3 Inference Pipeline

```python
# Pseudocode
for each_video_frame:
    # Detection & Tracking
    detections = yolo_detector(frame)
    tracked_students = tracker(detections)
    
    # Parallel Module Processing
    videomae_probs = videomae_model(frame_buffer)
    mobile_conf = yolo_mobile(frame)
    notes_conf = yolo_notes(frame)
    pose_score = compute_pose_features(frame)
    
    # Fusion
    fusion_probs = weighted_fusion(
        videomae_probs, mobile_conf, notes_conf, pose_score
    )
    
    # Temporal Smoothing
    stable_pred = temporal_smooth(fusion_probs)
    
    # Logging
    log_event(timestamp, student_id, stable_pred)
    draw_annotated_frame(frame, stable_pred)
```

---

## 11. Conclusion

This paper presented **InvigilAI**, a comprehensive multi-modal fusion framework for intelligent examination cheating detection. Through systematic analysis of CNN-based, rule-based, and transformer-based approaches, we demonstrated that:

1. **Transformer architectures substantially improve temporal modeling** for examination video analysis compared to 3D CNNs (78.25% vs 63.0%)
2. **Multi-modal fusion is essential** for robust detection of diverse cheating behaviors, achieving 89.8% accuracy vs. 78.25% for single-modality approaches
3. **Explainability and performance need not be sacrificed**: Pose-based reasoning provides interpretable decision justification alongside state-of-the-art accuracy
4. **Real-time deployment is feasible** on affordable consumer GPUs (~12 FPS)

The constructed dataset (3,100 annotated clips, 65 participants, 5 cheating classes) addresses a critical research gap by providing the first realistic multi-class examination cheating dataset. Our framework successfully integrates VideoMAE temporal reasoning, YOLO-based object detection, and pose-based behavioral analysis into a deployable system suitable for large-scale examination monitoring.

**Future work** should focus on larger multi-institutional datasets, multi-camera integration for occlusion handling, and specialized architectures for gesture disambiguation and small-object detection.

---

## References

Jahangir, M. T., Subhani, N., Nadeem, S., & Abid, F. (2024). AI-Powered Classification for Cheating Detection in Offline Examinations Using Deep Learning Techniques with CUI Dataset. *International Journal of Innovations in Science and Technology*, 6(4), 1658-1678.

Tong, Z., Song, Y., Wang, J., et al. (2022). VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training. *NeurIPS 2022*.

Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al. (2021). An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale. *ICLR 2021*.

Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention Is All You Need. *NeurIPS 2017*.

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR 2016*.

Jocher, G., et al. (2023). Ultralytics YOLOv8. [Software]. Retrieved from https://github.com/ultralytics/ultralytics

Tran, D., Wang, H., Torresani, L., et al. (2018). A Closer Look at Spatiotemporal Convolutions for Action Recognition. *CVPR 2018*.

Bazarevsky, V., Girdhar, R., Grangier, D., et al. (2020). BlazePose: On-device Real-time Body Pose Tracking. *CVPR Workshops 2020*.

---

## Contact and Citation

**Authors**: Rimsha Majeed (rimshamajeed2002@gmail.com)

**Team Mate**: Alina Idress, Khadija tul Kubra 

**Affiliation**: Punjab University College of Information Technology (PUCIT), University of the Punjab, Lahore, Pakistan

**Supervisor**: Dr. Muhammad Farooq (mfarooq@pucit.edu.pk)

**Dataset**: Available at [Kaggle: InvigilAI Exam Cheating Detection](https://www.kaggle.com/datasets/rimmajeed/examcheating-multiv-video-based-dataset)


### Citation

```bibtex
@misc{invigilai2025,
  title={InvigilAI: Multi-Modal Fusion Framework for Intelligent Examination Cheating Detection},
  author={Idrees, Alina and Kubra, Khadija Tul and Majeed, Rimsha},
  year={2025},
  institution={Punjab University College of Information Technology (PUCIT)},
  type={Final Year Project (FYP)},
  note={Video-based Action Recognition with Object Detection and Pose Reasoning}
}
```

---

**License**: MIT License - See LICENSE file for details

**Last Updated**: June 11, 2026

---
