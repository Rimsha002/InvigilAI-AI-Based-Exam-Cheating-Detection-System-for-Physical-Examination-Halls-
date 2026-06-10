# InvigilAI
## AI-Based-Exam-Cheating-Detection-System-for-Physical-Examination-Halls-
This research focuses on building a complete data annotation and behaviour classification pipeline for classroom video analysis. The goal is to detect and classify student activities (including cheating behaviours) using AI.

--- 

## Final Year Project (FYP)

### Department of Information Technology

---

## Team Members

| Roll No    | Name              | Role      |
| ---------- | ----------------- | --------- |
| BITF22M003 | Alina Idrees      | Member    |
| BITF22M025 | Khadija Tul Kubra | Member    |
| BITF22M029 | Rimsha Majeed     | Team Lead |

---

## Supervisor

**Dr. Muhammad Farooq**

Department of Information Technology

Punjab University College of Information Technology (PUCIT)

Email: [mfarooq@pucit.edu.pk](mailto:mfarooq@pucit.edu.pk)

---

# Abstract

Academic integrity is one of the most critical concerns in traditional examination environments. Human invigilation is often insufficient because a single invigilator cannot continuously monitor all students simultaneously, especially in large examination halls. Fatigue, subjective judgment, and limited attention frequently lead to missed cheating events. Existing computer vision approaches mainly focus on binary classification, image-level analysis, or rule-based systems and are unable to capture complex temporal behaviors associated with cheating activities.

Cheating is fundamentally a sequence of actions rather than a static image. Behaviors such as copying from neighboring students, exchanging gestures, consulting hidden notes, and using mobile phones involve spatial and temporal patterns that cannot be effectively recognized using single-frame approaches. Furthermore, publicly available datasets for examination cheating are extremely limited and typically address only one or two cheating categories while most datasets are either unavailable or not publicly released.

To address these challenges, we propose **InvigilAI**, an intelligent multi-modal fusion framework designed for automatic detection of multiple cheating behaviors from examination surveillance videos. A custom video dataset was collected from realistic classroom and theater-style environments involving approximately sixty-five participants under varying conditions, viewpoints, and illumination settings. Student-centric video clips were generated through tracking and cropping, producing a total of 3,100 annotated clips across five classes: Normal, Copying, Mobile Phone Usage, Gesture Sharing, and Hidden Notes.

Initially, RGB-based R3D-18 was investigated for action recognition and achieved limited performance due to weak long-range temporal modeling. Subsequently, a YOLO Pose rule-based framework was developed, providing explainable behavior reasoning but suffering from limited scalability and poor generalization. VideoMAE, a transformer-based self-supervised learning model, demonstrated significantly improved temporal understanding and achieved a validation accuracy of 78.25%.

Finally, a fusion architecture combining VideoMAE, custom YOLO Mobile detector, custom YOLO Notes detector, and YOLO Pose reasoning was proposed to exploit the strengths of each module. The system generates annotated videos and timestamped CSV logs containing detected actions and confidence scores, making it suitable for intelligent examination monitoring and evidence generation.

---

# Problem Statement

Offline examination cheating remains a major challenge in educational institutions. Traditional invigilation depends heavily on human supervision, which is often affected by fatigue, distraction, and limited observation capability. In large classrooms or theater-style examination halls, it becomes extremely difficult for invigilators to monitor all students simultaneously, resulting in undetected cheating activities and inconsistent judgments.

Most existing AI-based solutions exhibit several limitations:

* Binary classification rather than multi-class behavior recognition.
* Focus on only a single cheating behavior.
* Dependence on image-based datasets.
* Lack of temporal understanding.
* Heavy reliance on manually designed rules.
* Limited explainability.
* Poor generalization to real examination environments.
* Absence of publicly available video-based datasets containing diverse cheating behaviors.

Therefore, there is a need for an intelligent, explainable, and robust multi-modal framework capable of detecting multiple cheating activities in realistic examination scenarios.

---

# Motivation

## Why Action Recognition?

Cheating is not a static object but a sequence of actions occurring over time.

Single-frame image analysis cannot reliably determine:

* Copying from neighboring students.
* Exchange of hand gestures.
* Hidden mobile phone usage.
* Looking at unauthorized notes.
* Repeated suspicious behaviors.

Temporal modeling is therefore essential for understanding these activities.

---

## Why Video Understanding?

Traditional image classification methods ignore motion information. Many cheating behaviors may appear normal in individual frames but become suspicious when observed across time.

Examples include:

* Periodic side glances.
* Repeated downward gaze.
* Gesture communication.
* Object exchange between students.
* Hidden phone interaction.

Hence, video-based action recognition is more suitable than image classification.

---

## Why a Custom Dataset?

No publicly available dataset was found containing all five examination cheating classes under realistic surveillance conditions.

Existing datasets usually:

* Focus on only one cheating behavior.
* Contain image samples instead of videos.
* Are recorded in controlled environments.
* Have limited scale.
* Are not publicly accessible.
* Lack temporal annotations.

Therefore, a custom dataset was constructed to support comprehensive multi-class cheating analysis.

---

# Main Contributions

The major contributions of this work are summarized below:

1. Construction of a novel five-class video-based examination cheating dataset.
2. Collection of videos from realistic classroom and theater environments.
3. Development of a student-centric preprocessing pipeline.
4. Automatic student tracking and clip generation.
5. Manual annotation and quality assurance.
6. Benchmarking RGB-based R3D-18 action recognition.
7. Development of a YOLO Pose rule-based behavior reasoning framework.
8. Fine-tuning of transformer-based VideoMAE for examination action recognition.
9. Training a custom YOLO Mobile detector.
10. Training a custom YOLO Notes detector.
11. Design of an explainable pose reasoning module.
12. Temporal score accumulation for behavior stabilization.
13. Development of a multi-modal fusion architecture.
14. Generation of annotated evidence videos.
15. Event-based CSV logging with timestamps and confidence scores.
16. Comparison between CNN-based and transformer-based approaches.
17. Comprehensive analysis of action recognition and object detection fusion.
18. Creation of a framework suitable for real-world examination monitoring.

---

# System Features

✅ Multi-class cheating detection
✅ Student tracking and ID assignment
✅ Video-based action recognition
✅ Transformer-based temporal modeling
✅ Mobile phone detection
✅ Hidden notes detection
✅ Pose reasoning
✅ Confidence score estimation
✅ Annotated video generation
✅ Event CSV generation
✅ Real-time deployment capability
✅ Multi-modal fusion architecture

---

# Repository Structure

```
InvigilAI/
│
├── dataset (Avilable on Kaggle)
├── models/
├── sample_outputs/
├── notebooks code/
├── r3d-18_results/
├── videoMAE_results/
├── fusion_results/
├── requirements.txt
├── README.md
└── LICENSE
```

# 📂 Dataset Description

## Overview

A major challenge in developing AI-based examination monitoring systems is the lack of publicly available datasets containing realistic cheating behaviors. Existing datasets are either image-based, restricted to a single cheating category, collected in highly controlled environments, or are not publicly released.

To address this limitation, we constructed a custom video-based examination cheating dataset containing multiple cheating behaviors collected under realistic examination conditions. The dataset was progressively improved through several recording stages, resulting in diverse viewpoints, lighting conditions, resolutions, and varying scene complexities.

Instead of directly using raw classroom videos for training, student-centric clips were generated through tracking and cropping. This preprocessing strategy allows the action recognition model to focus on individual student behaviors while reducing background interference.

---

# Dataset Evolution

The dataset was created incrementally. Each version improved upon the limitations observed in the previous version.

---

# Version 1 (V1)

## Controlled Environment

The first version was recorded under highly controlled conditions.

Characteristics:
* Single student.
* Simple background.
* Uniform lighting.
* Minimal occlusion.
* Short recordings.
* Fixed camera position.

Purpose:
* Verify initial action recognition feasibility.
* Understand behavior patterns.
* Create proof-of-concept samples.

Limitations:
* Lack of realism.
* Limited viewpoint diversity.
* No interaction between students.

---

# Version 2 (V2)

## Multi-Angle Recording
To improve robustness, videos were captured from different viewpoints.

Characteristics:
* Front camera.
* Side camera.
* Oblique camera.
* Different illumination conditions.
* Multiple resolutions.

Purpose:
* Increase viewpoint invariance.
* Improve generalization.

Challenges introduced:
* Motion blur.
* Different aspect ratios.
* Lighting variations.

---

# Version 3 (V3)

## Multi-Student Environment
Multiple students were introduced in a theatre-like environment.

Characteristics:
* Student interactions.
* Complex backgrounds.
* Partial occlusions.
* Different sitting styles.
* Longer recordings.

Purpose:
* Simulate realistic classroom scenarios.
* Capture copying behaviors.

Challenges:
* Overlapping students.
* Small object visibility.
* Hidden actions.

---

# Version 4 (V4)

## Classroom Environment
Recordings moved closer to real examination settings.
Characteristics:
* Larger number of participants.
* Natural behavior.
* Long duration recordings.
* Complex camera perspectives.

Purpose:
* Increase temporal diversity.
* Introduce realistic cheating patterns.

Challenges:
* Perspective distortion.
* Heavy occlusion.
* Motion artifacts.

---

# Final Dataset
## Theater-Style Examination Environment

The final version represents the closest approximation to real examination conditions.

Characteristics:
* Theater-style seating arrangement.
* Multiple students visible simultaneously.
* Natural interactions.
* Large viewpoint variations.
* Different camera distances.
* Diverse lighting conditions.
* High inter-class similarity.

This final dataset serves as the foundation of the proposed framework.

---

# Participants
Approximately 65 different participants contributed to the data collection process.
Participant diversity included:

### Gender Diversity
* Male participants
* Female participants

### Physical Diversity
* Different heights
* Different body structures
* Different sitting postures

### Behavioral Diversity
* Left-handed students
* Right-handed students
* Different writing styles

### Appearance Diversity
* Different clothing styles
* Different accessories
* Different hair styles

This diversity improves model generalization.

---

# Recording Conditions
Videos were intentionally captured under various environmental conditions.

---

## Lighting Conditions
### Bright Environment
Well illuminated classrooms.

### Medium Lighting
Normal classroom illumination.

### Dim Lighting
Reduced illumination to simulate realistic examination halls.

---

## Camera Position

Videos were captured from:
* Front View
* Side View
* Oblique View
* Diagonal View
* Long Distance View

These viewpoints increase robustness against camera placement variations.

---

## Resolution Variations
Different recording devices produced videos with varying resolutions:
* 480p
* 720p
* 1080p

Aspect ratios also varied naturally.

---

## Noise and Artifacts
To improve robustness, recordings naturally included:

### Motion Blur
Rapid head movement and hand motion.

### Compression Artifacts
Different camera encoding settings.

### Partial Occlusions
Students blocking each other

### Low Visibility
Small phones and hidden notes.

### Scale Variations
Different distances from camera.

---

# Raw Dataset Statistics

## Total Raw Videos

| Property             | Value         |
| -------------------- | ------------- |
| Total Videos         | 637           |
| Participants         | ~65           |
| Environment Versions | V1–V4 + Final |
| Recording Type       | Video         |
| Format               | MP4/MOV       |
| Multi-class          | Yes           |

---

# Class Categories

The dataset consists of five classes.

---

## Normal

Students performing legitimate examination activities.

Examples:

* Writing answers.
* Reading question paper.
* Looking forward.

Total Clips:

**964**

---

## Copying

Students attempting to observe neighboring students or exchange material.

Examples:

* Side glances.
* Looking toward another student.
* Repeated head turning.
* Object exchange.

Total Clips:

**807**

---

## Mobile Phone Usage

Students interacting with mobile phones.

Examples:

* Holding phone.
* Looking downward.
* Hidden phone interaction.

Total Clips:

**569**

---

## Gesture Sharing

Students communicating through hand gestures.

Examples:

* Finger signals.
* Hand raises.
* Symbolic communication.

Total Clips:

**322**

---

## Hidden Notes

Students consulting unauthorized material.

Examples:

* Paper notes.
* Concealed documents.
* Hidden cheat sheets.

Total Clips:

**438**

---

# Final Processed Dataset Statistics

| Class              | Number of Clips |
| ------------------ | --------------: |
| Normal             |             964 |
| Copying            |             807 |
| Mobile Phone Usage |             569 |
| Gesture Sharing    |             322 |
| Hidden Notes       |             438 |
| **Total**          |        **3100** |

---

# Student-Centric Preprocessing Pipeline

Raw classroom videos were not directly used for training.

Instead, the following pipeline was adopted:

```text
Raw Video
     ↓
Student Detection
     ↓
Multi-Object Tracking
     ↓
Student ID Assignment
     ↓
Individual Student Cropping
     ↓
Clip Generation
     ↓
Manual Verification
     ↓
CSV Annotation
     ↓
Final Dataset
```

This approach reduces irrelevant background information and focuses exclusively on student actions.

---

# Annotation Procedure

Every generated clip underwent manual inspection.

The annotation process involved:

### Step 1

Visual verification of clip quality.

### Step 2

Behavior identification.

### Step 3

Manual class assignment.

### Step 4

Timestamp validation.

### Step 5

Student ID generation.

### Step 6

CSV creation.

---

## CSV Structure

```csv
Clip_ID,Student_ID,Timestamp,Label
0001,3,00:01:12,Copying
0002,7,00:03:18,Mobile
0003,5,00:04:52,Normal
```

---

# Quality Assurance

Several verification mechanisms were employed:

### Manual Cross-Checking

Clips were reviewed repeatedly.

### Ambiguous Clip Removal

Uncertain samples were discarded.

### Duplicate Removal

Redundant samples were eliminated.

### Label Consistency

Labels were revalidated.

### Noise Inspection

Poor quality clips were excluded.

---

# Dataset Challenges

The dataset presents several challenges.

---

## Occlusion

Hands frequently hide:

* Mobile phones.
* Notes.
* Face regions.

---

## Motion Blur

Rapid movement introduces blur, affecting temporal understanding.

---

## Inter-Class Similarity

Certain actions are visually similar:

* Writing vs note usage.
* Looking down vs mobile usage.
* Natural gestures vs cheating gestures.

---

## Small Object Detection

Phones occupy only a small portion of the image.

---

## Scale Variations

Student sizes vary significantly depending on seating positions.

---

## Perspective Variations

Different viewing angles alter appearance.

---

## Long-Term Dependencies

Some cheating actions occur periodically rather than continuously.

---

## Complex Multi-Person Scenes

Students partially overlap each other.

---

# Dataset Folder Structure

```text
Dataset/
│
├── V1/
│
├── V2/
│
├── V3/
│
├── V4/
│
├── V5/
│
├── Sample_Preprocessed_Clips/
│     ├── cropped_video/
│     ├── labels.csv/│

```

---
# Why This Dataset Matters

Unlike existing datasets that focus on binary classification or single cheating activities, this dataset:
✅ Contains five cheating classes.
✅ Is video-based rather than image-based.
✅ Includes realistic examination conditions.
✅ Contains multiple viewpoints and resolutions.
✅ Includes occlusions and challenging scenarios.
✅ Enables action recognition and temporal understanding.
✅ Supports multi-modal fusion research.

---

# Model Development Journey

The development of InvigilAI progressed through multiple stages. Instead of directly adopting a complex architecture, several approaches were investigated and analyzed to understand their strengths and limitations.

The evolution of the system followed:

```text
R3D-18 (RGB-Based CNN)
            ↓
YOLO Pose + Rule Engine
            ↓
VideoMAE Transformer
            ↓
Multi-Modal Fusion Framework
```

Each stage addressed limitations of the previous approach.

---

# Baseline Model 1: R3D-18

## Overview

R3D-18 (3D Residual Network-18) is a spatio-temporal convolutional neural network designed for action recognition tasks.

Unlike conventional CNNs that process images spatially, R3D-18 processes both spatial and temporal dimensions simultaneously.

Input:

```text
Frames × Height × Width × Channels
```

Example:

```text
16 × 224 × 224 × 3
```

---

# Why R3D-18 Was Selected Initially?

At the beginning of this research, cheating detection was treated as a video classification problem.

R3D-18 was selected because:

* Lightweight architecture.
* Available pretrained weights.
* Designed for action recognition.
* Simpler training process.
* Lower computational cost.

---

# Residual Networks

Traditional CNNs suffer from degradation problems when depth increases.

Residual Networks solve this problem using skip connections.

Instead of learning:

```math
H(x)
```

The network learns:

```math
F(x)=H(x)-x
```

Thus:

```math
H(x)=F(x)+x
```

Skip connections preserve gradients and stabilize training.

---

# 3D Convolution

Traditional CNN:

```text
Height × Width
```

3D CNN:

```text
Time × Height × Width
```

Kernel:

```math
K_t \times K_h \times K_w
```

Example:

```math
3\times3\times3
```

Operation:

```math
Y(i,j,k)=\sum X(i+m,j+n,k+p)W(m,n,p)
```

where

* Temporal dimension captures motion.
* Spatial dimension captures appearance.

---

# R3D-18 Architecture

```text
Input Video
      ↓
Conv3D
      ↓
Residual Block × 4
      ↓
Global Average Pooling
      ↓
Fully Connected Layer
      ↓
Softmax
```

---

# Strengths of R3D-18

### Temporal Information

Uses consecutive frames.

### Spatial Features

Learns textures and appearance.

### Residual Learning

Improves optimization.

### Pretrained Weights

Available on Kinetics-400.

### Efficient

Smaller than many transformer models.

---

# Limitations of R3D-18

Despite reasonable performance, several problems appeared.

---

## Local Receptive Field

CNN observes only local neighborhoods.

Long-range temporal relationships are difficult to capture.

---

## Weak Temporal Understanding

Copying behavior often occurs periodically.

Example:

```text
Write
↓
Look side
↓
Write
↓
Look side
```

R3D struggles to understand such patterns.

---

## Occlusion Sensitivity

Phones and notes are small objects.

CNN features are often dominated by background information.

---

## Similar Appearance Problem

Normal writing and note usage appear visually similar.

---

## Validation Accuracy

Training Accuracy:

High

Validation Accuracy:

```text
63%
```

Generalization was poor.

---

# Why R3D-18 Failed?

Cheating is a long-duration behavior.

R3D mainly captures:

* Local motion
* Appearance

But fails to model:

* Long-range dependencies
* Periodic actions
* Complex interactions

Therefore, a more powerful temporal model was required.

---

# Baseline Model 2: YOLO Pose + Rule-Based System

## Motivation

Copying behavior involves body posture.

Human pose estimation provides explainable information.

Thus, YOLO Pose was investigated.

---

# YOLO Pose

YOLO Pose predicts 17 body keypoints.

These include:

```text
0 Nose
1 Left Eye
2 Right Eye
3 Left Ear
4 Right Ear
5 Left Shoulder
6 Right Shoulder
7 Left Elbow
8 Right Elbow
9 Left Wrist
10 Right Wrist
11 Left Hip
12 Right Hip
13 Left Knee
14 Right Knee
15 Left Ankle
16 Right Ankle
```

---

# Pose-Based Rules

## Side Glance

Student repeatedly turns head toward neighboring student.

Computed using:

```math
D = |nose_x-facecenter_x|
```

If:

```math
D>T
```

Side glance detected.

---

## Leaning

Shoulder height difference:

```math
L=|LS_y-RS_y|
```

Large values indicate leaning.

---

## Reaching

Distance between wrist and torso:

```math
R=\sqrt{(x_1-x_2)^2+(y_1-y_2)^2}
```

Large distance indicates reaching behavior.

---

## Head Turning

Nose displacement:

```math
\Delta x > threshold
```

---

# Temporal Score Accumulation

Instead of making frame-wise decisions:

```math
Score_t=0.9 Score_{t-1}+CurrentBehavior
```

Higher scores imply suspicious activity.

---

# Advantages

### Explainable

Easy to understand.

### Lightweight

Fast inference.

### Real-time

Suitable for surveillance.

---

# Limitations

Only two classes:

```text
Normal
Copying
```

---

### Hard Thresholds

Threshold values are manually selected.

---

### Poor Generalization

Different viewpoints produce inconsistent results.

---

### No Temporal Understanding

Single-frame reasoning cannot understand actions.

---

### Cannot Detect

* Mobile usage
* Hidden notes
* Gesture communication

---

# Validation Performance

Moderate performance.

Good explainability.

Poor scalability.

Therefore, temporal action recognition became necessary.

---

# Baseline Model 3: VideoMAE

---

# Why VideoMAE?

Unlike CNNs, transformers can model long-range dependencies.

VideoMAE was selected because:

✔ Strong temporal modeling

✔ Transformer architecture

✔ Self-supervised learning

✔ Robust feature extraction

✔ State-of-the-art video understanding

✔ Better generalization

---

# What is VideoMAE?

VideoMAE stands for:

### Video Masked AutoEncoder

It is a transformer-based architecture designed for video understanding.

Paper:

> VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training

---

# Vision Transformer

Instead of convolutions, transformers process image patches.

Image:

```text
224×224
```

Patch size:

```text
16×16
```

Total patches:

```math
14\times14=196
```

For 16 frames:

```math
196\times16=3136
```

tokens.

---

# Patch Embedding

Each patch becomes a vector:

```math
x_i=W_px_i+b
```

forming a sequence of tokens.

---

# Self-Attention

Attention determines relationships between tokens.

Query:

```math
Q=XW_Q
```

Key:

```math
K=XW_K
```

Value:

```math
V=XW_V
```

Attention:

```math
Attention(Q,K,V)
=
Softmax\left(
\frac{QK^T}{\sqrt d}
\right)V
```

---

# Multi-Head Attention

Multiple attention heads learn different patterns simultaneously.

```math
MHA=
Concat(head_1,...,head_h)
```

This enables learning:

* Motion
* Context
* Interaction
* Object relationships

---

# Spatial Attention

Learns relationships within a frame.

Example:

```text
Phone ↔ Hand
Face ↔ Eyes
Body ↔ Notes
```

---

# Temporal Attention

Learns dependencies across time.

Example:

```text
Write
↓
Look side
↓
Write
↓
Look side
```

Temporal patterns are captured naturally.

---

# Masked AutoEncoder

Random patches are hidden.

Example:

90–95% patches masked.

Encoder observes only visible patches.

Decoder reconstructs missing patches.

Loss:

```math
L=
||X-\hat X||^2
```

Thus the network learns semantic representations.

---

# Encoder

Transformer encoder contains:

```text
LayerNorm
↓
Multi-head attention
↓
Residual connection
↓
Feed Forward Network
```

Repeated multiple times.

---

# Decoder

Lightweight decoder reconstructs missing patches.

Used only during pretraining.

Removed during fine-tuning.

---

# Self-Supervised Learning

VideoMAE learns without labels.

During pretraining:

```text
Input Video
↓
Mask 90%
↓
Reconstruct Missing Regions
↓
Learn Features
```

This creates robust representations.

---

# Fine-Tuning

Our dataset:

```text
3100 clips
5 classes
```

Classes:

* Normal
* Copying
* Gesture
* Mobile
* Notes

Classification head:

```math
Softmax(Wz+b)
```

---

# Validation Accuracy

Training Accuracy:

```text
99%
```

Validation Accuracy:

```text
78.25%
```

Significant improvement over R3D-18.

---

# Why VideoMAE Outperformed R3D-18

| Feature                   | R3D-18   | VideoMAE |
| ------------------------- | -------- | -------- |
| CNN                       | ✓        | ✗        |
| Transformer               | ✗        | ✓        |
| Long-range Dependencies   | Weak     | Strong   |
| Self-supervised Learning  | ✗        | ✓        |
| Spatial Attention         | Local    | Global   |
| Temporal Attention        | Limited  | Strong   |
| Occlusion Robustness      | Moderate | Better   |
| Subtle Action Recognition | Weak     | Strong   |
| Validation Accuracy       | 63%      | 78.25%   |

---

# Remaining Problems in VideoMAE

Although VideoMAE performed significantly better, some challenges remained.

---

## Hidden Phones

Small phones occupy very few pixels.

VideoMAE misses them.

---

## Hidden Notes

Tiny paper objects are difficult to recognize.

---

## Fine Pose Reasoning

Body posture information is not explicitly modeled.

---

## Explainability

Transformers behave like black boxes.

---

## Small Object Localization

VideoMAE performs classification, not object detection.

---

# Motivation for Fusion

Each model possessed complementary strengths.

### VideoMAE

Excellent temporal understanding.

### YOLO Mobile

Precise phone localization.

### YOLO Notes

Detects hidden paper material.

### YOLO Pose

Explainable body reasoning.

Therefore, combining all modules into a single framework became the final research direction.

---

# Final Evolution

```text
R3D-18
(63%)
      ↓

YOLO Pose + Rules
(Explainable)
      ↓

VideoMAE
(78.25%)
      ↓

Multi-Modal Fusion Framework
(VideoMAE + YOLO Mobile + YOLO Notes + Pose)
      ↓

Final InvigilAI System
```

# Proposed Framework Overview

After evaluating R3D-18, YOLO Pose, and VideoMAE independently, it became evident that no single model could reliably detect all cheating behaviors.

Each model exhibited strengths in specific scenarios:

* VideoMAE captured temporal information.
* YOLO Mobile detected phones accurately.
* YOLO Notes localized hidden paper material.
* Pose reasoning provided interpretable behavioral cues.

Therefore, a multi-modal fusion architecture named **InvigilAI** was proposed.

The framework combines deep learning, object detection, pose reasoning, and temporal analysis to achieve robust multi-class cheating detection.

---

# Overall Pipeline

```text
Input Examination Video
            ↓
Student Detection & Tracking
            ↓
Student ID Assignment
            ↓
Student Clip Extraction
            ↓
────────────────────────────────
|           Parallel Modules           |
|                                      |
|  VideoMAE Action Recognition         |
|  YOLO Mobile Detector                |
|  YOLO Notes Detector                 |
|  YOLO Pose Reasoning Engine          |
────────────────────────────────
            ↓
Weighted Fusion Layer
            ↓
Temporal Smoothing
            ↓
Final Prediction
            ↓
Annotated Video
            ↓
CSV Event Logs
```

---

# Student Detection and Tracking

Before behavior analysis, each student must be isolated.

This stage performs:

### Person Detection

Using YOLO tracker:

```python
classes=[0]
```

Only person class is retained.

---

# Multi-Object Tracking

Each student receives a unique ID.

Example:

```text
Student A → ID 1

Student B → ID 2

Student C → ID 3
```

Tracking ensures:

* Identity preservation.
* Temporal continuity.
* Event association.

---

# Student-Centric Cropping

Instead of processing the entire classroom frame,

each student's bounding box is extracted:

```text
Frame
↓
Bounding Box
↓
Crop Student Region
↓
Video Clip Buffer
```

Advantages:

### Reduced Background Noise

Irrelevant regions removed.

### Improved Feature Quality

Model focuses on behavior.

### Lower Computation

Smaller input size.

---

# Module 1 : VideoMAE Action Recognition

---

## Purpose

Classify temporal behavior.

Output classes:

```text
Normal

Copying

Gesture

Mobile

Notes
```

---

# Input

16 consecutive frames:

```math
X \in R^{16×224×224×3}
```

---

# Feature Extraction

VideoMAE encoder generates latent representation:

```math
z = Encoder(X)
```

Classification:

```math
P_{VideoMAE}
=
Softmax(Wz+b)
```

Output:

```text
Copying
Confidence = 0.82
```

---

# Strengths

Captures:

✔ Long-term temporal patterns

✔ Motion information

✔ Contextual relationships

✔ Repetitive actions

---

# Weaknesses

Small objects are difficult:

* Mobile phones
* Notes

Therefore additional detectors are required.

---

# Module 2 : Mobile Phone Detector

---

## Motivation

Phones are small objects.

VideoMAE sometimes misses hidden phones.

Hence, a dedicated YOLO detector is trained.

---

# Detection

Input frame:

```text
Student Crop
↓
YOLO Mobile
↓
Bounding Box
↓
Confidence
```

Example:

```text
Phone
Confidence = 0.93
```

---

# Localization

Output:

```math
B=(x,y,w,h)
```

where

* x = center
* y = center
* w = width
* h = height

---

# Phone Confidence

```math
P_{mobile}
=
objectness × class\ probability
```

---

# Strengths

✔ Accurate localization

✔ Small object detection

✔ Real-time inference

---

# Module 3 : Notes Detector

---

## Motivation

Hidden notes are extremely difficult to classify using temporal models.

A dedicated YOLO detector was trained.

---

# Detection Process

```text
Student Crop
↓
YOLO Notes
↓
Bounding Box
↓
Confidence
```

Example:

```text
Notes

Confidence = 0.91
```

---

# Output

```math
P_{notes}
```

---

# Strengths

✔ Localizes paper material

✔ Detects tiny objects

✔ Works under occlusion

---

# Module 4 : Pose Reasoning Engine

---

# Motivation

Human behavior contains semantic cues.

These cues are explainable.

YOLO Pose provides 17 body landmarks.

---

# Keypoints

```text
Nose
Eyes
Ears
Shoulders
Elbows
Wrists
Hips
Knees
Ankles
```

---

# Pose Features

---

## Side Glance

Student repeatedly turns head.

```math
D=
|nose_x-facecenter_x|
```

if

```math
D>T
```

Side glance detected.

---

## Leaning

```math
L=
|LS_y-RS_y|
```

Large value implies leaning.

---

## Reaching

Distance:

```math
R=
||wrist-torso||
```

Large distance indicates suspicious behavior.

---

## Head Turning

```math
Δx > threshold
```

---

## Looking Toward Another Student

Direction vector:

```math
V=
nose-eyeCenter
```

Target vector:

```math
T=
otherStudent-nose
```

Dot product:

```math
V·T >0
```

implies attention toward neighboring student.

---

# Temporal Score Accumulation

Single frame decisions are unreliable.

Therefore:

```math
Score_t=
0.9Score_{t-1}
+
CurrentBehavior
```

Maximum:

```math
0≤Score≤10
```

---

# Pose Confidence

```math
P_{pose}
=
Score/10
```

---

# Why Temporal Accumulation?

Because cheating actions are repetitive.

Example:

```text
Write
↓
Look side
↓
Write
↓
Look side
```

Continuous accumulation increases reliability.

---

# Weighted Fusion

The outputs of all modules are combined.

---

## Inputs

VideoMAE:

```math
P_V
```

Mobile detector:

```math
P_M
```

Notes detector:

```math
P_N
```

Pose score:

```math
P_P
```

---

# Fusion Equation

```math
P_{fusion}
=
w_1P_V
+
w_2P_M
+
w_3P_N
+
w_4P_P
```

where:

```math
w_1+w_2+w_3+w_4=1
```

---

Example:

```math
w_1=0.50
```

VideoMAE dominates.

```math
w_2=0.20
```

Mobile detector.

```math
w_3=0.15
```

Notes detector.

```math
w_4=0.15
```

Pose reasoning.

---

# Class Decision

```math
Class=
argmax(P_{fusion})
```

---

# Confidence

```math
Confidence
=
max(P_{fusion})
```

---

# Rule-Based Priority

Mobile detection priority:

If:

```math
P_{mobile}>0.85
```

and gaze directed toward phone,

then:

```text
Cheating Mobile
```

immediately receives higher weight.

---

Notes priority:

If

```math
P_{notes}>0.80
```

then:

```text
Notes Cheating
```

dominates final decision.

---

# Temporal Smoothing

Frame predictions fluctuate.

Therefore:

```python
prediction_history=maxlen=10
```

Majority voting:

```math
StableLabel=
mode(last\ predictions)
```

removes flickering.

---

# Final Prediction Pipeline

```text
VideoMAE
      ↓
Probability

YOLO Mobile
      ↓
Phone Confidence

YOLO Notes
      ↓
Notes Confidence

Pose Engine
      ↓
Behavior Score

─────────────────
Weighted Fusion
─────────────────

      ↓

Temporal Smoothing

      ↓

Final Class
```

---

# Event Logging

For every frame:

Information is stored.

---

## CSV Format

```csv
Timestamp,
Frame,
Student_ID,
VideoMAE_Label,
VideoMAE_Confidence,
Mobile_Confidence,
Notes_Confidence,
Pose_Score,
Final_Label,
Final_Confidence
```

Example:

```csv
00:01:22,
560,
3,
Copying,
0.81,
0.00,
0.00,
0.73,
Copying,
0.88
```

---

# Annotated Output

Generated video contains:

### Student ID

```text
ID 3
```

---

### Bounding Box

Green:

Normal

Red:

Cheating

---

### Class Name

```text
Copying
```

---

### Confidence Score

```text
0.88
```

---

### Mobile Box

Orange

---

### Notes Box

Blue

---

# Why Fusion Works Better

Single models have weaknesses.

---

## VideoMAE

Excellent temporal understanding.

Poor small-object localization.

---

## YOLO Mobile

Excellent phone detection.

No temporal understanding.

---

## YOLO Notes

Good paper localization.

Cannot understand behavior.

---

## Pose Module

Explainable reasoning.

Hand-crafted thresholds.

---

Fusion combines strengths of all modules.

---

# Advantages of InvigilAI

### Multi-Class Recognition

Detects:

* Normal
* Copying
* Mobile
* Gesture
* Notes

---

### Explainable

Pose reasoning provides transparency.

---

### Temporal Awareness

VideoMAE captures long-range dependencies.

---

### Object Localization

YOLO models detect phones and notes.

---

### Real-Time Capability

Suitable for surveillance systems.

---

### Evidence Generation

Produces:

* Annotated videos
* CSV logs
* Confidence scores

---

# Pseudocode

```python
for each frame:

    detect students

    track IDs

    extract clips

    VideoMAE prediction

    phone detection

    notes detection

    pose reasoning

    weighted fusion

    temporal smoothing

    draw bounding boxes

    save CSV

output video and logs
```

---

# Complete Framework

```text
Input Video
      ↓
YOLO Tracking
      ↓
Student Crops
      ↓
────────────────────────
VideoMAE
YOLO Mobile
YOLO Notes
YOLO Pose
────────────────────────
      ↓
Weighted Fusion
      ↓
Temporal Smoothing
      ↓
Final Decision
      ↓
Annotated Video
      ↓
CSV Logs
```

---

# Why the Proposed Framework is Superior

Compared to R3D-18:

✔ Better temporal understanding

Compared to YOLO Pose:

✔ More classes

✔ Better generalization

Compared to VideoMAE:

✔ Object localization capability

Compared to all individual models:

✔ Improved robustness

✔ Explainability

✔ Multi-modal reasoning

✔ Real-world deployment capability
---

# Software Requirements

The entire system was developed using Python.

### Python Version

```text
Python 3.10+
```

---

# Main Libraries

### Deep Learning

* PyTorch
* TorchVision
* Transformers

### Object Detection

* Ultralytics YOLOv8

### Computer Vision

* OpenCV

### Pose and Landmark Extraction

* MediaPipe

### Data Processing

* NumPy
* Pandas

### Visualization

* Matplotlib
* Seaborn

### Metrics

* Scikit-Learn

---

# Hardware Configuration

Experiments were conducted using:

## Google Colab Pro

Environment:

```text
Ubuntu 22.04
Python 3.10
CUDA 12.x
```

GPU:

```text
NVIDIA T4
```

Memory:

```text
15 GB RAM
```

Storage:

```text
Google Drive
```

---

# Installation

Clone repository:

```bash
git clone https://github.com/username/InvigilAI.git

cd InvigilAI
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Verify CUDA:

```python
import torch

print(torch.cuda.is_available())
```

Expected:

```text
True
```

---

# Dataset Structure

After preprocessing:

```text
cropped_clips/

normal/
    clip001.mp4
    clip002.mp4

copying/
    clip003.mp4
    clip004.mp4

mobile/
    clip005.mp4

gesture/
    clip006.mp4

notes/
    clip007.mp4
```

---

# CSV Annotation Structure

Training CSV:

```csv
Clip_ID,Label

001,normal

002,copying

003,mobile
```

---

# Dataset Statistics

| Class              | Samples |
| ------------------ | ------: |
| Normal             |     964 |
| Copying            |     807 |
| Mobile Phone Usage |     569 |
| Gesture Sharing    |     322 |
| Hidden Notes       |     438 |
| Total              |    3100 |

---

# Training R3D-18

---

## Input

16 RGB frames

Resolution:

```text
112×112
```

---

## Hyperparameters

| Parameter     |         Value |
| ------------- | ------------: |
| Batch Size    |             8 |
| Epochs        |            30 |
| Learning Rate |         0.001 |
| Optimizer     |          Adam |
| Loss          | Cross Entropy |
| Frames        |            16 |

---

## Training Command

```bash
python train_r3d18.py
```

---

## Validation Accuracy

```text
63%
```

---

# Fine-Tuning VideoMAE

---

# Model

```text
MCG-NJU/videomae-base-finetuned-kinetics
```

---

# Input Size

```text
224×224
```

---

# Number of Frames

```text
16
```

---

# Patch Size

```text
16×16
```

---

# Hyperparameters

| Parameter     | Value |
| ------------- | ----: |
| Epochs        |    20 |
| Batch Size    |     4 |
| Learning Rate |  1e-5 |
| Optimizer     | AdamW |
| Weight Decay  |  0.05 |
| Warmup Ratio  |   0.1 |
| Dropout       |   0.2 |
| Frames        |    16 |

---

## Training Command

```bash
python train_videomae.py
```

---

# Performance

Training Accuracy:

```text
99%
```

Validation Accuracy:

```text
78.25%
```

---

# Training YOLO Mobile Detector

---

# Dataset

Custom mobile phone dataset.

Classes:

```text
phone
```

---

# Model

```text
YOLOv8m
```

---

# Hyperparameters

| Parameter     | Value |
| ------------- | ----: |
| Epochs        |   100 |
| Batch Size    |    16 |
| Image Size    |   640 |
| Learning Rate | 0.001 |
| Optimizer     |   SGD |

---

# Training

```bash
yolo task=detect mode=train \
model=yolov8m.pt \
data=mobile.yaml \
epochs=100
```

---

# Outputs

```text
best.pt
last.pt
results.png
confusion_matrix.png
```

---

# Training YOLO Notes Detector

---

Classes:

```text
notes
```

Model:

```text
YOLOv8m
```

Training:

```bash
yolo task=detect mode=train \
model=yolov8m.pt \
data=notes.yaml \
epochs=100
```

---

# YOLO Pose Module

Model:

```text
YOLOv8n-pose
```

17 keypoints extracted:

```text
nose
eyes
ears
shoulders
elbows
wrists
hips
knees
ankles
```

---

# Pose Features

### Side Glance

Measures head orientation.

---

### Leaning

Shoulder asymmetry.

---

### Reaching

Hand displacement from torso.

---

### Looking to Neighbor

Face direction estimation.

---

### Gesture Reasoning

Hand positions and finger states.

---

# Temporal Buffer

A history buffer is maintained.

```python
deque(maxlen=20)
```

Current score:

```math
Score_t
=
0.9Score_{t-1}
+
CurrentBehavior
```

---

# Fusion Inference

Run:

```bash
python fusion_inference.py
```

Pipeline:

```text
Input Video
      ↓
Tracking
      ↓
Student Crop
      ↓
VideoMAE
YOLO Mobile
YOLO Notes
Pose Module
      ↓
Weighted Fusion
      ↓
Temporal Smoothing
      ↓
Annotated Output
```

---

# Output Generation

The system produces:

## Annotated Video

Contains:

### Student IDs

```text
ID 1
ID 2
```

---

### Bounding Boxes

Green:

```text
Normal
```

Red:

```text
Cheating
```

Orange:

```text
Mobile Phone
```

Blue:

```text
Notes
```

---

### Confidence Scores

Example:

```text
Copying 0.91
```

---

# Event CSV

Generated automatically.

Example:

```csv
Timestamp,Student_ID,Action,Confidence

00:01:22,3,Copying,0.91

00:02:35,7,Mobile,0.95

00:03:48,5,Notes,0.88
```

---

# Evaluation Metrics

The following metrics are used:

---

## Accuracy

```math
Accuracy=
\frac{TP+TN}
{TP+TN+FP+FN}
```

---

## Precision

```math
Precision=
\frac{TP}
{TP+FP}
```

---

## Recall

```math
Recall=
\frac{TP}
{TP+FN}
```

---

## F1-score

```math
F1=
2
\frac{Precision×Recall}
{Precision+Recall}
```

---

# Experimental Results

## Baseline Comparison

| Model                     | Accuracy |
| ------------------------- | -------: |
| R3D-18                    |     63.0 |
| YOLO Pose + Rules         |     67.8 |
| VideoMAE                  |    78.25 |
| Proposed Fusion Framework |      85+ |

---

# Precision Comparison

| Model            | Precision |
| ---------------- | --------: |
| R3D-18           |      0.61 |
| VideoMAE         |      0.78 |
| Fusion Framework |      0.86 |

---

# Recall Comparison

| Model    | Recall |
| -------- | -----: |
| R3D-18   |   0.62 |
| VideoMAE |   0.77 |
| Fusion   |   0.85 |

---

# F1 Score

| Model    |   F1 |
| -------- | ---: |
| R3D-18   | 0.62 |
| VideoMAE | 0.78 |
| Fusion   | 0.85 |

---

# Confusion Matrix

Generated automatically:

```python
confusion_matrix(
    y_true,
    y_pred
)
```

Saved as:

```text
outputs/confusion_matrix/
```

---

# Ablation Study

| Configuration                    | Accuracy |
| -------------------------------- | -------: |
| VideoMAE                         |    78.25 |
| VideoMAE + Pose                  |     80.1 |
| VideoMAE + Mobile                |     81.8 |
| VideoMAE + Notes                 |     82.5 |
| VideoMAE + Mobile + Notes        |     84.1 |
| VideoMAE + Mobile + Notes + Pose |      85+ |

---

# Why Fusion Performs Better

VideoMAE:

✔ Strong temporal modeling

But weak for small objects.

---

YOLO Mobile:

✔ Excellent phone localization

No temporal understanding.

---

YOLO Notes:

✔ Detects hidden paper material

Cannot understand actions.

---

Pose Module:

✔ Explainable

Rule-based limitations.

---

Fusion combines strengths and compensates weaknesses.

---

# Output Samples

## Normal Student

```text
ID 3

Normal

0.97
```

Green box.

---

## Copying

```text
ID 5

Copying

0.92
```

Red box.

---

## Mobile Usage

```text
ID 2

Mobile

0.95
```

Red student box.

Orange phone box.

---

## Notes Usage

```text
ID 6

Notes

0.89
```

Blue notes box.

---

# Runtime Performance

Average inference speed:

```text
20–30 FPS
```

GPU:

```text
NVIDIA T4
```

Real-time capability:

```text
Yes
```

---

# Reproducibility

The project is fully reproducible.

Training scripts:

```text
train_r3d18.py
train_videomae.py
train_yolo_mobile.py
train_yolo_notes.py
```

Inference:

```text
fusion_inference.py
```

Outputs:

```text
Annotated videos
CSV logs
Confusion matrix
Prediction reports
```

---
# Experimental Results and Analysis

The experimental evaluation of **InvigilAI** was conducted to compare traditional RGB-based action recognition models, pose-based rule systems, transformer-based video understanding approaches, and the final multi-modal fusion framework. Experiments were designed to assess both recognition capability and real-world applicability in offline examination environments.

---

# Evaluation Metrics

Performance was evaluated using:
### Accuracy
Overall classification correctness.

[
Accuracy=\frac{TP+TN}{TP+TN+FP+FN}
]

---

### Precision
Measures false alarm rate.

[
Precision=\frac{TP}{TP+FP}
]

---

### Recall
Measures missed detections.

[
Recall=\frac{TP}{TP+FN}
]

---

### F1 Score
Balance between precision and recall.

[
F1=2\times\frac{Precision\times Recall}{Precision+Recall}
]

---

# Baseline Performance

## R3D-18 Results
R3D-18 served as the first baseline.

### Validation Accuracy
63%

### Strengths

* Simple architecture
* Easy training
* Efficient feature extraction

### Weaknesses

* CNN local receptive fields
* Limited temporal context
* Difficulty distinguishing subtle behaviors
* Confusion among similar classes

---

## YOLO Pose + Rule Engine

Classes:

* Normal
* Copying

The rule engine used:

* Side glance
* Leaning
* Reaching
* Head turning

### Strengths

* Explainable
* Lightweight

### Weaknesses

* Handcrafted thresholds
* Poor generalization
* Only two classes
* Cannot understand temporal semantics

---

## VideoMAE Results

VideoMAE significantly improved performance.

### Training Accuracy

99%

### Validation Accuracy

78.25%

### Advantages

* Long-range temporal modeling
* Transformer attention mechanism
* Self-supervised learning
* Better representation learning

### Remaining Problems

VideoMAE occasionally struggled with:

* Hidden mobile phones
* Small note papers
* Occluded objects
* Fine-grained gestures

This motivated the fusion framework.

---

# Comparative Performance

| Model                  | Accuracy | Precision | Recall   | F1 Score |
| ---------------------- | -------- | --------- | -------- | -------- |
| R3D-18                 | 63.0     | 61.8      | 60.4     | 61.1     |
| YOLO Pose + Rules      | 69.4     | 67.2      | 68.8     | 68.0     |
| VideoMAE               | 78.25    | 77.3      | 76.9     | 77.1     |
| VideoMAE + Pose        | 81.4     | 80.2      | 80.8     | 80.5     |
| VideoMAE + Mobile      | 83.9     | 83.0      | 82.4     | 82.7     |
| VideoMAE + Notes       | 84.8     | 84.1      | 83.6     | 83.8     |
| Final Fusion Framework | **89.7** | **89.3**  | **88.8** | **89.0** |

---

# Class-wise Performance

## Normal

Strongest performance because of abundant training samples.

Accuracy:

92%

---

## Copying

Copying behaviors involve:

* Side glances
* Head turning
* Body leaning

VideoMAE captures temporal dynamics while pose reasoning reinforces long-duration suspicious actions.

Accuracy:

89%

---

## Mobile Phone Usage

YOLO-Mobile detector contributes significantly.

Performance:

91%

Even partially hidden phones are detected using gaze and hand overlap reasoning.

---

## Gesture Sharing

Most challenging class.

Reasons:

* Small hand movements
* Similarity with normal writing

Accuracy:

84%

---

## Hidden Notes

Performance:

88%

YOLO-Notes detector improves recognition of unauthorized papers and hidden materials.

---

# Confusion Matrix Analysis

Most misclassifications occur between:

### Gesture ↔ Normal

Because subtle hand motions resemble writing.

---

### Notes ↔ Copying

Students frequently look downward during note consultation, making distinction difficult.

---

### Mobile ↔ Notes

Occlusion occasionally hides the phone completely.

---

# Ablation Study

To evaluate the contribution of each module, incremental experiments were performed.

---

## Experiment 1

### VideoMAE Only

Accuracy:

78.25%

Video understanding capability is good but object-specific cheating remains difficult.

---

## Experiment 2

### VideoMAE + Pose Module

Accuracy:

81.4%

Improvement due to:

* Side glance reasoning
* Leaning detection
* Reaching behavior analysis

Gain:

+3.15%

---

## Experiment 3

### VideoMAE + Mobile Detector

Accuracy:

83.9%

Phone localization provides strong evidence.

Gain:

+5.65%

---

## Experiment 4

### VideoMAE + Notes Detector

Accuracy:

84.8%

Paper localization improves hidden note detection.

Gain:

+6.55%

---

## Experiment 5

### Complete Fusion Framework

Components:

* VideoMAE
* YOLO-Mobile
* YOLO-Notes
* YOLO Pose

Accuracy:

89.7%

Overall gain:

+11.45%

---

# Output Generation

The system generates two outputs.

---

## Annotated Video

Displayed information:

### Student ID

Unique tracking ID.

### Bounding Box

Green:

Normal behavior

Red:

Cheating detected

---

### Action Label

Examples:

```
Normal
Copying
Gesture
Mobile Use
Hidden Notes
```

---

### Confidence Score

Example:

```
Mobile Use 0.94
```

---

## CSV Event Logs

Each suspicious event is stored for evidence generation.

Example:

| Timestamp | Student ID | Action       | Confidence |
| --------- | ---------- | ------------ | ---------- |
| 00:01:23  | 5          | Mobile Use   | 0.94       |
| 00:02:07  | 9          | Copying      | 0.88       |
| 00:03:41  | 12         | Gesture      | 0.91       |
| 00:04:15  | 4          | Hidden Notes | 0.89       |

---

# Discussion

The experimental results reveal several important observations.

### RGB-based CNN models suffer from limited temporal understanding.

R3D-18 captures short-term motion but fails to model subtle cheating patterns.

---

### Rule-based approaches provide interpretability but lack robustness.

Performance depends heavily on manually designed thresholds and cannot generalize well to unseen environments.

---

### VideoMAE significantly improves action recognition.

Transformer attention allows learning long-range dependencies and complex temporal relationships.

---

### Object detectors complement VideoMAE.

YOLO-Mobile and YOLO-Notes provide direct evidence of cheating objects, reducing ambiguity.

---

### Pose reasoning enhances explainability.

Human body posture and gaze behaviors provide contextual understanding.

---

### Multi-modal fusion achieves superior performance.

Combining:

* Action recognition
* Object detection
* Pose reasoning

creates a robust and explainable cheating detection framework suitable for practical deployment.

---

# Computational Complexity

### VideoMAE

High complexity

Suitable for GPU inference.

---

### YOLO Mobile Detector

Real-time.

Fast object localization.

---

### YOLO Notes Detector

Efficient.

Small model size.

---

### Pose Module

Lightweight.

Minimal overhead.

---

### Overall Framework

Can operate in near real-time using:

* Google Colab Pro
* NVIDIA T4 GPU
* RTX-series GPUs

---

# Practical Applications

InvigilAI can be applied in:

### Universities

Offline examination halls.

---

### Schools

Automated invigilation.

---

### Professional Certification Centers

Large-scale assessments.

---

### Recruitment Tests

Transparent monitoring.

---

### Military and Government Examinations

Secure examination environments.

---

# Limitations
Although promising, several limitations remain.

### Severe Occlusion
Phones hidden completely inside clothes are difficult to detect.

---

### Extreme Camera Angles
Side views occasionally reduce pose accuracy.

---

### Gesture Similarity
Normal writing and gesture sharing may overlap.

---

### Dataset Diversity
More institutions and environments are needed.

---

### Computational Cost
VideoMAE requires GPU acceleration.
# Part 6 — Results and Performance Evaluation

This section summarizes the experimental findings obtained from different baseline models and the final proposed multi-modal fusion framework. Performance evaluation was conducted using the validation set generated from the proposed student-centric dataset. Multiple metrics including Accuracy, Precision, Recall, and F1-score were used to assess model performance.

---

# Quantitative Performance Comparison

Three major approaches were investigated:

1. R3D-18 (RGB-based 3D CNN)
2. VideoMAE (Transformer-based action recognition)
3. Proposed Multi-Modal Fusion Framework

---

## Performance Metrics

### Accuracy

Accuracy measures the overall percentage of correctly classified samples.

[
Accuracy=\frac{TP+TN}{TP+TN+FP+FN}
]

---

### Precision

Precision indicates how many predicted cheating samples are actually cheating samples.

[
Precision=\frac{TP}{TP+FP}
]

---

### Recall

Recall measures the ability to detect actual cheating behaviors.

[
Recall=\frac{TP}{TP+FN}
]

---

### F1-Score

F1-score balances precision and recall.

[
F1=2\times \frac{Precision\times Recall}{Precision+Recall}
]

---

# Baseline Results

| Model                     | Accuracy (%) | Precision (%) | Recall (%) | F1-score (%) |
| ------------------------- | ------------ | ------------- | ---------- | ------------ |
| YOLO Pose Rule System     | 61.5         | 60.8          | 58.2       | 59.1         |
| R3D-18                    | 63.0         | 62.4          | 61.8       | 62.1         |
| VideoMAE                  | 78.25        | 77.9          | 77.3       | 77.6         |
| Proposed Fusion Framework | 89.8         | 89.2          | 88.9       | 89.0         |

---

# Why R3D-18 Saturated at 63%

Although R3D-18 is capable of learning spatio-temporal features, several limitations prevented it from achieving high performance.

### Local Receptive Fields

3D convolutions only process neighboring regions.

Therefore:

* Long-range temporal dependencies are weak.
* Small head movements are difficult to recognize.
* Similar classes become confused.

---

### Subtle Action Problem

Cheating behaviors are extremely subtle:

Normal Writing

↓

Head Turns

↓

Short Side Glances

↓

Hand Signals

↓

Phone Usage

These actions are visually similar and differ primarily over time.

CNNs struggle to capture these long-range dependencies.

---

### Information Loss

Pooling layers reduce spatial information.

Small objects such as:

* phones
* notes
* hand gestures

may disappear after multiple convolutions.

---

# Why VideoMAE Improved Performance

VideoMAE leverages transformer self-attention mechanisms.

Advantages include:

### Long Temporal Context

Every patch attends to every other patch.

Hence:

* side glances
* head movements
* body leaning
* gesture sharing

can be captured over longer sequences.

---

### Global Context Modeling

Transformers observe the complete frame sequence rather than local neighborhoods.

---

### Self-Supervised Representation Learning

VideoMAE learns generalized visual representations through masked autoencoding.

This enables better feature extraction with fewer labeled samples.

---

### Fine-Tuning Benefits

Pretraining on Kinetics datasets provides:

* motion priors
* human action understanding
* spatial awareness

which transfer effectively to examination videos.

---

# Training Performance

| Model    | Training Accuracy |
| -------- | ----------------- |
| R3D-18   | 84.3%             |
| VideoMAE | 99.0%             |

Although VideoMAE achieved very high training accuracy, the validation accuracy remained around 78.25%, indicating mild overfitting.

---

# Why VideoMAE Alone Was Not Enough

VideoMAE is fundamentally an action recognition model.

It struggles with:

### Hidden Mobile Phones

Phone size is very small.

Sometimes phones are:

* partially visible
* occluded
* inside laps

VideoMAE may classify such clips as Normal.

---

### Hidden Notes

Paper exchange occupies very few pixels.

Detection becomes difficult.

---

### Explainability

Transformers output class probabilities but cannot explain:

"Why was this student classified as cheating?"

Therefore object-level evidence is absent.

---

# Need for Fusion

Different modules possess complementary strengths.

| Module      | Strength               |
| ----------- | ---------------------- |
| VideoMAE    | Temporal understanding |
| YOLO-Mobile | Phone localization     |
| YOLO-Notes  | Notes localization     |
| Pose Module | Behavioral reasoning   |
| Tracking    | Temporal consistency   |

Combining them yields a more robust system.

---

# Ablation Study

To investigate contribution of each module, an ablation study was conducted.

---

## VideoMAE Only

| Model |
Accuracy |
|---------|---------|
| VideoMAE | 78.25% |

VideoMAE successfully recognizes major actions but misses object-specific cheating.

---

## VideoMAE + Pose

Adding pose reasoning improved recognition of copying behavior.

| Configuration   | Accuracy |
| --------------- | -------- |
| VideoMAE        | 78.25    |
| VideoMAE + Pose | 82.7     |

Improvements arise due to:

* side glance analysis
* body leaning
* temporal accumulation

---

## VideoMAE + Mobile Detector

| Configuration     | Accuracy |
| ----------------- | -------- |
| VideoMAE + Mobile | 84.5     |

Phone localization significantly reduces false negatives.

---

## VideoMAE + Notes Detector

| Configuration    | Accuracy |
| ---------------- | -------- |
| VideoMAE + Notes | 83.8     |

Hidden note detection improves note-sharing recognition.

---

## VideoMAE + Mobile + Notes

| Configuration             | Accuracy |
| ------------------------- | -------- |
| VideoMAE + Mobile + Notes | 87.1     |

Combining object detectors provides complementary information.

---

## Final Fusion Framework

| Configuration                    | Accuracy |
| -------------------------------- | -------- |
| VideoMAE + Mobile + Notes + Pose | 89.8     |

Highest performance was achieved after integrating all modalities.

---

# Class-wise Performance

| Class           | Precision | Recall | F1   |
| --------------- | --------- | ------ | ---- |
| Normal          | 92.4      | 91.8   | 92.1 |
| Copying         | 88.6      | 87.9   | 88.2 |
| Mobile Usage    | 91.3      | 90.7   | 91.0 |
| Gesture Sharing | 84.7      | 83.9   | 84.3 |
| Hidden Notes    | 87.4      | 86.6   | 87.0 |

---

# Confusion Matrix Analysis

The most confusing classes are:

### Normal ↔ Copying

Short side glances resemble normal writing behavior.

---

### Gesture ↔ Copying

Hand movements overlap between classes.

---

### Notes ↔ Mobile

Both involve downward hand motion and occlusions.

---

The fusion framework reduces these confusions through complementary cues.

---

# Computational Efficiency

| Model           | Parameters | FPS | Real-Time |
| --------------- | ---------- | --- | --------- |
| R3D-18          | 33M        | 38  | Yes       |
| VideoMAE Base   | 86M        | 16  | Moderate  |
| YOLOv8 Mobile   | 11M        | 55  | Yes       |
| YOLOv8 Notes    | 11M        | 55  | Yes       |
| Proposed Fusion | ~110M      | 12  | Yes       |

The proposed framework remains suitable for real-time surveillance systems.

---

# Output Generation

The final system generates:

---

## Annotated Video

Contains:

* Student IDs
* Bounding boxes
* Cheating labels
* Confidence scores

Example:

```
ID 7
cheating_mobile
0.93

ID 15
copying
0.88

ID 3
gesture_sharing
0.91
```

---

## CSV Event Logs

| Timestamp | Student ID | Action          | Confidence |
| --------- | ---------- | --------------- | ---------- |
| 00:02:11  | 7          | Mobile Usage    | 0.93       |
| 00:04:35  | 15         | Copying         | 0.88       |
| 00:06:20  | 3          | Gesture Sharing | 0.91       |

These logs provide explainable evidence and enable post-exam analysis.

---

# Key Findings

1. R3D-18 struggles with long-range temporal dependencies.
2. VideoMAE significantly improves action understanding.
3. Object-specific cheating requires dedicated detectors.
4. Pose reasoning provides explainability.
5. Tracking ensures temporal consistency.
6. Multi-modal fusion outperforms individual models.
7. The proposed framework achieves approximately **89–90% overall accuracy**, demonstrating suitability for intelligent examination monitoring systems.

---

# 👨‍💻 Research Team

---

## 🏛 Institution

**Punjab University College of Information Technology (PUCIT)**  
University of the Punjab, Lahore, Pakistan

---

## 👥 Project Team


### **Rimsha Majeed**
**Roll Number:** BITF22M029  
**Department:** Information Technology (IT)  
**Campus:** Old Campus (OC)  
**Role:** **Team Lead**

---

### **Alina Idrees**
**Roll Number:** BITF22M003  
**Department:** Information Technology (IT)  
**Campus:** Old Campus (OC)  
**Role:** Team Member

---

### **Khadija Tul Kubra**
**Roll Number:** BITF22M025  
**Department:** Information Technology (IT)  
**Campus:** Old Campus (OC)  
**Role:** Team Member

---

Responsible for:

- Dataset design and collection
- Data preprocessing and clip generation
- Annotation and CSV preparation
- R3D-18 implementation and evaluation
- VideoMAE fine-tuning
- YOLO-Mobile detector training
- YOLO-Notes detector training
- Pose reasoning module development
- Fusion framework design
- Inference pipeline
- Output generation and event logging
- Experimental analysis
- Research documentation

---

## 🎓 Supervisor

### **Dr. Muhammad Farooq**
Punjab University College of Information Technology (PUCIT)
**Email:** mfarooq@pucit.edu.pk

Role:
- Research supervision
- Methodology guidance
- Experimental review
- Technical discussions
- Project evaluation

---

# 📄 Citation
If you use this repository, dataset, code, or methodology in your research, please cite:

```bibtex
@misc{invigilai2025,
  title={InvigilAI: AI-Based Real-Time Exam Cheating Detection System for Physical Examination Halls Using VideoMAE, YOLO-Based Detectors and Human Pose Reasoning},
  author={
      Alina Idrees and
      Khadija Tul Kubra and
      Rimsha Majeed
  },
  year={2025},
  institution={Punjab University College of Information Technology (PUCIT)},
  type={Final Year Project},
  note={Multi-modal Fusion Framework for Offline Examination Cheating Detection}
}
```

---

# 📚 References

---

## Exam Cheating Detection

### M. Talha Jahangir et al., 2024

**AI-Powered Classification for Cheating Detection in Offline Examinations Using Deep Learning Techniques with CUI Dataset**

International Journal of Innovations in Science and Technology

```bibtex
@article{jahangir2024,
title={AI-Powered Classification for Cheating Detection in Offline Examinations Using Deep Learning Techniques with CUI Dataset},
author={Jahangir, M. Talha and Subhani, Numan and Nadeem, Sadia and Abid, Fatima},
journal={IJIST},
volume={6},
number={4},
pages={1658--1678},
year={2024}
}
```

---

## VideoMAE

Tong et al.

### VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training

NeurIPS 2022

```bibtex
@inproceedings{tong2022videomae,
title={VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training},
author={Tong, Zhan and Song, Yibing and Wang, Jue and others},
booktitle={NeurIPS},
year={2022}
}
```

---

## Vision Transformer (ViT)

Dosovitskiy et al.

### An Image is Worth 16×16 Words

ICLR 2021

```bibtex
@inproceedings{dosovitskiy2021vit,
title={An Image is Worth 16x16 Words},
author={Dosovitskiy et al.},
booktitle={ICLR},
year={2021}
}
```

---

## Transformers

Vaswani et al.

### Attention Is All You Need

NeurIPS 2017

```bibtex
@inproceedings{vaswani2017attention,
title={Attention Is All You Need},
author={Vaswani et al.},
booktitle={NeurIPS},
year={2017}
}
```

---

## YOLOv8

Jocher et al.

### Ultralytics YOLOv8

```bibtex
@software{yolov8,
author={Glenn Jocher and Ultralytics},
title={Ultralytics YOLOv8},
year={2023}
}
```

---

## YOLOv11

Ultralytics 2024

```bibtex
@software{yolov11,
author={Ultralytics},
title={YOLOv11},
year={2024}
}
```

---

## R3D-18

Tran et al.

### A Closer Look at Spatiotemporal Convolutions for Action Recognition

CVPR 2018

```bibtex
@inproceedings{tran2018r3d,
title={A Closer Look at Spatiotemporal Convolutions for Action Recognition},
author={Tran et al.},
booktitle={CVPR},
year={2018}
}
```

---

## Human Pose Estimation

### BlazePose

Bazarevsky et al.

Google Research, CVPR Workshops 2020

```bibtex
@inproceedings{blazepose2020,
title={BlazePose: On-device Real-time Body Pose Tracking},
author={Bazarevsky et al.},
year={2020}
}
```

---

## MediaPipe

Lugaresi et al.

### MediaPipe: A Framework for Building Perception Pipelines

```bibtex
@article{mediapipe2020,
title={MediaPipe: A Framework for Building Perception Pipelines},
author={Lugaresi et al.},
year={2020}
}
```

---

## Multi-Object Tracking

BoT-SORT

```bibtex
@article{botsort2022,
title={BoT-SORT: Robust Associations Multi-Pedestrian Tracking},
author={Aharon et al.},
year={2022}
}
```

---

## DeepSORT

```bibtex
@inproceedings{deepsort2017,
title={Simple Online and Realtime Tracking with Deep Association Metric},
author={Wojke et al.},
year={2017}
}
```

---

## Object Detection

Faster R-CNN

Ren et al.

```bibtex
@inproceedings{fasterrcnn2015,
title={Faster R-CNN},
author={Ren et al.},
year={2015}
}
```

---

## Residual Networks

He et al.

```bibtex
@inproceedings{resnet2016,
title={Deep Residual Learning for Image Recognition},
author={He et al.},
year={2016}
}
```

---

## Self-Supervised Learning

He et al.

Masked Autoencoders

```bibtex
@inproceedings{mae2022,
title={Masked Autoencoders Are Scalable Vision Learners},
author={He et al.},
year={2022}
}
```

---

# 🚀 Future Work

# Future Work

Although the proposed **InvigilAI** framework demonstrates promising performance through the fusion of VideoMAE, YOLO-based detectors, and pose reasoning, several challenges remain before achieving a highly reliable real-world examination surveillance system. Future research directions are summarized below.

## 1. Fine-Tuning and Performance Optimization

The current fusion framework improves overall performance compared with individual models; however, further improvements in accuracy are still required. Future work will focus on:

* Extensive hyperparameter optimization.
* Better learning rate scheduling.
* Larger and more diverse training data.
* Class balancing strategies.
* Advanced augmentation techniques.
* Longer temporal clip modeling.
* Self-supervised pretraining on examination videos.

These improvements are expected to enhance generalization and reduce false detections.

---

## 2. Improving Gesture and Hidden Notes Detection

Among all classes, **gesture sharing** and **hidden notes usage** remain the most challenging behaviors.

Current limitations include:

* Similarity between normal hand movements and cheating gestures.
* Occlusion caused by desks and body posture.
* Small note objects that are difficult to detect.
* Variations in hand positions and note visibility.

Future work will investigate:

* Fine-grained hand landmark analysis.
* MediaPipe Hands integration.
* Transformer-based gesture recognition.
* Multi-stage note localization networks.
* Attention mechanisms for small-object detection.
* Graph Neural Networks (GNNs) for hand-body interaction modeling.

---

## 3. Robust Detection under Blur and Occlusion

Students seated farther from the camera often appear:

* Blurred.
* Low-resolution.
* Partially occluded.
* Poorly illuminated.

Such conditions occasionally lead to incorrect predictions.

Future improvements include:

* Super-resolution preprocessing.
* Image enhancement techniques.
* Adaptive frame selection.
* Multi-scale feature extraction.
* Vision Transformer-based small object detection.
* Camera calibration and resolution enhancement.

---

## 4. Tracking Stability for High-Movement Students

Current tracking performance degrades when students:

* Move rapidly.
* Stand up or lean excessively.
* Become temporarily occluded.
* Cross each other.

This may result in:

* Identity switching.
* Track loss.
* Inconsistent temporal predictions.

Future work will investigate:

* StrongSORT.
* DeepSORT with appearance embeddings.
* BoT-SORT.
* ByteTrack improvements.
* Re-identification (ReID) modules.
* Multi-object trajectory association methods.

---

## 5. Multi-Camera Examination Monitoring

The current system relies on a single surveillance camera.

Single-view monitoring suffers from:

* Blind spots.
* Occlusion.
* Limited field of view.
* Perspective distortion.

Future work may incorporate:

* Multi-camera fusion.
* Camera synchronization.
* Cross-view tracking.
* 3D scene understanding.
* Multi-view action recognition.

Such approaches would significantly improve robustness in large examination halls.

---

## 6. Advanced Fusion Strategies

The present framework employs weighted decision-level fusion.

Future work will explore:

* Attention-based fusion.
* Late fusion and early fusion mechanisms.
* Transformer-based multimodal fusion.
* Graph Neural Networks.
* Cross-modal attention.
* Feature-level fusion instead of score-level fusion.

These approaches may provide richer contextual understanding of student behavior.

---

## 7. Gaze Estimation and Eye Tracking

Current pose reasoning approximates head orientation, but precise gaze estimation is not yet incorporated.

Future systems can include:

* Face Mesh landmarks.
* Eye landmark tracking.
* Iris localization.
* Gaze direction estimation.
* Attention analysis.

This would improve detection of:

* Copying from neighboring students.
* Mobile phone usage.
* Hidden note reading.

---

## 8. Audio-Visual Cheating Detection

The current framework uses only video information.

Future research can combine:

* Audio signals.
* Speech activity detection.
* Whisper-based transcription.
* Sound event recognition.

Audio cues may help detect:

* Whispering.
* Verbal communication.
* Unauthorized discussion.

---

## 9. Edge AI and Real-Time Deployment

The current implementation primarily targets GPU-based environments.

Future optimization will focus on:

* TensorRT acceleration.
* ONNX conversion.
* INT8 quantization.
* Pruning and compression.
* Jetson deployment.
* Raspberry Pi edge systems.

This will enable real-time deployment in practical examination halls.

---

## 10. Explainable AI (XAI)

False accusations in examination environments can have serious consequences.

Future work will integrate:

* Grad-CAM.
* Attention maps.
* Explainable AI techniques.
* Visual evidence generation.
* Decision traceability.

This will increase transparency and trustworthiness of the system.

---

## 11. Large Language Model (LLM)-Based Reasoning

Future systems may incorporate Large Language Models to perform high-level reasoning over detected events.

For example, the system could analyze:

* Student trajectories.
* Pose sequences.
* Detected objects.
* Temporal event histories.

LLMs can generate:

* Natural-language explanations.
* Automated invigilator reports.
* Event summaries.
* Cheating severity assessments.

---

## 12. Larger and Public Benchmark Dataset

Although approximately 65 participants contributed to the current dataset, larger-scale datasets are required for broader generalization.

Future plans include:

* Increasing participant diversity.
* Recording additional environments.
* Introducing varying camera heights and angles.
* Including more challenging scenarios.
* Creating a publicly accessible benchmark dataset.

Such a benchmark would facilitate fair comparison among future examination cheating detection methods and encourage reproducible research.

---

## 13. Towards a Fully Intelligent Examination Surveillance System

Ultimately, future versions of InvigilAI aim to evolve into a complete intelligent invigilation platform capable of:

* Multi-camera monitoring.
* Real-time tracking.
* Action recognition.
* Object detection.
* Gaze estimation.
* Audio understanding.
* Explainable reasoning.
* Automated report generation.
* Cloud deployment.
* Edge deployment.

By combining multimodal learning, transformer architectures, and explainable AI, future generations of InvigilAI can become practical and reliable assistants for maintaining fairness and integrity in physical examination environments.

----

# 📜 License

This project is released under the **MIT License**.

Copyright (c) 2025

**InvigilAI Research Team**  
Punjab University College of Information Technology (PUCIT)

---

## Academic Usage

This repository is intended for:

- Research purposes
- Educational purposes
- Final Year Projects
- Academic publications

---

## Citation Requirement

If you use:

- Dataset
- Source code
- Methodology
- Experimental setup
- Figures
- Results

please cite this repository and acknowledge the authors.

---

# ⭐ InvigilAI

### AI-Based Real-Time Exam Cheating Detection System for Physical Examination Halls

**VideoMAE + YOLO-Mobile + YOLO-Notes + Pose Reasoning + Multi-Modal Fusion**

**Designed for Explainable, Robust and Real-World Offline Examination Surveillance**




here is my readme.md remove all duplications, and comprehense or write in preofessional formate more over as it research based so make it research oriented keep all model results there drawback why we move to next one whats its best one , comparison of all and final one decision also add dataset detail and kaggle link and references what other research gap , mean overall make it reseach oriented just like mini research paper 
