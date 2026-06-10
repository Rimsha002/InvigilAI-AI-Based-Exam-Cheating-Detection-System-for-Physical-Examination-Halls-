# InvigilAI

**AI-Based Exam Cheating Detection System for Physical Examination Halls**

This repository contains the research, dataset, models, and code for InvigilAI — a multi-modal framework that combines transformer-based video action recognition, object detection, and pose reasoning to detect and explain multiple offline examination cheating behaviors.

## Abstract

We present InvigilAI, a multi-modal system for detecting five classes of student behavior in classroom surveillance videos: Normal, Copying, Mobile Usage, Gesture Sharing, and Hidden Notes. We built a novel video dataset (3,100 clips, ~65 participants) and evaluated several modeling paradigms: (1) 3D-CNN (R3D-18), (2) pose-based rule engine (YOLO+rules), (3) transformer-based VideoMAE, and (4) a final multi-modal fusion combining VideoMAE, YOLO detectors (phone & notes), and pose reasoning. We document stepwise experiments, failure modes of each approach, and how each informed the next design choice. The final fusion model achieves ~89–90% accuracy on validation data and produces explainable evidence (annotated video + CSV logs).

---

## 1. Introduction

Academic integrity in offline examinations remains critical yet difficult to scale with human invigilation. Cheating behaviors are temporal and multimodal (gaze, hand motion, small objects), so robust detection requires temporal modeling, small-object detection, tracking, and explainable reasoning. InvigilAI explores these components in a research-driven pipeline: dataset creation → baseline models → model analysis → multi-modal fusion → evaluation and ablation.

---

## 2. Related Work (brief)

- 3D CNNs (R3D) for action recognition: capture local spatio-temporal features but struggle with long-range dependencies.
- Pose-based rule systems: explainable but brittle and not comprehensive for object-based cheating.
- Transformer-based video models (VideoMAE): strong temporal/contextual modeling but weak at small-object localization.
- YOLO family: state-of-the-art real-time object detectors for phones/notes and pose estimation (YOLO-pose variants).

Our contribution is an empirical, stepwise comparison and a fusion architecture that addresses the complementary weaknesses.

---

## 3. Dataset

- Total processed clips: 3,100 (classes: Normal 964, Copying 807, Mobile 569, Gesture 322, Hidden Notes 438)
- Raw videos: 637; Participants: ~65
- Environments: V1 (controlled) → V4 (realistic classroom) → Final (theater-style)
- Preprocessing: person detection → multi-object tracking → student-centric cropping → clip generation (16 frames) → manual annotation → QA
- Challenges: occlusion, motion blur, small objects, inter-class similarity, scale & viewpoint variation

Dataset is available to collaborators (Kaggle link placeholder in repo). See dataset/ for structure and CSV annotations.

---

## 4. Methods — stepwise experiments and analysis

This section is written like a mini research narrative: for each model we describe objective, experimental setup, quantitative result, qualitative failure modes, and the decision that led to the next model.

### 4.1 Baseline A: R3D-18 (RGB 3D CNN)

- Objective: Evaluate a compact spatio-temporal CNN on student-centric clips to establish a baseline for action recognition.
- Setup:
  - Input: 16 frames, 112×112 (or 224×224 in some runs)
  - Pretraining: Kinetics weights where available
  - Training: Adam, LR 1e-3, batch size 8, 30 epochs
- Result: Validation accuracy ~63%
- Observed failure modes:
  - Limited modeling of long-range or periodic behaviors (e.g., repeated side glances interleaved with writing)
  - Small-object events (phones, notes) are not reliably detected; model confuses downward gaze with mobile use
  - Occlusion and scale variation reduce performance
- Conclusion: R3D-18 is useful as an initial baseline but insufficient for multi-class cheating detection. Motivates exploring pose reasoning for explainability and transformers for long-range temporal modeling.

### 4.2 Baseline B: YOLO Pose + Rule Engine (Pose-based reasoning)

- Objective: Build an explainable module that uses pose keypoints to detect posture/gaze-based cheating (copying).
- Setup:
  - Pose model: YOLOv8n-pose / MediaPipe landmarks
  - Rules: side glance (nose vs face center displacement), leaning (shoulder height difference), reaching (wrist-torso distance), head turn thresholds
  - Temporal accumulation: exponential smoothing Score_t = 0.9·Score_{t-1} + CurrentBehavior
- Result: Binary detection (Copying vs Normal) accuracy improved in some scenarios (~67–69% when used standalone for copying detection)
- Observed failure modes:
  - Hard thresholds sensitive to camera viewpoint and subject variability
  - Cannot detect object-specific cheating (phones/notes) or fine-grained gestures reliably
  - Limited class coverage (only copying vs normal)
- Conclusion: Pose rules provide explainability and help reduce false positives for copying, but require complementary object detectors and a stronger temporal model.

### 4.3 Baseline C: VideoMAE (Transformer-based action recognition)

- Objective: Use VideoMAE to capture long-range temporal patterns and global context in clips.
- Setup:
  - Model: VideoMAE base pretrained (fine-tuned on our dataset)
  - Input: 16 frames, 224×224
  - Optimizer: AdamW, LR 1e-5, epochs 20
- Result: Validation accuracy ~78.25% (large improvement over R3D)
- Observed failure modes:
  - Misses small-object evidence: hidden phones/notes sometimes not recognized as separate evidence; model relies on contextual patterns which can be ambiguous
  - Lacks explicit localization/explainability for object detections
  - Occasional overfitting signs (training accuracy very high)
- Conclusion: VideoMAE captures temporal dependencies but needs object-level detectors and pose-based cues for reliable multi-class decisions and explainability.

### 4.4 Final: Multi-Modal Fusion (VideoMAE + YOLO-Mobile + YOLO-Notes + Pose)

- Architecture: parallel modules (VideoMAE action classifier; YOLOv8m detectors for phone and notes; YOLO-pose for keypoints). Outputs fused via weighted score-level fusion and rule-based priorities.
- Fusion equation: P_fusion = w1·P_VideoMAE + w2·P_Mobile + w3·P_Notes + w4·P_Pose, with w1=0.50, w2=0.20, w3=0.15, w4=0.15 (tuned via validation)
- Priority rules: if Mobile confidence > 0.85 and gaze directed at phone → Mobile label strongly favored; if Notes confidence > 0.80 → Notes label favored
- Temporal smoothing: majority voting over sliding window (10 frames)
- Result: Final accuracy ~89.8% (validation); precision/recall per class reported in Results section
- Advantages:
  - Combines temporal reasoning (VideoMAE) with precise small-object localization (YOLO detectors) and explainable pose cues
  - Produces evidence (annotated bounding boxes and CSV logs)
- Remaining failure cases:
  - Severe occlusion where an object is fully hidden inside clothing remains hard
  - Similarity between gesture-sharing and normal hand movements still causes confusion

---

## 5. Experimental Setup

- Hardware: NVIDIA T4 (Google Colab Pro) or equivalent
- Software: Python 3.10, PyTorch, Ultralytics YOLOv8, Transformers, OpenCV, MediaPipe (optional)
- Evaluation: standard metrics (Accuracy, Precision, Recall, F1) and class-wise confusion matrices
- Training commands and hyperparameters are in the repository scripts (train_videomae.py, training scripts for YOLO detectors)

---

## 6. Results (consolidated)

### 6.1 Overall Performance

| Model | Accuracy |
|-------|----------|
| R3D-18 | 63.0% |
| YOLO Pose + Rules | ~67–69% (copying vs normal) |
| VideoMAE | 78.25% |
| Final Fusion | 89.8% |

### 6.2 Class-wise (Final Fusion)

| Class | Precision | Recall | F1 |
|-------|-----------:|-------:|---:|
| Normal | 92.4% | 91.8% | 92.1% |
| Copying | 88.6% | 87.9% | 88.2% |
| Mobile Usage | 91.3% | 90.7% | 91.0% |
| Gesture Sharing | 84.7% | 83.9% | 84.3% |
| Hidden Notes | 87.4% | 86.6% | 87.0% |

### 6.3 Ablation Summary

- VideoMAE Only: 78.25%
- VideoMAE + Pose: ~82.7% (+4.45)
- VideoMAE + Mobile Detector: ~84.5% (+6.25)
- VideoMAE + Notes Detector: ~87.1% (+8.85)
- Complete Fusion: ~89.8% (+11.55)

Observations: each complementary module contributes measurable gains, largest from object detectors.

---

## 7. Discussion

This research-style narrative shows the iterative design: starting from a compact 3D-CNN baseline, we identified long-range dependency and small-object detection as key failure modes. Pose rules added explainability but were insufficient alone. VideoMAE resolved temporal modeling but lacked object localization. Fusion of complementary modules led to substantial performance improvements and practical evidence generation. The remaining errors motivate future directions: improved small-object detection under occlusion, hand / finger-level gesture models (MediaPipe Hands), multi-view fusion, and improved tracking to prevent ID switches.

---

## 8. Reproducibility & How to Run

See the repository root scripts and notebooks. High-level steps:
1. Prepare dataset and annotations under `dataset/` following the CSV schema.
2. Train YOLO detectors (mobile, notes) using provided YAML configs.
3. Fine-tune VideoMAE on training set.
4. Run `fusion_inference.py` with pretrained weights to generate annotated outputs and `events.csv`.

Example inference command:

```bash
python fusion_inference.py --input_video ./sample.mp4 --output_dir ./outputs --gpu 0
```

Outputs: annotated video, per-frame CSV with timestamps, per-student event logs.

---

## 9. Limitations & Ethical Considerations

- Privacy: deployment in real institutions requires ethical approval, privacy-preserving measures, and human oversight. The system should be an aid for invigilators, not an autonomous judge.
- Bias & generalization: dataset recorded in limited institutions; cross-institution validation is needed.
- Legal/Policy: follow local laws and institutional policies for camera use and automated monitoring.

---

## 10. Future Work (research directions)

- Multi-camera synchronization & cross-view fusion
- Hand/finger-level gesture recognition (MediaPipe Hands)
- Gaze estimation and eye-tracking integration
- Transformer-based multimodal fusion (feature-level attention)
- Edge deployment (TensorRT, quantization)
- Public benchmark release and standardized evaluation

---

## 11. Installation & Repository Structure

Refer to the top-level README sections (Installation & Usage). Main scripts:
- `train_videomae.py`, `train_yolo_mobile.py`, `train_yolo_notes.py`
- `fusion_inference.py` (end-to-end pipeline)
- `notebooks/` with experiments and analysis

---

## 12. Team, Citation, and License

Project team, supervisor, citation format, and MIT license are included in the repository header and footer.

---

If you would like, I can now:
- Update the README in the repository to this research-oriented version, committing the change; or
- Produce a separate `PAPER.md` or `MANUSCRIPT.md` formatted as a mini research paper (IMRaD style) that you can use for submission.

Which would you prefer?