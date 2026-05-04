# Independent Study: Intensity–Range Multimodal Pavement Distress Detection

This repository contains the final materials for my independent study project at Texas State University.

The project is based on my ACM ICMR 2026 accepted paper:

**Data-Centric Multimodal Pavement Distress Detection Using Intensity–Range Imagery**

## Project Overview

This independent study investigates multimodal pavement distress detection using paired 2D intensity images and 3D range images from the Texas Department of Transportation (TxDOT) Asphalt Concrete Pavement (ACP) dataset.

The main goal is to better use the complementary information from intensity and range imagery for pavement distress detection. Instead of simply stacking the two modalities as input channels, this project studies how different distress classes rely on different modality cues and proposes a split-backbone YOLOv5s model with class- and modality-aware augmentation.

## Main Objectives

1. Analyze class-wise modality preference between intensity and range.
2. Design a split-backbone YOLOv5s model to preserve modality-specific features.
3. Apply class- and modality-aware augmentation for paired 2D–3D inputs.
4. Evaluate the proposed method against early-fusion and single-modality baselines.

## Repository Structure

```text
independent-study-icmr-2026/
│
├── presentation/
│   └── independent_study_presentation.pptx
│
├── report/
│   └── project-final.md
│
├── code/
│   ├── data_pipeline/
│   ├── models/
│   ├── augmentation/
│   └── training_configs/
│
└── README.md
````

## Method Summary

The proposed workflow includes:

* paired 2D intensity and 3D range input;
* YOLO-style shared geometric augmentation to preserve 2D–3D alignment;
* customized pixel-level augmentation applied separately to intensity and range;
* split-backbone YOLOv5s using ConvSplit, C3Split, and SPPFSplit modules;
* late fusion in the neck and head for final pavement distress detection.

## Experimental Setup

The study compares multiple YOLOv5s-based settings:

* Intensity-only baseline;
* Range-only baseline;
* Early-fusion baseline;
* Split-backbone YOLOv5s;
* Split-backbone YOLOv5s with uniform augmentation;
* Split-backbone YOLOv5s with designed class- and modality-aware augmentation.

The main evaluation metrics are:

* AP@0.5;
* AP@0.5:0.95;
* precision;
* recall.

## Key Findings

The experimental results show that:

* intensity and range provide complementary but class-dependent evidence;
* simple early fusion can mix modality signals too early;
* the split-backbone design preserves modality-specific features before fusion;
* modality-aware augmentation further improves robustness;
* the best performance comes from combining architecture design and data-centric augmentation.

## Data Availability

The TxDOT ACP dataset is not included in this repository because it is project-restricted and not publicly released.

This repository only contains presentation materials, report files, and project-related code/configuration files.

## Final Materials

* Final presentation: `presentation/independent_study_presentation.pptx`
* Final report: `report/project-final.md`

## Author

**Wenhan Tao**
PhD Student, Department of Computer Science
Texas State University

## Advisors

* Dr. Jelena Tešić
* Dr. Feng Wang
* Dr. Yongsheng Bai

