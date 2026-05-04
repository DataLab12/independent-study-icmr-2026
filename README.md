# Independent Study Final Report
## Data-Centric Multimodal Pavement Distress Detection Using Intensity–Range Imagery

**Wenhan Tao**  
PhD Student, Department of Computer Science  
Texas State University

**Advisors:** Dr. Jelena Tešić, Dr. Feng Wang, Dr. Yongsheng Bai  
**Presentation Date:** May 4, 2026

---

## Problem Statement

Large-scale pavement condition monitoring requires accurate and efficient pavement distress detection. Modern pavement inspection vehicles can collect paired 2D intensity images and 3D range images from the same road surface. The 2D intensity modality captures appearance, texture, and contrast, while the 3D range modality captures surface geometry and depth-related variation.

A central challenge is that the usefulness of each modality is class-dependent. Some pavement distress classes are easier to detect from appearance cues, while others depend more strongly on geometric cues. Standard early-fusion methods simply stack intensity and range channels at the input level and allow the model to mix them from the first convolution layer. This strategy is simple, but it may entangle weak and strong modality signals too early and suppress class-critical information.

The goal of this independent study is to better use the complementary information from 2D intensity and 3D range imagery for pavement distress detection. The study focuses on four main objectives:

1. Analyze modality characteristics and class-wise modality preference.
2. Design a split-backbone YOLOv5s model that preserves modality-specific features.
3. Design a class- and modality-aware data augmentation module.
4. Evaluate the proposed method against early-fusion and single-modality baselines.

---

## Dataset

### Where does the data come from?

The data comes from the **Texas Department of Transportation (TxDOT) 2D/3D Asphalt Concrete Pavement (ACP) dataset**. It was collected by a pavement inspection vehicle for large-scale roadway condition monitoring.

Each pavement sample contains two spatially aligned modalities:

- **2D intensity image:** camera-based surface appearance, texture, and contrast.
- **3D range image:** laser-based surface geometry and depth variation.

The original pavement data is collected along continuous road sections and then tiled into image patches for detection annotation and model training. In this study, the original image tile size is **1536 × 900**, and the YOLO training input size is **640 × 640**.

### Dataset characteristics

The TxDOT ACP subset used in this study contains nine pavement distress classes. The class distribution is highly imbalanced, which makes rare-class detection more difficult.

| Class | # Images | # Instances |
|---|---:|---:|
| Transverse | 1,183 | 1,673 |
| Joint | 369 | 488 |
| Sealed transverse | 1,506 | 2,101 |
| Longitudinal | 1,369 | 1,853 |
| Lane longitudinal | 1,383 | 1,487 |
| Sealed longitudinal | 2,411 | 4,025 |
| Block | 293 | 296 |
| Alligator | 678 | 878 |
| Pothole | 115 | 154 |
| **Total** | **6,936** | **12,945** |

The most frequent class is **sealed longitudinal**, with 4,025 instances. Rare classes such as **pothole** and **block** have far fewer examples. Therefore, in addition to multimodal fusion, class imbalance is also an important challenge in this dataset.

### Train / validation / test split

The ACP subset is divided into train, validation, and test sets. The held-out test set is used only for final evaluation.

| Split | # Images | # Instances |
|---|---:|---:|
| Train | 4,853 | 8,842 |
| Validation | 1,041 | 2,043 |
| Test | 1,042 | 2,060 |

---

## Related Work

Pavement distress detection has been widely studied using deep learning detectors and segmentation models. YOLO-style object detectors are commonly used because they provide efficient detection and are practical for large-scale pavement image analysis. However, many existing approaches focus mainly on 2D intensity images and texture-based appearance cues.

Paired 2D/3D pavement sensing provides an opportunity to use complementary information. Intensity images can highlight surface appearance and sealed crack texture, while range images can highlight depth or geometric surface variation. However, simply adding a second modality does not automatically solve the problem. Naive early fusion can mix modalities too early and may reduce class-specific evidence when one modality is weak or noisy.

This project is positioned around three ideas:

1. **Class-wise modality preference:** different distress classes may rely on intensity or range differently.
2. **Split-backbone learning:** intensity and range features should be preserved separately before fusion.
3. **Data-centric augmentation:** geometric transformations should preserve 2D/3D alignment, while pixel-level augmentation should respect the physical meaning of each modality.

---

## Data Management and Pre-processing

### Data representation

Standard YOLO pipelines are commonly built around three-channel image loading. To make intensity–range data compatible with this image-loading pipeline, each sample is first stored as a fused three-channel image:

| Channel | Meaning |
|---:|---|
| Channel 0 | Intensity image |
| Channel 1 | Range image |
| Channel 2 | Constant dummy channel, value 128 |

This fused image is not a normal RGB image. The third channel is only a placeholder for compatibility with standard three-channel image handling.

### Dual-channel data pipeline

The data pipeline follows these steps:

1. Load the fused three-channel image.
2. Apply YOLO-style geometric augmentation to the fused image.
3. Select channel 0 and channel 1.
4. Remove the dummy channel.
5. Apply customized pixel-level augmentation separately to intensity and range.
6. Concatenate intensity and range into a **640 × 640 × 2** tensor for the split-backbone model.

Geometric augmentation must be shared between intensity and range because the two modalities come from the same pavement location and are spatially aligned. Shared operations such as scaling, flipping, translation, and mosaic preserve alignment between the two modalities and the detection labels.

Pixel-level augmentation is applied separately because intensity and range have different physical meanings:

- **Intensity-side augmentation:** brightness, contrast, gamma, CLAHE.
- **Range-side augmentation:** Gaussian noise, blur, multiplicative noise, gamma.

The augmentation strength is also class-aware. I first compare intensity-only and range-only validation results to identify which modality is more useful for each distress class. For range-dominant classes, range-side augmentation can be stronger while intensity augmentation is kept lighter. For intensity-dominant classes, the opposite strategy is used. The purpose is to improve robustness without damaging the weaker but still useful modality.

---

## Proposed Method

### Baseline: Early Fusion

The early-fusion baseline directly stacks intensity and range channels as input and sends them into a standard YOLOv5s detector. This is simple and deployment-friendly, but the two modalities are mixed from the first convolution layer.

This can be problematic because different pavement distress classes depend on different modality cues. If one modality is weak for a class, early fusion can mix weak and strong signals too early and suppress useful evidence.

### Split-Backbone YOLOv5s

The proposed model modifies only the YOLOv5s backbone. The neck and detection head remain the same as standard YOLOv5s.

In the backbone, the original modules are replaced with split versions:

| Original YOLOv5s Block | Split Version |
|---|---|
| Conv | ConvSplit |
| C3 | C3Split |
| SPPF | SPPFSplit |

The core operation is **ConvSplit**, which uses grouped convolution with `groups = 2`. In a standard convolution, `groups = 1`, so all input channels are mixed together. In ConvSplit, the feature channels are divided into two groups corresponding to intensity and range. This keeps the two modality streams separated during backbone feature extraction.

Based on ConvSplit:

- **C3Split** keeps the original C3 macro-structure but replaces internal convolution layers with ConvSplit.
- **SPPFSplit** keeps the original spatial pyramid pooling structure but replaces convolution projections with ConvSplit.

This design preserves the YOLOv5s architecture while delaying modality fusion until after backbone feature extraction.

### Modality-Aware Augmentation

The augmentation module is designed around two principles:

1. **Shared geometry:** geometric transformations must be synchronized across intensity and range.
2. **Separated pixel augmentation:** pixel-level perturbations should be modality-specific and class-aware.

This design allows the model to learn robust representations while respecting the physical meaning of the two modalities.

---

## Experimental Setup

| Item | Setting |
|---|---|
| Task | ACP pavement distress detection |
| Dataset | TxDOT intensity–range ACP dataset |
| Model family | YOLOv5s variants |
| Baselines | Intensity-only, Range-only, Early Fusion |
| Proposed method | Split-backbone YOLOv5s + designed augmentation |
| Input size | 640 × 640 |
| Epochs | 250 |
| Metrics | AP@0.5 / AP@0.5:0.95 |

The experiments compare early-fusion, single-modality, split-backbone, and augmentation variants under the same general YOLOv5s-based setting. This controlled comparison helps verify whether performance gains come from the fusion strategy and augmentation design rather than from unrelated changes in input size or evaluation protocol.

---

## Results

### Main quantitative results on held-out test set

The proposed method remains effective on the held-out TxDOT ACP test set. The comparison shows that the improvement is not simply caused by reducing the input from three channels to two channels. Early Fusion 2ch performs worse than Early Fusion 3ch, while Split Backbone 2ch improves over both early-fusion settings. This suggests that the gain comes from the fusion strategy, not just from the input channel number.

| Method | AP@0.5 | AP@0.5:0.95 | Precision | Recall |
|---|---:|---:|---:|---:|
| Early Fusion 3ch | 0.775 | 0.391 | 0.781 | 0.724 |
| Early Fusion 2ch | 0.736 | 0.367 | 0.782 | 0.667 |
| Split Backbone 2ch | 0.799 | 0.402 | 0.781 | 0.754 |
| Split Backbone + Uniform Aug | 0.811 | 0.412 | 0.775 | 0.763 |
| **Split Backbone + Designed Aug** | **0.833** | **0.428** | **0.770** | **0.785** |

The final model, Split Backbone + Designed Augmentation, achieves the best test performance. Recall also improves compared with the early-fusion baseline, indicating fewer missed distress detections.

### Ablation study on validation set

The validation ablation study further confirms that both the split-backbone architecture and augmentation design contribute to the performance improvement.

| Setting | AP@0.5 | AP@0.5:0.95 |
|---|---:|---:|
| Early Fusion | 0.766 | 0.396 |
| Intensity only | 0.644 | 0.332 |
| Range only | 0.737 | 0.368 |
| Split Backbone | 0.812 | 0.416 |
| Split Backbone + Uniform Aug | 0.846 | 0.434 |
| **Split Backbone + Designed Aug** | **0.869** | **0.446** |

The single-modality models are weaker than the proposed multimodal model, showing that intensity and range provide complementary information. The split-backbone model improves over early fusion, and designed augmentation further improves the result.

### Class-wise modality preference

Class-wise comparison shows that modality usefulness is not uniform across classes. For example, transverse and longitudinal cracks perform much better with range, while sealed transverse and sealed longitudinal are stronger with intensity.

Early fusion can improve some classes by combining modalities, but it is not always better than the best single modality. For sealed longitudinal and pothole, early fusion can perform worse than intensity-only. This suggests that early mixing may damage useful modality-specific information.

In contrast, the split-backbone model is usually better than, or at least close to, the best single-modality result. With designed augmentation, most classes improve further. This supports the central idea that different classes rely on different modality cues, and preserving modality-specific features is important.

### Qualitative results

Qualitative examples from the test set show three representative outcomes:

1. **Recovered detection:** early fusion misses a sealed longitudinal crack, while the proposed model detects it.
2. **False-positive suppression:** early fusion produces an incorrect detection, while the proposed model suppresses it.
3. **Remaining limitation:** both early fusion and the proposed model still struggle with pothole cases.

These examples are consistent with the quantitative results. The proposed method can recover some missed detections and reduce some false positives, but rare and ambiguous classes such as pothole remain challenging.

---

## Discussion: Key Findings

The experiments lead to several main findings:

- Early fusion is simple, but it can mix modality signals too early.
- Intensity and range provide complementary but class-dependent evidence.
- Split-backbone learning preserves modality-specific features before fusion.
- Modality-aware augmentation respects the physical meaning of each channel.
- The best performance comes from combining architecture design and data-centric augmentation.

Overall, this study shows that multimodal learning is not simply a matter of stacking input channels. For intensity–range pavement data, the model and training pipeline should reflect the different physical meanings and class-wise usefulness of each modality.

---

## What I Learned from This Independent Study

This independent study helped me understand several research lessons:

1. **Multimodal learning requires more than simple channel stacking.**  
   Intensity and range are aligned, but they represent different physical information.

2. **Data preprocessing and augmentation are part of the research design.**  
   Geometric augmentation should keep modalities aligned, while pixel-level augmentation should be modality-specific.

3. **Controlled ablation studies are necessary.**  
   By comparing intensity-only, range-only, early fusion, split backbone, and different augmentation settings, I can better verify where improvements come from.

4. **Model complexity and real inference speed do not always follow the same trend.**  
   Hardware efficiency, grouped convolution, and implementation details can affect actual latency.

5. **The independent study developed into a complete research contribution.**  
   This work was developed into an ACM ICMR 2026 accepted paper.

---

## Limitations and Future Work

This study has several limitations:

- The experiments are based on one TxDOT ACP dataset and one intensity–range sensing setup.
- Generalization to other pavement types, regions, and sensors still needs validation.
- The current model focuses on frame-level detection and does not use temporal or multi-view context.
- Rare classes such as pothole remain difficult because of limited samples and visual ambiguity.

Future work will focus on:

- Evaluating the method on additional pavement datasets, such as JCP and CRCP.
- Incorporating temporal or multi-view information from continuous pavement survey data.
- Improving rare-class robustness through better data balancing or targeted augmentation.
- Extending the split-backbone and modality-aware augmentation design to other paired sensing modalities.

---

## Conclusion

This independent study investigated multimodal pavement distress detection using paired 2D intensity and 3D range imagery.

The main finding is that early fusion is limited because different distress classes rely on different modality cues. To address this problem, I used a split-backbone YOLOv5s model that preserves modality-specific features before late fusion. I also applied class- and modality-aware augmentation to improve robustness while respecting the physical meaning of intensity and range.

The proposed method achieved the best overall performance on the TxDOT ACP dataset, demonstrating that data-centric multimodal design can improve pavement distress detection.

---

## Repository Contents

The final repository should include:

```text
Independent-Study-ICMR-2026/
├── presentation/
│   ├── independent_study_presentation.pptx
│   └── independent_study_presentation.pdf
├── report/
│   └── project-final.md
├── code/
│   ├── data_pipeline/
│   ├── models/
│   ├── augmentation/
│   └── training_configs/
└── README.md
```

---

## References

Wenhan Tao, Yongsheng Bai, Feng Wang, and Jelena Tešić. 2026. *Data-Centric Multimodal Pavement Distress Detection Using Intensity–Range Imagery*. ACM International Conference on Multimedia Retrieval (ICMR 2026).

