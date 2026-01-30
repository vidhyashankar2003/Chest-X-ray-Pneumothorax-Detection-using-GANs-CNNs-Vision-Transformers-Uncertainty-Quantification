# Chest X-ray Pneumothorax Detection using GANs, CNNs, Vision Transformers & Uncertainty Quantification

This project presents a **comprehensive deep learning pipeline for Pneumothorax detection in chest X-rays**, combining **data generation (CycleGAN)**, **multiple classification models (CNN, Multi-task CNN, Vision Transformer, Ensemble)**, and **uncertainty quantification** to improve robustness and clinical reliability.

The work focuses not only on classification accuracy, but also on:

* **Synthetic data realism**
* **Model comparison and ensembling**
* **Explainability and uncertainty awareness**, which are critical for medical AI deployment.

---

## Key Contributions

* **CycleGAN-based image generation** for Pneumothorax ↔ No Finding domain translation
* **Balanced and domain-aware dataset splitting**
* **Multiple classifiers**:

  * Simple CNN (ResNet18 backbone)
  * Multi-task CNN (disease + image quality)
  * Vision Transformer (ViT)
  * Ensemble of CNN architectures
* **Real vs Synthetic image discrimination**
* **Monte Carlo Dropout–based uncertainty quantification**
* **Extensive evaluation, visualization, and comparison tools**

---

## Project Structure

```
.
├── data/
│   └── chestxray/
│       ├── images/
│       ├── splits/
│       │   └── pneumothorax/
│       └── Data_Entry_2017_v2020.csv
│
├── checkpoints/
│   ├── vit/
│   ├── cnn/
│   ├── ensemble/
│   └── CycleGAN checkpoints
│
├── generated_images/
│   └── synthetic_pneumothorax/
│
├── visualizations/
│   ├── metrics/
│   └── uncertainty/
│
├── dataset & dataloaders
│   ├── dataset_v2.py
│   ├── pneumo_dataset_v2.py
│   ├── xraydataset_v2.py
│   ├── dataloader_v2.py
│   └── preprocessor_v2.py
│
├── models
│   ├── networks_v2.py          # CycleGAN Generator & Discriminator
│   ├── generator_v2.py         # Image generation
│   ├── multitask_cnn.py
│   ├── train_simple_cnn.py
│   ├── train_real_vs_fake_cnn.py
│   ├── vit_model_v2.py
│   └── ensemble_learning.py
│
├── training
│   └── train_v2.py              # CycleGAN training
│
├── evaluation & analysis
│   ├── compare_models.py
│   ├── three_way_comparison.py
│   ├── retrieve_metrics.py
│   ├── visualize_metrics.py
│   ├── ensemble_viz.py
│   ├── gradcam_visualization.py
│   └── uncertainty_quantification.py
│
└── README.md
```

---

## Models Overview

### 1. CycleGAN (Image Generation)

* **Unpaired domain translation**
* Domain A: No Finding
* Domain B: Pneumothorax
* Used to generate realistic synthetic Pneumothorax X-rays for augmentation

### 2. CNN-Based Classifiers

* **Simple CNN** (ResNet18 backbone)
* **Multi-task CNN**:

  * Pneumothorax classification
  * Image quality estimation

### 3. Vision Transformer (ViT)

* Patch-based transformer architecture
* Binary classification using CLS token
* Optimized for medical image resolution

### 4. Ensemble Learning

* Combines:

  * ResNet18
  * DenseNet121
  * EfficientNet-B0
* Weighted probability fusion for improved robustness

### 5. Real vs Synthetic Classifier

* Evaluates **synthetic image realism**
* High accuracy → synthetic images are easily detectable
* Lower accuracy → better augmentation quality

---

## Dataset Preparation

### Balanced Splits

Scripts provided to create:

* Balanced Pneumothorax vs Healthy sets
* Small subsets for fast experimentation
* Domain-specific splits for CycleGAN

Key scripts:

* `create_balanced_split.py`
* `create_balanced_val.py`
* `build_domain_splits_v2.py`
* `create_small_splits.py`

---

## Training

### CycleGAN

```bash
python train_v2.py \
  --epochs 15 \
  --batch_size 8 \
  --size 256 \
  --use_amp
```

### CNN (Pneumothorax Detection)

```bash
python train_simple_cnn.py \
  --csv Data_Entry_2017_v2020.csv \
  --images images/ \
  --train_list train_BALANCED.txt \
  --val_list val_BALANCED.txt
```

### Vision Transformer

```bash
python train_vit.py
```

### Real vs Synthetic Detection

```bash
python train_real_vs_fake_cnn.py \
  --real_dir images/ \
  --synthetic_dir generated_images/synthetic_pneumothorax
```

---

## Evaluation & Comparison

* **ROC curves**
* **Confusion matrices**
* **Precision / Recall / F1 / AUC**
* **Threshold analysis**
* **Training efficiency comparison**
* **Confidence distribution analysis**

Key scripts:

* `three_way_comparison.py`
* `compare_models.py`
* `retrieve_metrics.py`
* `visualize_metrics.py`

---

## Uncertainty Quantification (Clinical Safety)

* Uses **Monte Carlo Dropout**
* Runs multiple stochastic forward passes
* Outputs:

  * Mean prediction
  * Uncertainty (standard deviation)
  * Confidence scores
* Identifies cases where the model is unsure

```bash
python uncertainty_quantification.py \
  --checkpoint cnn_best.pth \
  --n_forward_passes 30
```

Outputs:

* Uncertainty heatmaps
* Confidence vs correctness plots
* Individual high-risk case visualizations

---

## Explainability

* **Grad-CAM visualizations**
* Highlights regions influencing predictions
* Useful for clinical interpretation

---

## Notes & Limitations

* Synthetic data quality must be validated using the Real vs Synthetic classifier
* CycleGAN does not provide validation metrics by design
* Not intended for direct clinical deployment without further validation

---

##  License

This project is intended for **academic and research use** only.

---

## 👨‍💻Author

**VidhyaShankara Bharathi**
Deep Learning | Medical Imaging | GANs | Vision Transformers

---

⭐ If this project helped you, consider starring the repository.
