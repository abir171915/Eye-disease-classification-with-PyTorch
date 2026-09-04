# Eye Infection Classification (Bacterial vs Fungal vs Healthy)

A deep learning classifier that distinguishes bacterial keratitis, fungal keratitis, and healthy eyes from clinical photographs, built with PyTorch and transfer learning.

## Problem

Given a photograph of a patient's eye, classify it into one of three categories:
- **Bacterial** infection
- **Fungal** infection
- **Healthy**

This is a 3-class image classification problem built on real clinical images, with the eventual goal of extending to fungal subtype classification (Aspergillus, Candida, Fusarium, Others) as a follow-up project.

## Dataset

- ~4,600 images total across three classes, sourced from patient photographs (multiple images per patient).
- Fungal subtypes (Aspergillus, Candida, Fusarium, Others) are folded into a single `fung` class for this stage; the subtype information is preserved in the folder structure for future use.
- **Patient-level train/val/test split (70/15/15)**: all images from a given patient are kept entirely within one split. This avoids data leakage — a naive random image-level split would let the same patient's eye appear in both training and test, letting the model "cheat" by recognizing the patient rather than the pathology, and inflating test accuracy artificially.
- Data is not included in this repository (see `.gitignore`) due to size and the sensitivity of clinical images.

## Approach

- **Model**: ResNet50, pretrained on ImageNet (`torchvision.models.resnet50`).
- **Transfer learning strategy**: backbone frozen except for the last residual block (`layer4`), plus a replaced classification head (`Dropout(0.5)` + `Linear`) for the 3-class problem. This lets the model adapt higher-level features to the medical domain instead of relying purely on ImageNet-level features.
- **Regularization**: dropout in the head, L2 weight decay, and a differential learning rate (`layer4` fine-tuned slowly at `1e-5`, the new head at `1e-4`) to avoid destroying pretrained weights while still adapting them.
- **Training loop**: custom `train_step` / `eval_step` / `train` functions, with automatic checkpointing (best validation loss) and early stopping (patience-based) to avoid wasting compute on overfitting epochs and to always retain the best-performing model, not just the last one.
- **Evaluation**: accuracy alone was not treated as sufficient for a medical classification problem — per-class precision/recall/F1 and a confusion matrix were used to understand *where* the model was failing, not just *how often*.

## Results

Final evaluation on the held-out test set (572 images, never used during training or model selection):

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Bacterial | 0.57 | 0.50 | 0.53 |
| Fungal | 0.72 | 0.77 | 0.74 |
| Healthy | 0.96 | 1.00 | 0.98 |

**Overall accuracy: ~72%**, macro F1: ~0.75

Healthy eyes are classified essentially perfectly. The main source of error is confusion between bacterial and fungal classes, not between either infection type and healthy.

## Key findings and what didn't work

Documenting the negative results is deliberate — most of the useful signal in this project came from ruling things out, not from one clean improvement:

- **Overfitting from unregularized fine-tuning**: unfreezing `layer4` without weight decay/dropout caused the model to memorize the training set within 1-2 epochs (train accuracy >99%, validation loss exploding). Adding weight decay, dropout, and a lower learning rate for the unfrozen layers fixed this and gave the largest single improvement in the project.
- **Higher input resolution (384×384 vs 224×224) did not help**, despite source images being very high resolution (3168×4752). This suggests the bacterial/fungal confusion is not primarily a lost-detail problem — the two classes plausibly have genuine visual overlap in a meaningful fraction of cases, which is consistent with informal visual inspection of the images.
- **Class weighting in the loss function** (`CrossEntropyLoss(weight=...)`) to counter the training set's class imbalance (fungal has ~1.85x more images than bacterial) did meaningfully improve bacterial recall (0.34 → 0.55 as weighting increased), but at the cost of bacterial precision and fungal recall — a genuine trade-off, not a free win. Macro F1 plateaued around a bacterial weight of ~1.86, with further increases trading performance between classes rather than improving it overall.

## Limitations

- Bacterial vs fungal separation (~50-55% recall for bacterial) is not reliable enough for clinical or diagnostic use — this model is a learning project and a baseline, not a deployable tool.
- Some of the bacterial/fungal misclassification likely reflects genuine visual ambiguity in the source images, which a model trained on images alone may not fully resolve without additional clinical context (e.g. lab culture results).
- Test set size (572 images) is small enough that reported metrics carry meaningful variance.

## Possible future work

- Fungal subtype classification (Aspergillus/Candida/Fusarium/Others), using the already-preserved subtype folder structure.
- Try `WeightedRandomSampler` (oversampling minority classes) instead of / combined with loss weighting.
- Try a different backbone architecture, or unfreeze additional layers with appropriately staged learning rates.

## Project structure

```
eye/
├── classification.ipynb   # Full pipeline: data loading, transforms, training, evaluation
├── Data/                   # Not tracked in git — train/val/test split by class
│   ├── train/{bact,fung,healthy}/
│   ├── val/{bact,fung,healthy}/
│   └── test/{bact,fung,healthy}/
├── best_model.pth          # Not tracked in git — best checkpoint by validation loss
└── README.md
```

## Tech stack

Python, PyTorch, torchvision, scikit-learn (evaluation metrics), matplotlib, Jupyter.
