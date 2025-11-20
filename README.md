# Thoracic Disease Classification

A PyTorch-based project for classifying thoracic diseases from chest X-ray images, using a multimodal deep neural network that combines images and tabular clinical data.

---

## Features

- **Multimodal Neural Network:** Combines ResNet50 (for images) and MLP (for tabular features).
- **Efficient Data Storage:** Stores resized JPEG images in LMDB for fast access.
- **Flexible Data Splitting:** Supports training with 25%, 50%, 75%, or 100% of available data.
- **Evaluation:** Computes per-class accuracy, multilabel confusion matrices, and training/validation loss curves.
- **Highly Parallel DataLoaders:** Fully utilizes multicore CPUs for rapid training.

---

## Data & Preprocessing

- **CSV Metadata:** Contains image paths, 14 disease label columns, demographics, and acquisition details.
- **Preprocessing Includes:**
  - Missing value imputation.
  - Categorical encoding.
  - Feature scaling (standardization).
  - Optional PCA on meta features.
- **Images:** Resized (e.g., to 512x512), stored as JPEG in LMDB, with directory name reflecting size.

---

## Model

- **Image Stream:** ResNet50 backbone (ImageNet weights, last layer replaced).
- **Tabular Stream:** Deep multi-layer perceptron with LayerNorm and Dropout.
- **Fusion:** Gating mechanism adaptively blends feature spaces.
- **Output:** 14-label multilabel sigmoid output with `BCEWithLogitsLoss`.

---

## Usage

### 1. Install requirements

pip install torch torchvision pandas numpy scikit-learn lmdb tqdm pillow matplotlib seaborn opencv-python

text

### 2. Ensure Large Model Files Are Ignored

Create a `.gitignore` file (if not present) containing:

*.pth

text

### 3. Create LMDB Datasets

from your_module import create_lmdb_with_resized_images, ensure_lmdb

ensure_lmdb(train_df, "lmdb_train", image_root, resize=(512, 512))

text

### 4. Training Loop Example

train_dataset = XrayLMDBDatasetWithImages("lmdb_train_512x512", transform=train_transform)
val_dataset = XrayLMDBDatasetWithImages("lmdb_val_512x512", transform=val_transform)
Dataloaders, model, optimizer, training loop as described in code

text

---

## Results

- Plots validation loss and per-class accuracy over epochs.
- Compares training time for different data fractions.
- Stores best checkpoints as `.pth` (locally; not tracked in git).

---

## Notes

- **DO NOT commit `.pth` files:** These exceed GitHub's size limit. Clean your Git history if necessary—use [BFG Repo Cleaner](https://rtyley.github.io/bfg-repo-cleaner/) if you need to purge large files.
- Hyperparameters, model choice, and image size are configurable in the codebase.
- Adapt for multi-GPU or mixed precision as needed.

---

## Acknowledgements

- Data: Stanford CheXpert project.
- Libraries: PyTorch, torchvision, scikit-learn, LMDB, and others listed above.
- Inspired by open source contributions to medical imaging ML.

---

**For questions or contributions, please open an issue.**