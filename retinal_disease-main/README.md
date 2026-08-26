# Retinal Disease Classification (ODIR-5K)

Deep learning project that classifies retinal fundus images into 8 ocular disease categories using transfer learning with ResNet50 in PyTorch.

## Dataset

This project uses the [ODIR-5K (Ocular Disease Intelligent Recognition)](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k) dataset — fundus photographs of ~5,000 patients (left and right eyes) collected by Shanggong Medical Technology Co., Ltd.

Each image is labeled with one of 8 classes:

| Label | Condition                          | Samples |
|-------|------------------------------------|---------|
| N     | Normal                             | 2,873   |
| D     | Diabetic Retinopathy               | 1,608   |
| O     | Other diseases/abnormalities       | 708     |
| C     | Cataract                           | 293     |
| G     | Glaucoma                           | 284     |
| A     | Age-related Macular Degeneration   | 266     |
| M     | Pathological Myopia                | 232     |
| H     | Hypertension                       | 128     |

Total: 6,392 labeled images. Note the strong class imbalance — Normal and Diabetic Retinopathy dominate.

> **Note:** The image folders (`ODIR-5K/` and `preprocessed_images/`) are not included in this repository due to their size. Download the dataset from Kaggle and place the folders in the project root. The metadata file `full_df.csv` is included.

## Project Structure

```
.
├── main.ipynb              # Full pipeline: EDA → preprocessing → training → evaluation
├── full_df.csv             # Image metadata and labels
├── ODIR-5K/                # Raw dataset (not tracked — download separately)
└── preprocessed_images/    # Preprocessed fundus images (not tracked)
```

## Approach

1. **EDA** — check for missing values/duplicates, inspect label distribution, visualize sample images.
2. **Preprocessing** — map images to preprocessed paths, label-encode the 8 classes.
3. **Split** — 80/20 train/test split, stratified by label (`random_state=42`).
4. **Augmentation (train only)** — resize to 224×224, random horizontal flip, random rotation (±10°), color jitter, ImageNet normalization.
5. **Model** — ResNet50 pretrained on ImageNet, final fully-connected layer replaced with an 8-class head.
6. **Training** — CrossEntropyLoss, Adam optimizer (lr = 1e-4), batch size 32, 10 epochs on GPU.
7. **Evaluation** — accuracy, per-class precision/recall/F1, confusion matrix.

## Results

| Metric              | Score  |
|---------------------|--------|
| Train accuracy      | 91.71% |
| Test accuracy       | 66.85% |
| Macro F1            | 0.59   |
| Weighted F1         | 0.66   |

Per-class performance is strongest for Pathological Myopia (F1 = 0.88) and Cataract (F1 = 0.84), and weakest for Hypertension (F1 = 0.12) — a direct consequence of class imbalance (only 128 hypertension samples). The gap between train and test accuracy also indicates overfitting.

## Planned Improvements

- [ ] Class-weighted loss to counter imbalance (weights computed, retraining pending)
- [ ] Regularization / early stopping to reduce overfitting
- [ ] Learning-rate scheduling
- [ ] Try other backbones (EfficientNet, ViT)

## Requirements

- Python 3.x
- PyTorch (CUDA build recommended) + torchvision
- pandas, numpy, scikit-learn
- matplotlib, seaborn
- Pillow

```bash
pip install torch torchvision pandas numpy scikit-learn matplotlib seaborn pillow
```

## Usage

1. Download the ODIR-5K dataset from Kaggle and extract it into the project root.
2. Open `main.ipynb` in Jupyter/VS Code.
3. Run all cells — a CUDA-capable GPU is recommended (training takes ~10 epochs on 5,113 images).
