# TransDTI: Diffusion Tensor Imaging Classification Framework

An end-to-end deep learning framework for classifying neuroimaging diffusion MRI scans into three diagnostic categories: Cognitively Normal (CN), Mild Cognitive Impairment (MCI), and Alzheimer's Disease (AD).

## Project Overview

- **Input Modality**: Diffusion Tensor Imaging (DTI) DICOM series reconstructed into multi-channel tensor maps (Fractional Anisotropy, Mean Diffusivity, and unweighted $b_0$).
- **Architecture**: Hybrid Vision Transformer featuring a 3-stage convolutional stem for localized feature representation, a patch tokenization layer, and a multi-head self-attention Transformer encoder.
- **Evaluation Strategy**: Patient-level stratified partitioning with assertion checks ensuring strict separation of subjects across train, validation, and test splits. Slice predictions are aggregated to patient decisions via probability averaging (soft voting).

---

## Directory Structure

```text
├── notebooks/              # Directory for interactive analysis and Jupyter/Kaggle notebooks
├── results/
│   ├── figures/            # Vector plots (workflows, confusion matrices, ROC/PR curves)
│   ├── logs/               # Training progression records across epochs
│   ├── metrics/            # Evaluation metrics, patient predictions, and summary statistics
│   └── checkpoints/        # Saved model weights and training checkpoints
├── src/
│   ├── config.py           # Experiment configuration parameters
│   ├── data/               # DICOM preprocessing and patient partition routines
│   ├── models/             # PyTorch implementation of the hybrid architecture
│   ├── training/           # Loss functions with label smoothing and balancing
│   └── evaluation/         # Soft-voting logic, confidence intervals, and Grad-CAM
├── requirements.txt        # Python package dependencies
├── environment.yml         # Conda environment definition
└── README.md               # Repository documentation
```

---

## Setup and Installation

### 1. Conda Environment
```bash
conda env create -f environment.yml
conda activate transdti
```

### 2. Pip Installation
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

---

## Usage

### Running the Notebook
Place your `.ipynb` notebook file inside the `notebooks/` directory. Open the notebook in your environment of choice (Kaggle or local JupyterLab):
```bash
jupyter lab notebooks/
```

### Modular Pipeline
1. **Preprocessing**: Ingest raw DICOM series, compute motion metrics, skull strip, and fit diffusion tensors using DIPY.
2. **Model Training**:
   ```bash
   python -m src.train --epochs 30 --batch_size 32 --lr 0.0003
   ```
3. **Artifact Generation**:
   ```bash
   python scripts/render_all_figures.py
   ```

---

## Summary of Results

Detailed experimental logs and metric files are available in `results/`:
- **Slice Performance**: Found in `results/metrics/slice_confusion_matrix.csv` and `classification_report.csv`.
- **Patient-Level Decisions**: Soft-voting aggregation details saved in `results/metrics/patient_predictions.csv`.
- **Visualizations**: ROC/PR curves, training trajectories, and architecture diagrams stored in `results/figures/`.

---

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
