<div align="center">

# BREAST CANCER DIAGNOSIS USING MACHINE LEARNING

### Intelligent Classification of Benign and Malignant Tumors

**Intern ID: CITS7306**

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Classification-orange)]()
[![Dataset](https://img.shields.io/badge/Dataset-Wisconsin%20Diagnostic%20Breast%20Cancer-blue)]()
[![Status](https://img.shields.io/badge/Status-Completed-success)]()

</div>

---

## Overview

Breast cancer diagnosis is a major application area of machine learning in healthcare and bioinformatics. This project uses supervised machine-learning techniques to classify breast tumors as **benign** or **malignant** from cell-nuclei measurements.

The main goal is to build and compare classification models that can support early-stage diagnostic analysis. Early detection can help healthcare professionals make faster, better-informed decisions; however, this project is strictly an educational machine-learning implementation and **must not be used as a replacement for clinical diagnosis or medical advice**.

This project evaluates two widely used classifiers:

- **K-Nearest Neighbors (KNN)**
- **Support Vector Machine (SVM)**

It also explores the effect of **Principal Component Analysis (PCA)** on classification performance.

---

## Problem Statement

Diagnosing breast cancer early can be challenging because initial symptoms may be absent or difficult to recognize. Medical datasets contain several measurements that can help distinguish between benign and malignant tumors, but analyzing many variables manually is difficult.

This project applies machine learning to identify meaningful patterns in diagnostic data and predict the tumor category. The classifier learns from previously labelled samples and then predicts the label for unseen samples.

**Target classes:**

| Class | Meaning |
|---|---|
| `B` | Benign tumor |
| `M` | Malignant tumor |

---

## Dataset

The project uses the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset, originally collected from the University of Wisconsin Hospital. It contains measurements computed from digitized images of fine-needle aspirate (FNA) samples of breast masses.

The dataset contains **569 samples**, **30 real-valued input features**, and two diagnostic classes: 357 benign and 212 malignant cases.

### Feature Examples

The model uses properties of cell nuclei, including:

- Radius
- Texture
- Perimeter
- Area
- Smoothness
- Compactness
- Concavity
- Concave points
- Symmetry
- Fractal dimension

Each of these measurements is represented using mean, standard-error, and worst-value statistics, resulting in 30 numerical features.

> Dataset source: [UCI Machine Learning Repository — Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)

---

## Machine Learning Models

### K-Nearest Neighbors (KNN)

K-Nearest Neighbors is a simple, non-parametric, supervised-learning algorithm. It classifies an unseen sample by examining the labels of its nearest data points in the feature space.

For a selected value of \(K\), the predicted class is decided by majority voting among the \(K\) nearest neighbors.

For example, when `K = 9`, the algorithm finds the nine most similar training samples. If most belong to the malignant class, the new sample is predicted as malignant.

**Why KNN?**

- Easy to understand and implement
- Effective baseline classifier
- Works well for smaller datasets
- Performance can be improved by selecting an appropriate value of `K`

---

### Support Vector Machine (SVM)

Support Vector Machine is a supervised machine-learning algorithm commonly used for classification tasks. It attempts to find an optimal decision boundary, called a **hyperplane**, that separates benign and malignant samples with the largest possible margin.

In simple terms, SVM tries to create the clearest separation between the two diagnostic classes.

**Why SVM?**

- Effective in high-dimensional feature spaces
- Often performs well on structured classification datasets
- Can use different kernel functions for non-linear decision boundaries
- Provides a strong comparison against KNN

---

## Principal Component Analysis (PCA)

Principal Component Analysis is a dimensionality-reduction technique used to transform the original features into a smaller set of meaningful variables called **principal components**.

PCA is used in this project to:

- Reduce feature complexity
- Remove redundant information
- Improve computational efficiency
- Study the effect of reduced dimensions on KNN and SVM performance
- Visualize patterns in the dataset where applicable

PCA is fitted only on training data to prevent data leakage, and the same transformation is applied to the test data.

---

## Project Workflow

```text
Load Dataset
     ↓
Data Cleaning and Exploration
     ↓
Separate Features and Target Label
     ↓
Train-Test Split
     ↓
Feature Scaling
     ↓
Apply PCA (Optional Experiment)
     ↓
Train KNN and SVM Models
     ↓
Evaluate Accuracy and Classification Metrics
     ↓
Compare Results
```

---

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data loading and preprocessing |
| NumPy | Numerical computation |
| Matplotlib / Seaborn | Data visualization |
| Scikit-learn | Model building, PCA, training, and evaluation |
| Jupyter Notebook | Experimentation and result analysis |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/Breast-cancer-diagnosis-using-Machine-Learning.git
cd Breast-cancer-diagnosis-using-Machine-Learning
```

### 2. Create a virtual environment (optional but recommended)

```bash
python -m venv venv
```

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the project

```bash
jupyter notebook
```

Open the project notebook and run all cells in sequence.

---

## Required Libraries

Create a file named `requirements.txt` with the following content:

```txt
numpy
pandas
matplotlib
seaborn
scikit-learn
jupyter
```

Install them using:

```bash
pip install -r requirements.txt
```

---

## Evaluation Metrics

The models are evaluated using more than accuracy because medical classification requires careful interpretation of prediction errors.

- **Accuracy** — proportion of correctly predicted samples
- **Precision** — how reliable positive/malignant predictions are
- **Recall (Sensitivity)** — ability to identify malignant cases
- **F1-score** — balance between precision and recall
- **Confusion Matrix** — detailed count of correct and incorrect predictions

> In a diagnostic setting, false negatives are especially important because a malignant case should not be incorrectly predicted as benign.

---

## Results

The experiments compare KNN and SVM using different values of `K` and different PCA component counts.

| Model | Configuration | Accuracy |
|---|---|---:|
| KNN | Best selected `K` value | `Add your measured result` |
| SVM | Default / tuned parameters | `Add your measured result` |
| KNN + PCA | Selected principal components | `Add your measured result` |
| SVM + PCA | Selected principal components | `Add your measured result` |

### Observations

- SVM performed better than KNN as the amount of training data increased in the conducted experiment.
- KNN performance depended strongly on the selected value of `K`; low values can be more sensitive to noisy data.
- PCA changed model performance by reducing the original feature space into principal components.
- In the original experiment, the highest reported accuracy was **97.95%** with `PC = 1` and `K = 9`.
- Results can change depending on the random train-test split, scaling method, PCA configuration, and model hyperparameters.

> **Important:** Report only the accuracy produced by your own final code execution. Do not claim 100% accuracy unless it is genuinely achieved on a clearly defined evaluation set, and always report the train-test method used.

---

## Suggested Repository Structure

```text
Breast-cancer-diagnosis-using-Machine-Learning/
│
├── data/
│   └── breast_cancer_data.csv
│
├── notebooks/
│   └── breast_cancer_diagnosis.ipynb
│
├── src/
│   ├── preprocess.py
│   ├── train_knn.py
│   ├── train_svm.py
│   └── evaluate.py
│
├── results/
│   ├── confusion_matrix_knn.png
│   ├── confusion_matrix_svm.png
│   └── model_comparison.png
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Limitations

- The dataset is relatively small and may not represent all populations or clinical environments.
- A strong accuracy score does not guarantee real-world medical reliability.
- This model does not use patient history, imaging scans, genomic information, or other clinical context.
- The output should only be treated as an academic prediction, not a treatment recommendation.

---

## Future Improvements

- Perform hyperparameter tuning using `GridSearchCV`
- Compare additional models such as Logistic Regression, Random Forest, and XGBoost
- Use cross-validation for more reliable evaluation
- Add ROC-AUC curves and precision-recall curves
- Deploy the model with Streamlit or Flask
- Add an interactive interface for educational demonstration
- Experiment with feature selection techniques

---

## Academic Integrity and Attribution

This repository is developed for learning and internship-project purposes.

- **Intern ID:** `CITS7306`
- **Project Title:** `Breast Cancer Diagnosis Using Machine Learning`
- **Dataset:** Wisconsin Diagnostic Breast Cancer (WDBC)
- **Original concepts used:** Supervised learning, KNN, SVM, PCA, and standard machine-learning evaluation methods

If this project is adapted from an existing public repository, provide proper credit below:

```text
Base implementation/reference:
[Add original repository URL here]

Modifications, documentation, experiments, and analysis:
[Your Name] — Intern ID: CITS7306
```

---

## Disclaimer

This project is created for **educational and research purposes only**. It is not a medical device and must not be used to diagnose, treat, cure, or prevent any disease. Medical decisions must be made by qualified healthcare professionals.

---

<div align="center">

### Developed by [Your Name]

**Intern ID: CITS7306**

⭐ If you find this project useful, consider starring the repository.

</div>
