
## 📊 Project Overview

- **Goal:** Classify crop seedlings and weed species from image data to support precision agriculture.
- **Dataset:** 12 species of seedlings provided in separate training and testing directories.
- **Approach:**
  - A lightweight **Convolutional Neural Network (CNN)** was trained on the dataset for feature extraction.
  - Extracted feature vectors (size `256`) were used to train:
    - **Support Vector Machine (SVM)** with a linear kernel.
    - **XGBoost** with tuned hyperparameters.
- **Evaluation Metrics:** Accuracy, precision, recall, weighted F1-score, and micro-average ROC-AUC.

---

## ⚙️ Installation

 1. Clone this repository
```bash
git clone https://github.com/your-username/seedling-classification.git
cd seedling-classification

 2. Create and activate a Python environment
python -m venv env
source env/bin/activate   # On Linux/Mac
env\Scripts\activate      # On Windows

3. Install dependencies
pip install -r requirements.txt

