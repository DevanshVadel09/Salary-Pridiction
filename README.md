# Employee Salary Prediction

A machine learning project that predicts employee salary from job-related features. It trains and compares a **Linear Regression** baseline against an **Artificial Neural Network (ANN)** built with Keras, and serves predictions through a small app (`app.py`).

## Features

- End-to-end pipeline: data cleaning, preprocessing, training, evaluation
- Baseline model (Linear Regression) vs. deep learning model (ANN)
- Saved preprocessing artifacts so inference matches training exactly
- Side-by-side model metrics in `model_comparison.csv`
- App for interactive predictions

## Project Structure

```
.
├── employee-salary-pridiction.ipynb   # EDA, preprocessing, training, evaluation
├── job_salary_prediction_dataset.csv  # Dataset (~16 MB)
├── app.py                             # Prediction app
├── model.h5                           # Keras model (legacy H5 format)
├── models/
│   ├── salary_ann.keras               # Trained ANN
│   ├── linear_regression.joblib       # Trained baseline model
│   ├── salary_preprocessor.joblib     # Feature preprocessing pipeline
│   ├── salary_scaler.joblib           # Scaler (e.g. for target / numeric features)
│   ├── salary_config.joblib           # Config: feature names, column order, etc.
│   └── model_comparison.csv           # Metrics for each model
└── README.md
```

## Tech Stack

- Python 3.x
- pandas, NumPy
- scikit-learn
- TensorFlow / Keras
- joblib
- Jupyter Notebook
- <!-- Streamlit / Flask / Gradio: fill in whichever app.py uses -->

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd <your-repo-folder>

# Create and activate a virtual environment (Windows)
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

If you don't have a `requirements.txt` yet, generate one with:

```bash
pip freeze > requirements.txt
```

## Usage

### Run the app

```bash
# Streamlit
streamlit run app.py

# or, if using Flask
python app.py
```

### Use the models in Python

```python
import joblib
import pandas as pd
from tensorflow.keras.models import load_model

preprocessor = joblib.load("models/salary_preprocessor.joblib")
scaler = joblib.load("models/salary_scaler.joblib")
config = joblib.load("models/salary_config.joblib")
model = load_model("models/salary_ann.keras")

# Example input: replace with your dataset's actual feature columns
sample = pd.DataFrame([{
    # "job_title": "Data Scientist",
    # "experience_years": 5,
    # "education": "Master's",
    # "location": "New York",
}])

X = preprocessor.transform(sample)
prediction = model.predict(X)

# If the target was scaled during training, invert it:
# prediction = scaler.inverse_transform(prediction)

print(f"Predicted salary: {prediction[0][0]:,.2f}")
```

> Adjust the feature names and the scaler step to match what the notebook does.

## Models

| Model | File | Description |
|---|---|---|
| Linear Regression | `models/linear_regression.joblib` | Baseline for comparison |
| ANN | `models/salary_ann.keras` | Feed-forward neural network built with Keras |

### Results

See `models/model_comparison.csv` for the full comparison. Summary:

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | — | — | — |
| ANN | — | — | — |

<!-- Fill in the numbers from model_comparison.csv -->

## Dataset

`job_salary_prediction_dataset.csv` contains job-related records with salary as the target variable.

- **Rows:** <!-- fill in -->
- **Features:** <!-- list columns -->
- **Target:** salary

## Workflow

1. Load and explore the dataset (EDA)
2. Clean data and handle missing values
3. Encode categorical features and scale numeric features
4. Split into train and test sets
5. Train Linear Regression and ANN models
6. Evaluate and compare using MAE, RMSE and R²
7. Save the models and preprocessing artifacts
8. Serve predictions through `app.py`

## Notes

- Load the preprocessor and scaler saved alongside the models. Using different preprocessing at inference will give wrong predictions.
- `model.h5` is the legacy Keras format; prefer `salary_ann.keras`.
- Use the same scikit-learn and TensorFlow versions as in training to avoid joblib/Keras loading errors.

## DataSet:
https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
