# Fake Social Media Account Detection — Account Lens

Flask application for account screening with Random Forest predictions, model confidence, held-out model comparisons, and CSV batch predictions.

## Run on Windows

Run from the folder containing app.py:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train_model.py
.\.venv\Scripts\python.exe app.py
```

Open http://127.0.0.1:5000. The app refreshes its cached prediction model when the saved model file changes.

## Features

- /prediction: validated numeric inputs, sample autofill, predicted class, model confidence and fake-class probability.
- /dashboard (also /model): actual model accuracy, precision, recall, F1, confusion matrix, and Random Forest global feature importance.
- /batch: upload up to 2 MB / 10,000 rows; download predictions as CSV. Extra columns are preserved, apart from reserved output names prediction, confidence_percent and fake_probability_percent which are replaced.
- /batch/sample: download the expected CSV schema.
- /view: first 100 original dataset records.

## Training

train_model.py removes exact duplicates before the stratified 70/30 split, fixes random seeds, and trains Random Forest, Decision Tree, and scaled Logistic Regression. It saves all three models and model_metrics.json. Original model files are preserved as .legacy.sav backups on the first rebuild.

Current evaluation: 692 unique accounts, 484 training, 208 test. Random Forest accuracy: 93.75%; Decision Tree: 90.87%; Logistic Regression: 91.35%. These results use a different cleaned split from the original notebook and are not a direct measure of improvement over its historical scores.

Model confidence is uncalibrated and is not proof of account authenticity. Dataset collection and labeling provenance are not documented. The app uses manually entered account statistics; it does not fetch social platform data.

## Checks

```powershell
.\.venv\Scripts\python.exe -m unittest -v test_app
```

Twelve regression tests cover page rendering, internal links and CSS, preserved feature values, invalid and overflowing numeric values, duplicate CSV headers, row and upload limits, missing artifacts, batch output, and agreement between saved models and recorded metrics. Requirements match the environment used to train the saved models. Retrain with train_model.py if changing scikit-learn versions.

## Project maintenance

features.py defines the shared numeric schema and validation. The notebook calls train_model.py instead of independently overwriting model artifacts. Metrics record a UTC training timestamp and dataset SHA-256 hash. Training rejects invalid labels and contradictory labels for identical account features.

Use the project Python environment as the Jupyter kernel. Runtime requirements do not install Jupyter itself. Old static assets, dataset.csv, fake.html, and unused contact/service/model templates remain as historical files; active routes use the shared base template and data.csv. No login, database, social API integration or production deployment is included.

See PROJECT_REVIEW.md for the review findings, verification scope and suggested next additions.
