from pathlib import Path
from functools import lru_cache
import io
import csv
import json
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, Response

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024
from features import FEATURES, LABELS, KEYS, FIELDS, validate

class ModelUnavailable(RuntimeError):
    pass

@lru_cache(maxsize=1)
def load_predictor(modified_ns):
    import warnings
    from sklearn.exceptions import InconsistentVersionWarning
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', InconsistentVersionWarning)
            with (BASE_DIR / 'Random_forest.sav').open('rb') as handle:
                model = pickle.load(handle)
        if list(model.feature_names_in_) != FEATURES or set(model.classes_) != {0,1}:
            raise ValueError('Unexpected model schema')
        return model
    except (OSError, ValueError, TypeError, AttributeError, ImportError, EOFError, pickle.UnpicklingError, InconsistentVersionWarning) as exc:
        raise ModelUnavailable('The prediction model needs rebuilding. Run train_model.py with the project Python environment.') from exc

def predictor():
    try:
        stamp = (BASE_DIR / 'Random_forest.sav').stat().st_mtime_ns
    except OSError as exc:
        raise ModelUnavailable('The prediction model is missing. Run train_model.py with the project Python environment.') from exc
    return load_predictor(stamp)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/view')
def view():
    data = pd.read_csv(BASE_DIR / 'data.csv').head(100)
    return render_template('view.html', columns=data.columns, rows=data.values.tolist())

@app.route('/model', methods=['GET','POST'])
@app.route('/dashboard')
def model():
    try:
        metrics = json.loads((BASE_DIR / 'model_metrics.json').read_text())
        if not {'models','importance','test_rows','rows','train_rows','genuine','fake','duplicates_removed'} <= metrics.keys():
            raise ValueError('Incomplete metrics')
    except (OSError, ValueError) as exc:
        raise ModelUnavailable('Model evaluation results are unavailable. Run train_model.py to rebuild them.') from exc
    return render_template('dashboard.html', metrics=metrics)

@app.route('/prediction', methods=['GET','POST'])
def prediction():
    result = None
    error = None
    if request.method == 'POST':
        try:
            x = validate(pd.DataFrame([[request.form.get(k, '') for k in KEYS]], columns=FEATURES))
            mdl = predictor()
            label = int(mdl.predict(x)[0])
            probabilities = mdl.predict_proba(x)[0]
            fake_index = list(mdl.classes_).index(1)
            result = dict(label='Likely fake' if label else 'Likely genuine', confidence=round(float(max(probabilities))*100,1), fake_probability=round(float(probabilities[fake_index])*100,1))
        except ValueError as exc:
            error = str(exc)
    return render_template('prediction.html', fields=FIELDS, values=request.form, result=result, error=error), (400 if error else 200)

@app.route('/batch', methods=['GET','POST'])
def batch():
    if request.method == 'POST':
        try:
            upload = request.files.get('data')
            if not upload or not upload.filename.lower().endswith('.csv'):
                raise ValueError('Select a CSV file.')
            text = upload.read().decode('utf-8-sig')
            header = next(csv.reader(io.StringIO(text)), [])
            if len(header) != len(set(header)):
                raise ValueError('CSV column names must be unique.')
            data = pd.read_csv(io.StringIO(text), nrows=10001)
            if data.columns.duplicated().any():
                raise ValueError('CSV column names must be unique.')
            x = validate(data)
            mdl = predictor()
            labels = mdl.predict(x)
            probabilities = mdl.predict_proba(x)
            data['prediction'] = ['Likely fake' if value else 'Likely genuine' for value in labels]
            data['confidence_percent'] = (probabilities.max(axis=1)*100).round(1)
            data['fake_probability_percent'] = (probabilities[:,list(mdl.classes_).index(1)]*100).round(1)
            return Response(data.to_csv(index=False), mimetype='text/csv', headers={'Content-Disposition': 'attachment; filename=account_predictions.csv'})
        except (ValueError, pd.errors.ParserError, UnicodeError, csv.Error) as exc:
            return render_template('batch.html', error=str(exc)), 400
    return render_template('batch.html')

@app.route('/batch/sample')
def batch_sample():
    sample = pd.read_csv(BASE_DIR / 'data.csv').loc[:4, FEATURES]
    return Response(sample.to_csv(index=False), mimetype='text/csv', headers={'Content-Disposition':'attachment; filename=sample_accounts.csv'})

@app.errorhandler(ModelUnavailable)
def model_unavailable(error):
    return render_template('error.html', title='Model unavailable', message=str(error)), 503

@app.errorhandler(404)
def not_found(error):
    return render_template('error.html', title='Page not found', message='Choose a page from the navigation to continue.'), 404

@app.errorhandler(413)
def too_large(error):
    return render_template('batch.html', error='File is too large. Maximum upload size is 2 MB.'), 413

if __name__ == '__main__':
    app.run()
