"""Train reproducible models and save held-out evaluation results."""
from pathlib import Path
import json
import hashlib
from datetime import datetime, timezone
from features import FEATURES, validate
import pickle
import shutil
import sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
BASE_DIR = Path(__file__).resolve().parent

def main():
    raw = pd.read_csv(BASE_DIR / 'data.csv')
    data = raw.drop_duplicates()
    x = validate(data)
    y = pd.to_numeric(data['fake'], errors='raise')
    if not y.isin([0,1]).all() or y.nunique() != 2:
        raise ValueError('Training labels must contain both genuine (0) and fake (1) accounts.')
    conflicting = data.groupby(FEATURES, dropna=False)['fake'].nunique()
    if (conflicting > 1).any():
        raise ValueError('Identical feature rows have conflicting labels. Correct the dataset before training.')
    xt, xv, yt, yv = train_test_split(x, y, test_size=0.3, stratify=y, random_state=42)
    models = {
        'Random Forest': ('Random_forest.sav', RandomForestClassifier(n_estimators=200, random_state=42)),
        'Decision Tree': ('Decision_tree.sav', DecisionTreeClassifier(random_state=42)),
        'Logistic Regression': ('Logistic_Regression.sav', make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=42)))
    }
    report = {'trained_at': datetime.now(timezone.utc).isoformat(), 'dataset_sha256': hashlib.sha256((BASE_DIR / 'data.csv').read_bytes()).hexdigest(), 'sklearn_version': sklearn.__version__, 'rows': len(data), 'duplicates_removed': len(raw)-len(data), 'train_rows': len(xt), 'test_rows': len(xv), 'genuine': int((y==0).sum()), 'fake': int((y==1).sum()), 'features': list(x.columns), 'models': {}}
    for name, (filename, model) in models.items():
        model.fit(xt, yt)
        pred = model.predict(xv)
        report['models'][name] = {key: float(fn(yv, pred)) for key, fn in [('accuracy', accuracy_score), ('precision', precision_score), ('recall', recall_score), ('f1', f1_score)]}
        report['models'][name]['confusion'] = confusion_matrix(yv, pred, labels=[0,1]).tolist()
        path = BASE_DIR / filename
        backup = path.with_suffix('.legacy.sav')
        if path.exists() and not backup.exists():
            shutil.copy2(path, backup)
        temporary = path.with_suffix('.sav.tmp')
        with temporary.open('wb') as handle:
            pickle.dump(model, handle)
        temporary.replace(path)
        if name == 'Random Forest':
            report['importance'] = sorted(zip(x.columns, model.feature_importances_.tolist()), key=lambda item: item[1], reverse=True)
    metrics_path = BASE_DIR / 'model_metrics.json'
    temporary = metrics_path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(metrics_path)
    print(json.dumps(report, indent=2))
if __name__ == '__main__':
    main()
