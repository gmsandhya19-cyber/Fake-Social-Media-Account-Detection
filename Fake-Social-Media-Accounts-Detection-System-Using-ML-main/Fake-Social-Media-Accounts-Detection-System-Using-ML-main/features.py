"""Shared feature schema and validation for training and inference."""
import numpy as np
import pandas as pd

FEATURES = ['profile pic', 'nums/length username', 'fullname words', 'nums/length fullname', 'name==username', 'description length', 'external URL', 'private', '#posts', '#followers', '#follows']
LABELS = ['Profile picture', 'Username digit ratio', 'Full-name word count', 'Full-name digit ratio', 'Name matches username', 'Bio length', 'External URL', 'Private account', 'Posts', 'Followers', 'Following']
KEYS = ['text'] + [f'f{i}' for i in range(2,12)]
BINARY = {0,4,6,7}
RATIOS = {1,3}
FIELDS = [dict(key=k, label=l, binary=i in BINARY, ratio=i in RATIOS) for i,(k,l) in enumerate(zip(KEYS,LABELS))]

def validate(frame):
    if frame.columns.duplicated().any():
        raise ValueError('CSV column names must be unique.')
    missing = [f for f in FEATURES if f not in frame.columns]
    if missing:
        raise ValueError('Missing columns: ' + ', '.join(missing))
    if frame.empty or len(frame) > 10000:
        raise ValueError('Provide between 1 and 10,000 accounts.')
    try:
        x = frame[FEATURES].apply(pd.to_numeric, errors='raise')
    except (ValueError, TypeError):
        raise ValueError('All account features must contain numbers.')
    if not np.isfinite(x.to_numpy(dtype=float)).all():
        raise ValueError('All fields must contain finite numbers; blank fields are not allowed.')
    for i, name in enumerate(FEATURES):
        values = x[name]
        if i in BINARY and not values.isin([0,1]).all():
            raise ValueError(f'{LABELS[i]} must be 0 or 1.')
        if i in RATIOS and not values.between(0,1).all():
            raise ValueError(f'{LABELS[i]} must be between 0 and 1.')
        if i not in BINARY | RATIOS and ((values < 0).any() or (values % 1 != 0).any()):
            raise ValueError(f'{LABELS[i]} must be a nonnegative whole number.')
    if (x.abs() > np.finfo(np.float32).max).any().any():
        raise ValueError('A value is too large for the model. Enter realistic account statistics.')
    return x

