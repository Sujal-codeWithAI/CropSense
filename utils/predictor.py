try:
    import joblib
except ImportError:
    try:
        from sklearn.externals import joblib
    except ImportError:
        joblib = None

import numpy as np
import pandas as pd

FEATURE_ORDER = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'soil_moisture', 'rainfall']

RANGES = {
    'N':             (0, 250),
    'P':             (0, 250),
    'K':             (0, 250),
    'temperature':   (0, 50),
    'humidity':      (0, 100),
    'ph':            (0, 14),
    'soil_moisture': (0, 100),
    'rainfall':      (0, 3000),
}

_model = None
_scaler = None


def _load():
    global _model, _scaler
    if joblib is None:
        raise ImportError(
            "The joblib package is required to load the trained model. "
            "Install it with: pip install joblib"
        )
    if _model is None:
        _model = joblib.load('models/model.pkl')
        _scaler = joblib.load('models/scaler.pkl')


def validate(form_data):
    values = {}
    for field in FEATURE_ORDER:
        raw = form_data.get(field)
        if raw is None or str(raw).strip() == '':
            return None, f"Missing field: {field}"
        try:
            val = float(raw)
        except ValueError:
            return None, f"'{field}' must be a number, got: {raw}"
        lo, hi = RANGES[field]
        if not (lo <= val <= hi):
            return None, f"'{field}' must be between {lo} and {hi}, got: {val}"
        values[field] = val
    return values, None


def build_feature_vector(feature_values):
    return np.array([feature_values[f] for f in FEATURE_ORDER])


def predict_crop(features):
    _load()
    df = pd.DataFrame([features], columns=FEATURE_ORDER)
    scaled = _scaler.transform(df)
    return _model.predict(scaled)[0]
