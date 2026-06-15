import logging
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

FEATURE_ORDER = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'soil_moisture', 'rainfall']

data = pd.read_csv('data/Crop_recommendation.csv')
if 'soil_moisture' not in data.columns:
    logging.info('Soil moisture column not found in dataset; estimating synthetic values from humidity and rainfall.')
    rng = np.random.default_rng(42)
    base = (data['humidity'] * 0.7) + (data['rainfall'] / 30 * 0.3)
    noise = rng.normal(0, 8, size=len(data))
    data['soil_moisture'] = (base + noise).clip(0, 100)

x = data[FEATURE_ORDER]
y = data['label']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

rng2 = np.random.default_rng(7)
x_train = x_train.copy()
x_test  = x_test.copy()
for col in FEATURE_ORDER:
    std = x_train[col].std()
    x_train[col] += rng2.normal(0, std * 0.22, size=len(x_train))
    x_test[col]  += rng2.normal(0, std * 0.22, size=len(x_test))

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled  = scaler.transform(x_test)

model = LogisticRegression(max_iter=150, C=0.1, solver='lbfgs', random_state=42)
model.fit(x_train_scaled, y_train)

y_pred = model.predict(x_test_scaled)

logging.info("Accuracy: %.4f", model.score(x_test_scaled, y_test))
logging.info("Classification Report:\n%s", classification_report(y_test, y_pred))
logging.info("Confusion Matrix:\n%s", confusion_matrix(y_test, y_pred))

joblib.dump(model,  'models/model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
logging.info("Model and scaler saved to models/")
