<div align="center">

# 🌱 CropSense
### AI-Powered Crop Recommendation System

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.x-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-Educational-green?style=for-the-badge)](LICENSE)

> CropSense helps farmers and agricultural professionals identify the most suitable crop based on soil nutrients, environmental conditions, and regional factors — powered by Machine Learning.

</div>

---

## 🖼️ Project Screenshots

### 🏠 Input — Soil & Climate Parameters
![System Interface Showing Input Parameters](Outcome/System%20Interface%20Showing%20Input%20Parameters.png)

### 🌾 Output — Crop Prediction Result
![System Interface Showing Output Parameters](Outcome/System%20Interface%20Showing%20Output%20Parameters.png)

### 🧪 Soil Requirements & Growth Information
![Crop Recommendation Output Showing Soil Requirements and Growth Information](Outcome/Crop%20Recommendation%20Output%20Showing%20Soil%20Requirements%20and%20Growth%20Information.png)

### 🐛 Pesticide & Disease Management
![Pesticide and Disease Management Recommendations for Predicted Crop](Outcome/Pesticide%20and%20Disease%20Management%20Recommendations%20for%20Predicted%20Crop.png)

### 📊 Model Evaluation Report
![Model Evaluation Report Showing Accuracy and Dataset Overview](Outcome/Model%20Evaluation%20Report%20Showing%20Accuracy%20and%20Dataset%20Overview.png)

### 🔌 IoT Sensor Integration (ESP32)
![IoT Sensor Integration using ESP32 and Environmental Sensors](Outcome/IoT%20Sensor%20Integration%20using%20ESP32%20and%20Environmental%20Sensors.png)

---

## 🚀 Features

| Feature | Description |
|---|---|
| 🌾 **Crop Prediction** | Recommends the best crop using 8 soil & climate parameters |
| 📊 **Model Evaluation** | Live performance charts — accuracy, F1, confusion matrix |
| 🐛 **Pesticide Guide** | Crop-specific pest/disease treatment with dosage & method |
| 🧪 **Soil Analysis** | Visualizes NPK suitability vs ideal crop requirements |
| 📍 **Region Awareness** | Checks crop suitability for 28 Indian states |
| 🌐 **REST API** | `/predict_api` endpoint for external integrations |
| 🤖 **IoT Ready** | ESP32 sensor integration for real-time soil data |

---

## 🤖 Machine Learning

### Model
- **Algorithm:** Logistic Regression (L2 regularization, C=0.1)
- **Preprocessing:** StandardScaler (Z-score normalization)
- **Train/Test Split:** 80% / 20%

### Performance

| Metric | Score |
|---|---|
| ✅ Test Accuracy | **85.91%** |
| 📐 Macro Avg F1 | **0.860** |
| 📐 Weighted Avg F1 | **0.860** |
| 🌾 Crop Classes | **22** |
| 📦 Total Samples | **2,200** |

### Input Features

```
N (Nitrogen)  ·  P (Phosphorus)  ·  K (Potassium)
Temperature   ·  Humidity        ·  Soil pH
Soil Moisture ·  Rainfall
```

### Output
```
→ Recommended Crop  (one of 22 classes)
```

---

## 🛠️ Tech Stack

```
Backend       →  Python · Flask
Frontend      →  HTML5 · CSS3 · JavaScript · Bootstrap 5
ML            →  Scikit-learn · Pandas · NumPy · Matplotlib · Seaborn
Data          →  CSV Datasets (2,200 samples · 22 crop classes)
IoT           →  ESP32 · DHT11 · Soil Moisture Sensor
```

---

## 📂 Project Structure

```
CropSense/
│
├── app.py                   # Flask application & routes
├── train_model.py           # Model training script
├── model.py                 # EDA & dataset exploration
│
├── data/
│   ├── Crop_recommendation.csv
│   ├── crop_details.csv
│   ├── pesticides_recommendation.csv
│   └── crop_region.json
│
├── models/
│   ├── model.pkl            # Trained Logistic Regression model
│   └── scaler.pkl           # Fitted StandardScaler
│
├── utils/
│   └── predictor.py         # Feature validation & prediction logic
│
├── templates/
│   ├── index.html           # Input form
│   ├── result.html          # Prediction result page
│   ├── results.html         # Model evaluation report
│   └── error.html
│
├── static/
│   ├── css/style.css
│   └── images/
│
├── esp32_sensor/
│   └── esp32_sensor.ino     # Arduino sketch for IoT sensor
│
└── tests/
    └── request.py
```

---

## ⚙️ Installation

**1. Clone the repository**
```bash
git clone https://github.com/Sujal-codeWithAI/CropSense.git
cd CropSense
```

**2. Create & activate virtual environment**
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirments.txt
```

**4. Train the model**
```bash
python train_model.py
```

**5. Run the application**
```bash
python app.py
```

Open `http://localhost:5000` in your browser.

---

## 🌐 API Usage

**Endpoint:** `POST /predict_api`

```json
{
  "N": 90, "P": 42, "K": 43,
  "temperature": 21.0, "humidity": 82.0,
  "ph": 6.5, "rainfall": 200.0,
  "soil_moisture": 60.0,
  "region": "Maharashtra"
}
```

**Response:**
```json
{
  "crop": "rice",
  "region": "Maharashtra",
  "region_suitable": true,
  "region_alternatives": [],
  "pesticides": [...]
}
```

---

## 🔌 IoT Integration (ESP32)

The `esp32_sensor/esp32_sensor.ino` sketch reads live data from:
- **DHT11** — Temperature & Humidity
- **Soil Moisture Sensor** — Analog soil moisture %

Hosts a web dashboard at the ESP32's local IP that auto-refreshes every 5 seconds.

---

## 📈 Future Enhancements

- [ ] Real-time weather API integration
- [ ] Fertilizer recommendation system
- [ ] Crop disease detection using Deep Learning
- [ ] Market price prediction
- [ ] Mobile application (React Native)
- [ ] IoT pipeline: ESP32 → CropSense API → Auto prediction

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | Crop Recommendation Dataset |
| Total Samples | 2,200 |
| Crops | 22 (100 samples each — balanced) |
| Features | N, P, K, Temperature, Humidity, pH, Rainfall |
| Format | CSV |

---

## 👨‍💻 Author

**Sujal Raut** — AI & Software Engineering Student

[![GitHub](https://img.shields.io/badge/GitHub-Sujal--codeWithAI-181717?style=flat-square&logo=github)](https://github.com/Sujal-codeWithAI)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-sujal--raut-0077B5?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/sujal-raut/)

---

## 📜 License

This project is developed for **educational and research purposes**.

---

<div align="center">
  <sub>Built with ❤️ by Sujal Raut</sub>
</div>
