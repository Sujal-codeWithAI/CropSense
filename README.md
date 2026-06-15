# 🌱 CropSense - AI-Powered Crop Recommendation System

CropSense is an intelligent crop recommendation system that helps farmers and agricultural professionals identify the most suitable crop based on soil nutrients, environmental conditions, and regional factors.

The system uses Machine Learning to analyze key agricultural parameters and provides crop recommendations along with pesticide suggestions and crop-specific insights through an interactive dashboard.

---

## 🚀 Features

### 🌾 Crop Prediction

Predicts the most suitable crop using:

* Nitrogen (N)
* Phosphorus (P)
* Potassium (K)
* Temperature
* Humidity
* Soil pH
* Rainfall
* Region Selection

### 📊 Interactive Dashboard

Provides detailed crop information including:

* Crop description
* Suitable climate conditions
* Growth duration
* Water requirements
* Profitability insights
* Seasonal recommendations

### 🧪 Soil Analysis

Visualizes soil nutrient levels and environmental parameters.

### 🐛 Pesticide Recommendations

Displays crop-specific pesticide recommendations including:

* Pest/Disease name
* Recommended pesticide
* Dosage information
* Application method

### 📍 Region-Based Recommendations

Supports region selection to improve crop suitability and recommendation accuracy.

---

## 🤖 Machine Learning Model

The project uses a supervised Machine Learning model trained on agricultural datasets containing soil and climate parameters.

### Input Features

* Nitrogen (N)
* Phosphorus (P)
* Potassium (K)
* Temperature
* Humidity
* pH
* Rainfall
* Region

### Output

* Recommended Crop

### Libraries Used

* Scikit-learn
* Pandas
* NumPy

---

## 🛠️ Tech Stack

### Backend

* Python
* Flask

### Frontend

* HTML5
* CSS3
* JavaScript

### Machine Learning

* Scikit-learn
* Pandas
* NumPy

### Data Storage

* CSV Datasets

---

## 📂 Project Structure

```text
CropSense/
│
├── app.py
├── train_model.py
├── model.py
├── model.pkl
│
├── data/
├── models/
├── utils/
├── templates/
├── static/
├── tests/
│
└── README.md
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/Sujal-codeWithAI/CropSense.git
cd CropSense
```

### Create Virtual Environment

```bash
python -m venv .venv
```

### Activate Environment

Windows:

```bash
.venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
python app.py
```

---

## 📈 Future Enhancements

* Real-time weather API integration
* Fertilizer recommendation system
* Crop disease detection using Deep Learning
* Market price prediction
* Mobile application support
* IoT sensor integration with ESP32

---

## 👨‍💻 Author

**Sujal Raut**

AI & Software Engineering Student

GitHub: https://github.com/Sujal-codeWithAI

LinkedIn: https://www.linkedin.com/in/sujal-raut/

---

## 📜 License

This project is developed for educational and research purposes.
