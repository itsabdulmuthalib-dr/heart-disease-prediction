# Heart Disease Prediction Using Data Mining

## 📌 Project Overview
This project predicts heart disease using machine learning techniques. It helps in early detection by analyzing patient health data.

## 📊 Dataset
UCI Heart Disease Dataset  
https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data

## 🤖 Algorithms Used
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Random Forest

## 🏆 Best Model
Random Forest achieved the highest accuracy (~86%)

## ⚙️ How to Run the Project

1. Install dependencies:
pip install -r requirements.txt

2. Run the application:
streamlit run app.py

## 📁 Project Files
- app.py → Streamlit application
- model_training.py → Model training code
- model.pkl → Trained model
- scaler.pkl → Data scaler
- dataset.csv → Dataset
- requirements.txt → Dependencies

## 📌 Output
The system predicts:
- Low Risk / High Risk of heart disease
- Prediction confidence

## 📌 Conclusion
This project demonstrates how data mining techniques can be used in healthcare to support early diagnosis and better decision-making.
