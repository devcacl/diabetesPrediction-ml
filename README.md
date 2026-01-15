# 🏥 Diabetes Prediction System (Full Stack ML)

End-to-end full stack application for **diabetes risk prediction** using **Machine Learning (XGBoost)**, a **Flask REST API**, and a **React frontend**.

---

## 📌 Project Overview

This system allows users to input medical data through a web interface and receive a diabetes risk prediction powered by a trained **XGBoost classification model**.

### 🔄 How it works
1. User enters medical data in the React frontend
2. Frontend sends data to the Flask API
3. The ML model processes the input
4. The API returns prediction, probability, risk level, and recommendations
5. Results are displayed visually to the user

---

## 🧠 Machine Learning Model

- **Algorithm:** XGBoost (Extreme Gradient Boosting)
- **Type:** Binary classification
- **Dataset:** Pima Indians Diabetes Dataset
- **Outputs:**
  - Diabetes prediction (Yes / No)
  - Probability score
  - Risk level (Low / Medium / High)
  - Confidence level
  - Medical interpretation
  - Personalized recommendations

### Why XGBoost?
- High accuracy for tabular medical data
- Handles non-linear relationships
- Robust to noisy data
- Widely used in healthcare ML research

---

## 🧩 Tech Stack

### Backend
- Python 3.10+
- Flask
- XGBoost
- Scikit-learn
- NumPy
- Pandas
- Gunicorn

### Frontend
- React
- Axios
- CSS
- Responsive UI

### Tools
- Git
- GitHub
- REST APIs
--

### 🖥️ Frontend Interface (React)

<img width="1828" height="821" alt="Screenshot 2026-01-15 152141" src="https://github.com/user-attachments/assets/84b65165-6a48-4339-b716-8d354029bc1b" />

<img width="1836" height="839" alt="Screenshot 2026-01-15 152201" src="https://github.com/user-attachments/assets/5d1509bd-3a4d-4e82-a1f8-e7ca9003c99c" />

### 📊 Confusion Matrix

<img width="2720" height="2068" alt="confusion_matrix" src="https://github.com/user-attachments/assets/9f23e6b6-c16d-4201-90b5-efdace7b6d1f" />
---

### 🧠 Feature Importance

<img width="2969" height="1763" alt="feature_importance" src="https://github.com/user-attachments/assets/d489b5e4-fdf2-44ef-bb78-f53c32160455" />


### 🔍 Interpretation Notes

- **Glucose** is the most influential variable in diabetes prediction
- **BMI** and **Age** strongly affect risk classification
- Family history (Diabetes Pedigree Function) adds contextual risk
- The model prioritizes clinically relevant features

---

👨‍💻 Author
Camilo Coronado
Systems Engineering 
Machine Learning & Full Stack Developer

⭐ If you like this project
Give it a ⭐ on GitHub and feel free to fork or contribute!
