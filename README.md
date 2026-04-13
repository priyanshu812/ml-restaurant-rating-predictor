# 🍽️ Zomato Restaurant Rating Predictor

A machine learning web app that predicts restaurant ratings based on location, cuisine, type, and pricing — trained on real Zomato Bangalore data.

**Live Demo →** [Coming Soon](#) *(Streamlit Cloud deployment in progress)*

---

## 📌 Problem Statement

Given details about a restaurant (area, cuisine, type, average cost, online order availability), predict its rating on a scale of 1–5.

This is a **regression problem** — the target is a continuous value (e.g., 3.8 ⭐).

---

## 🧠 Model Performance

| Model | R² | MAE | RMSE |
|---|---|---|---|
| Linear Regression | ~0.35 | ~0.37 | ~0.46 |
| Random Forest (baseline) | ~0.63 | ~0.23 | ~0.34 |
| Gradient Boosting (baseline) | ~0.61 | ~0.24 | ~0.35 |
| **Random Forest (tuned) ✅** | **0.652** | **0.228** | **~0.34** |

Best model: **Tuned Random Forest** via GridSearchCV  
Best params: `max_depth=10`, `n_estimators=200`, `min_samples_split=10`

---

## 🗂️ Project Structure

```
ml-restaurant-rating-predictor/
│
├── data_preprocessing.ipynb   # EDA, cleaning, feature engineering
├── model_training.ipynb       # Model comparison, tuning, evaluation
│
├── app.py                     # Streamlit web app
│
├── Zomato.csv                 # Raw dataset (Zomato Bangalore)
├── zomato_cleaned.csv         # Processed dataset (61 features)
│
├── model.pkl                  # Trained Random Forest model
├── feature_columns.pkl        # Feature column names for inference
├── area_rating.pkl            # Area → mean rating mapping (target encoding)
│
└── requirements.txt           # Python dependencies
```

---

## ⚙️ Feature Engineering

| Feature | Type | Description |
|---|---|---|
| `online_order` | Binary | Whether online ordering is available |
| `table_booking` | Binary | Whether table booking is available |
| `avg_cost` | Numeric | Average cost for two people |
| `is_<cuisine>` × 39 | Binary | One column per cuisine type |
| `is_<rest_type>` × 13 | Binary | One column per restaurant type |
| `encoded_area` | Numeric | Target encoding — mean rating of the area |
| `area_avg_cost` | Numeric | Average cost of restaurants in the area |
| `competition_density` | Numeric | Number of same-type restaurants in the area |
| `price_position` | Numeric | Restaurant cost relative to area average |

---

## 🚀 Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/priyanshu812/ml-restaurant-rating-predictor.git
cd ml-restaurant-rating-predictor
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

---

## 🛠️ Tech Stack

- **Python 3.11**
- **Pandas, NumPy** — data processing
- **Scikit-learn** — model training, GridSearchCV
- **Streamlit** — web app
- **Matplotlib, Seaborn** — EDA visualizations

---

## 👤 Author

**Priyanshu Soni**  
B.Tech CSE (AI/ML) — Jain (Deemed-to-be University), Bangalore  
[LinkedIn](https://linkedin.com/in/priyanshu-soni-ai) • [GitHub](https://github.com/priyanshu812)
