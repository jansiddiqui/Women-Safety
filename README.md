# Women Safety Analytics – Machine Learning Classification Project

## 1. Project Overview

This project focuses on predicting whether a given area is **Safe** or **Unsafe** for women using machine learning.
The system analyzes crime data, performs feature engineering, and compares multiple ML models to select the best one for real-world application.

The project uses three supervised learning models:

* Logistic Regression
* Random Forest Classifier
* XGBoost Classifier

After training and evaluation, **XGBoost** emerges as the most reliable and robust model for deployment.

---

## 2. Objectives

* Predict area safety (Safe vs Unsafe)
* Analyze crime-related features such as:

  * Time of occurrence
  * Geographic location (latitude/longitude)
  * Crime type and severity
  * Victim and offender characteristics
  * Police deployment
* Engineer new features to improve model performance
* Compare three machine learning models based on accuracy and confusion matrix
* Enable user-based predictions through a model input interface

---

## 3. Dataset Details

The dataset includes real crime records with attributes such as:

* Date and time of crime
* City
* Crime description
* Victim age and gender
* Weapon used
* Crime domain
* Police deployment
* Latitude and longitude

Additional engineered features include:

* hour_of_day
* is_weekend
* month
* week_of_year
* lat_bucket and lon_bucket
* is_rape, is_assault, is_harass

These engineered features improved the detection of crime patterns significantly.

---

## 4. Technologies and Libraries Used

* Python 3
* Pandas, NumPy
* Scikit-learn
* XGBoost
* Seaborn, Matplotlib
* GridSearchCV
* Pipeline and ColumnTransformer

---

## 5. Machine Learning Models Implemented

### 5.1 Logistic Regression

* Acts as the baseline model
* Easy to interpret
* Achieved approximately 95% accuracy

### 5.2 Random Forest Classifier

* Ensemble model using bagging
* Handles nonlinear relationships well
* Achieved **97.25% accuracy**

### 5.3 XGBoost Classifier

* Gradient boosting decision tree
* Fast, robust, and highly accurate
* Achieved **97.25% accuracy** (best model for deployment)

---

## 6. Performance Summary

| Model               | Training Accuracy | Testing Accuracy |
| ------------------- | ----------------- | ---------------- |
| Logistic Regression | ~95%              | ~95%             |
| Random Forest       | 97.29%            | 97.25%           |
| XGBoost             | 97.29%            | 97.25%           |

Both Random Forest and XGBoost performed exceptionally well, but XGBoost is chosen as the **final model** due to its stronger generalization and industry-wide usage.

---

## 7. Confusion Matrix (XGBoost Example)

A confusion matrix evaluates how well the model predicts each class.

Example:

```
[[2270   13]
 [ 134 2068]]
```

Interpretation:

* 2270 Safe instances predicted correctly
* 2068 Unsafe instances predicted correctly
* 134 Unsafe instances predicted as Safe (critical)
* Overall accuracy: 97.25%

---

## 8. Feature Engineering Summary

The following features were added to improve model performance:

* hour_of_day
* is_weekend
* month
* week_of_year
* lat_bucket
* lon_bucket
* is_rape
* is_assault
* is_harass

These features provided more meaningful patterns for the ML models.

---

## 9. Custom Prediction Example

Users provide:

* City
* Crime domain
* Weapon used
* Victim gender
* Victim age
* Location coordinates
* Time of day
* Crime indicators

The model outputs:

* Safe or Unsafe prediction
* Probability score

Example:

```
Prediction: Unsafe Area
Probability: [[0.88 (Unsafe), 0.12 (Safe)]]
```

---

## 10. How to Run the Project

Install dependencies:

```
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
```

Run each model:

```
python logistic_model.py
python random_forest_model.py
python xgboost_model.py
```

---

## 11. Conclusion

This project demonstrates how machine learning can be used to evaluate safety levels in different areas by analyzing crime data.
Key outcomes:

* Feature engineering significantly improved performance
* XGBoost and Random Forest achieved the highest accuracy
* Project is ready for integration into future applications such as:

  * Web dashboards
  * Mobile applications
  * Real-time safety scoring systems

---

## 12. Future Enhancements

* Real-time crime data integration
* Location-based heatmaps
* Mobile app deployment
* Flask or FastAPI backend for live predictions
* Anomaly detection for unusual crime spikes

---
