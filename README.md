# ✈️ Flight Fare Prediction — Machine Learning + Hyperparameter Tuning + Hugging Face Deployment

A complete **machine learning prediction system** built with
**Python**, **scikit-learn**, **XGBoost**, and deployed to **Hugging Face Spaces**.
This project trains, evaluates, tunes, and deploys a production-ready model to predict *flight ticket fares* using the Flight Price dataset from Kaggle.

---

## 📘 Table of Contents

* [Overview](#-overview)
* [Dataset](#-dataset)
* [Modeling Approach](#-modeling-approach)
* [Architecture](#-architecure)
* [Tech Stack](#-tech-stack)
* [Project Structure](#-project-structure)
* [Training & Tuning](#-training--tuning)
* [Cloud Deployment (Hugging Face)](#-cloud-deployment-hugging-face)
* [Running Locally](#-running-locally)
* [How It Works](#-how-it-works)
* [Key Findings](#-key-findings)
* [Final Result](#-final-results)

---

## 🧠 Overview

This project demonstrates how to build and deploy a complete **end-to-end machine learning regression pipeline** for predicting flight fares.
The focus is on:

* **Feature preprocessing**
* **Model experimentation**
* **Hyperparameter tuning**
* **Model versioning**
* **Cloud deployment on Hugging Face Spaces**

### Flight Price Dataset — Overview

The **Flight Price Prediction** dataset contains flight records including airline, route, stops, timings, duration, and final price.
It is widely used for evaluating supervised regression models.

* **Records:** ~300,000
* **Target Variable:** Price
* **Source:** Kaggle – Flight Price Prediction
  [https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction/data](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction/data)

### Project Summary

1. Clean and preprocess the flight dataset
2. Vectorize using **DictVectorizer**
3. Train multiple regression models
4. Perform tuning on Decision Tree, Random Forest, and XGBoost
5. Select the best model based on RMSE
6. Package model + vectorizer
7. Deploy to **Hugging Face Spaces** using Gradio

---

## 📊 Dataset

| Column          | Type        | Description               |
| --------------- | ----------- | ------------------------- |
| Airline         | Categorical | Type of airline           |
| Flight          | Categorical | Type of airline           |
| Arrival         | Categorical | Type of airline           |
| Deparature      | Categorical | Type of airline           |
| Source          | Categorical | Origin city               |
| Destination     | Categorical | Arrival city              |
| Stops           | Categorical | 0, 1, 2+ stops            |
| Duration        | Numeric     | Flight duration (minutes) |
| Days Left       | Numeric     | Gap to booking & trip date|
| Price (target)  | Numeric     | Final flight fare         |

---

## 🧪 Modeling Approach

We trained and compared the following models:

| Model                     | Status                                 |
| ------------------------- | -------------------------------------- |
| **Linear Regression**     | Baseline model                         |
| **DecisionTreeRegressor** | Improved immensly                      |
| **RandomForestRegressor** | ⭐ **Best overall performance**       
| **XGBoost Regressor**     | Competitive but slightly lower results |

---

## 🏗️ Architecture

```
               ┌──────────────────────────────┐
               │     Model Training (Local)    │
               │  - Data cleaning              │
               │  - Feature encoding           │
               │  - Regression models          │
               │  - Hyperparameter tuning      │
               └───────────────┬──────────────┘
                               │
                     Export best model
                               │
               ┌───────────────▼──────────────┐
               │      Hugging Face Space       │
               │  Model + Inference Script     │
               │  Web UI + API endpoint        │
               └───────────────────────────────┘
```

---

## ⚙️ Tech Stack

| Component               | Purpose                     |
| ----------------------- | --------------------------- |
| **Python 3.12+**        | Core runtime                |
| **Pandas / NumPy**      | Data manipulation           |
| **Scikit-learn**        | ML models + metrics         |
| **XGBoost**             | Gradient boosting model     |
| **DictVectorizer**      | Feature encoding            |
| **Joblib / Pickle**     | Model serialization         |
| **Hugging Face Spaces** | Cloud deployment platform   |
| **Gradio**              | Simple ML web UI (for demo) |

---

## 📁 Project Structure

```text
flight-fare-prediction/
├── dataset.csv
├── app.py                      # Gradio inference API
├── train.py                    # Model training + tuning script
├── notebook.ipynb              # Full ML workflow notebook
├── requirements.txt            # Dependencies for deployment
└── README.md                   # Project Description
```

---

## 🧪 Training & Tuning

### Steps performed:

1. **Preprocessing**

   * Clean missing values
   * Normalize target variable
   * Encode categorical + numerical columns using `DictVectorizer`

2. **Model Training**
   Models trained:

   * Linear Regression
   * Decision Tree
   * Random Forest
   * XGBoost

3. **Hyperparameter Tuning**

   * `max_depth`
   * `min_samples_leaf` / `min_child_weight`
   * `num_boost_rounds`
   * `eta` (learning rate for XGB)

4. **Result**

   * **RandomForestRegressor achieved the best RMSE**
   * Packaged into `model.bin`

---

## ☁️ Cloud Deployment (Hugging Face)

The model is deployed entirely in the cloud using **Hugging Face Spaces**.

### 1. Create a New Space

* Go to: [https://huggingface.co/spaces](https://huggingface.co/spaces)
* Choose:

  * **SDK:** `Gradio`
  * **Hardware:** CPU
* Create the Space

### 2. Run `train.py` to generate the model.

### 3. Upload Files

Upload the following:

```
app.py
model.bin
requirements.txt
```

### 4. Space will auto-launch your application.

---

## 🖥️ Running Locally

### **1. Install dependencies**

```bash
pip install -r requirements.txt
```

### **2. Train the model**

This generates `model.bin`.

```bash
python train.py
```

### **3. Run the Gradio app**

```bash
python app.py
```

Your app will start at:

➡️ [http://127.0.0.1:7860](http://127.0.0.1:7860)

---

## 🔍 How It Works

| Step | Component       | Description                               |
| ---- | --------------- | ----------------------------------------- |
| 1    | Preprocessing   | Clean + extract + encode features         |
| 2    | Modeling        | Train 4 regression models                 |
| 3    | Tuning          | Improve performance via hyperparameters   |
| 4    | Model Selection | Best RMSE → RandomForest                  |
| 5    | Serialization   | Save model + DictVectorizer               |
| 6    | Deployment      | Serve via Hugging Face Space using Gradio |

---

## 🧠 Key Learnings

* Hyperparameter tuning has a major impact on RMSE
* RandomForestRegressor provided the best balance of bias–variance
* Tree-based models handle categorical + numeric features well
* Hugging Face Spaces allows simple, production-level cloud ML deployment

---

## 🎉 Final Result

You now have a **cloud-hosted machine learning service** on Hugging Face that:

* Accepts real flight details
* Runs a fully tuned Random Forest model
* Returns accurate fare predictions instantly
* Is accessible to anyone with an internet connection
