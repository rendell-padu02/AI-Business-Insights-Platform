# 📊 AI Business Insights Platform  
*End-to-end data analytics, churn prediction, and sales forecasting platform built with Python, Streamlit, and ML.*

---

## 1. 🚀 Overview

The **AI Business Insights Platform** is a full-stack data analytics project designed as a capstone portfolio piece for junior Data Analysts and Data Scientists. It transforms raw business datasets into actionable insights through:

- 📈 **Sales & Profit Analytics**  
- 🤖 **Machine Learning–based Customer Churn Prediction**  
- 🔮 **Time-Series Sales Forecasting (Prophet)**  
- 📊 **Automated Business Intelligence Dashboard (Streamlit)**  
- 🧠 **Rule-Based Insights Engine**  

This project integrates real-world tools, industry patterns, and a production-like workflow suitable for resumes, interviews, and academic submissions.

---

## 2. 🧠 Key Features

### 🔹 Superstore Sales Analytics
- Total Sales, Profit, Orders, Discount analysis  
- Segment, Category, and Region–wise profitability  
- Monthly sales & profit trends  
- **Forecasting future sales using Facebook Prophet**  
- KPI Cards for business reporting  

### 🔹 Telco Customer Churn Analytics
- Churn KPIs: churn rate, tenure, monthly charges  
- Churn breakdown by contract, payment method, services  
- Missing value handling  
- **ML churn prediction models:**
  - Logistic Regression (AUC: **0.9746**)  
  - Random Forest Classifier (AUC: **0.9446**)  
- ROC curves and model comparison  

### 🔹 Automated Insights Engine
- High-discount warnings  
- Most profitable vs least profitable categories  
- Churn-risk contract types  
- Risky customer behaviors  

### 🔹 Streamlit Dashboard (End-to-End App)
A clean, interactive dashboard with sections for:
- Superstore Overview  
- Sales Trends & Forecast  
- Churn Summary  
- Auto Insights  

### 🔹 Runs locally with one command:
```bash
streamlit run app/app.py
```
---

## 3. Project Architecture
```bash
AI-Business-Insights-Platform/
│
├── app/
│   └── app.py                  # Streamlit UI
│
├── src/
│   ├── data_cleaning.py        # Cleaning logic for both datasets
│   └── insights_engine.py      # KPIs + trends + forecasting + ML helpers
│
├── data/
│   ├── raw/                    # Raw datasets (not tracked in Git)
│   └── processed/              # Cleaned datasets
│
├── notebooks/
│   └── eda.ipynb               # EDA, profiling, visualizations
│
├── models/                     # Optional: store ML models here
│
├── requirements.txt
└── README.md
```
---

## 4. How to Run the Project

### 4.1 Setup

```bash
git clone https://github.com/rendell-padu02/AI-Business-Insights-Platform.git
cd AI-Business-Insights-Platform
python -m venv .venv
```

#### Windows:
```bash .venv\Scripts\activate ```
#### macOS / Linux:
```bash source .venv/bin/activate ```

```bash pip install -r requirements.txt```

### 4.2 Prepare Data

Place the raw CSVs in data/raw/:
superstore.csv
Telco_customer_churn.csv

Then run the cleaning script: 
```bash python src/data_cleaning.py ```

This will create:
- data/processed/superstore_clean.csv
- data/processed/telco_churn_clean.csv

### 4.3 Run the Streamlit App
```bash streamlit run app/app.py ```

Open the URL shown in the terminal:
```bash http://localhost:8501 ```

---

## 5. Churn Prediction Models

For the Telco customer churn dataset, two supervised models were trained and compared:

- **Model 1:** Logistic Regression  
- **Model 2:** Random Forest Classifier  

### 5.1 Evaluation Metrics

Using a hold-out test set (20% split, stratified by churn label):

| Metric        | Logistic Regression | Random Forest |
|--------------|---------------------|---------------|
| Accuracy     | **0.9120**          | 0.8928        |
| Precision    | 0.7765              | 0.7636        |
| Recall       | **0.9385**          | 0.8636        |
| F1-Score     | **0.8499**          | 0.8105        |
| ROC-AUC      | **0.9746**          | 0.9446        |

**Interpretation (business-friendly):**

- Logistic Regression gives **higher recall** and F1, making it better at catching customers who are likely to churn (fewer false negatives).
- Random Forest performs slightly worse overall but still provides strong performance and feature importance insights.
- For this use case, we would recommend **Logistic Regression as the primary model**, especially if the business cost of missing a churner is high.

### 5.2 Key Learnings

- Class imbalance and cost of false negatives make **recall** and **ROC-AUC** especially important.
- Interpretable models (like Logistic Regression) are easier to explain to business stakeholders while still delivering high performance.

---

## 6. Defining Business Task
1. **Business Problem**
   - “How can we give non-technical stakeholders AI-augmented insights on sales and customer churn without writing code?”
2. **Data**
   - Superstore: 9,994 orders, 21 columns (sales, profit, discount, category, region, etc.).
   - Telco churn: 7,043 customers, 33 columns (contract type, tenure, charges, churn label, etc.).
3. **EDA & Data Quality**
   - Mention `ydata_profiling` use.
   - Key findings: which regions/categories are profitable, churn distributions, missing values in `Churn Reason`, etc.
4. **Modeling**
   - Explain Logistic Regression vs RandomForest, train/test split, metrics (use the table above).
   - Show ROC curves if you saved them.
5. **Dashboard Design**
   - Tabs/pages in Streamlit: Superstore overview, Churn analytics, Forecast.
   - Business questions each tab answers.
6. **AI / “Augmented Insights” Layer**
   - Prophet forecast for future sales.
   - Auto-insights functions summarizing risky discounts, churn drivers, etc.
7. **Impact & “What Would a Business Do With This?”**
   - Examples: adjust discount policy, target at-risk churn segments, prioritize regions/categories.
8. **Limitations & Next Steps**
   - Data is historical and synthetic; would need real data + cost assumptions for production.
   - Could add “what-if” simulations, SHAP explanations, or auto-generated narrative reports.

---

## 🙌 Acknowledgements

This project uses the following public datasets:
- Superstore Sales Dataset
- IBM Telco Customer Churn Dataset
