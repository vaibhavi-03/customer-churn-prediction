# Daily Learning Log

## Day 1 — 27 Jun 2026
- Completed Weekend 1 of Customer Churn Prediction project
- Set up GitHub repo, added dataset, ran first data inspection
- Dataset: 7043 rows, 21 cols, 26.5% churn rate, zero missing values

## Day 2 — 28 Jun 2026
- Reviewed all 21 columns of the Telco Churn dataset
- Noted class imbalance: 73.5% No churn vs 26.5% Yes churn
- Plan for Weekend 2: churn by contract type, tenure histogram, heatmap

## Day 3 — 29 Jun 2026
- Revised Pandas: groupby, value_counts, crosstab
- Studied what makes a good EDA — univariate → bivariate → multivariate
- Read about correlation heatmaps and when to use them

## Day 4 — 30 Jun 2026
- Revised ML Internship Module 1 notes (Python + NumPy + Pandas)
- Practiced 2 SQL problems on LeetCode (aggregations + GROUP BY)
- Understood difference between .info() and .describe()

## Day 5 — 1 Jul 2026
- Studied class imbalance handling: oversampling, undersampling, class_weight
- Learned why accuracy is a bad metric for imbalanced datasets
- Will use F1-score + recall as primary metrics in Weekend 4

## Day 6 — 2 Jul 2026
- Revised Statistics: mean, median, variance, standard deviation
- Practiced 1 SQL window function problem (RANK + PARTITION BY)
- Looked up Seaborn docs for heatmap and boxplot syntax

## Day 7 — 3 Jul 2026
- Studied Logistic Regression — how it works, sigmoid function, decision boundary
- This will be the baseline model in Weekend 3
- Revised confusion matrix: TP, FP, TN, FN and what each means for churn

## Day 8 — 4 Jul 2026
- Weekend 2 starts tomorrow — prepped EDA plan
- Will analyze: churn by Contract, tenure distribution, MonthlyCharges vs Churn
- Set up Matplotlib + Seaborn in the project environment

## Day 9 — 5 Jul 2026
- Completed Weekend 2 EDA notebook
- Key finding: Month-to-month customers churn at ~43% vs 11% for yearly contracts
- Correlation heatmap shows tenure and TotalCharges are strongly correlated
- Charts saved: churn distribution, contract type, tenure, monthly charges, heatmap

## Day 10 — 6 Jul 2026
- Reviewed EDA findings and wrote plain-English insights in notebook
- Practiced 2 SQL problems: subqueries + CTEs
- Tomorrow: start thinking about feature encoding strategy for Weekend 3

## Day 11 — 7 Jul 2026
- Studied Label Encoding vs One-Hot Encoding — when to use which
- Telco dataset has many categorical columns: Contract, PaymentMethod, InternetService
- Will apply pd.get_dummies() in Weekend 3 preprocessing

## Day 12 — 8 Jul 2026
- Revised feature scaling: StandardScaler vs MinMaxScaler
- Logistic Regression and SVM are sensitive to scale — must scale before fitting
- Practiced 2 SQL problems: JOIN + GROUP BY combinations

## Day 13 — 9 Jul 2026
- Studied train-test split and why we never train on test data
- Revised confusion matrix: precision, recall, F1-score
- For churn prediction: recall matters more than precision (catching churners is priority)

## Day 14 — 10 Jul 2026
- Weekend 3 starts tomorrow — prepped preprocessing plan
- Columns to encode: gender, Partner, Dependents, PhoneService, Contract, PaymentMethod etc.
- Will drop customerID (not a feature), handle TotalCharges null from pd.to_numeric
- Baseline model: Logistic Regression — simple, interpretable, good starting point

## Day 15 — 11 Jul 2026
- Completed Weekend 3: preprocessing + baseline Logistic Regression model
- One-hot encoded 16 categorical columns using pd.get_dummies()
- Baseline results: ~80% accuracy, 57% recall on churn class
- Next weekend: Random Forest + XGBoost to improve recall beyond 57%

## Day 16 — 12 Jul 2026
- Reviewed baseline confusion matrix: 916 TN, 215 TP, 159 FN, 117 FP
- Studied why 57% recall means we're missing 43% of actual churners
- Designed Streamlit app layout: sidebar inputs, gauge chart, risk signals breakdown
- Decided to merge Weekend 4+5 and ship full app by 17 Jul

## Day 17 — 13 Jul 2026
- Built complete Streamlit frontend: dark theme, Space Grotesk font, Plotly charts
- App features: real-time churn probability gauge, EDA charts, key risk signals
- Created train_model.py to save model artifacts (pkl files) for Streamlit Cloud
- Tested model loading via joblib — all 3 artifacts saved successfully

## Day 18 — 14 Jul 2026
- Fixed streamlit PATH issue on Windows: used python -m streamlit run app.py
- Tested full prediction flow locally — gauge, risk signals, probability all working
- Pushed all files: app.py, train_model.py, requirements.txt, .streamlit/config.toml
- Model pkl files committed to repo for Streamlit Cloud deployment

## Day 19 — 15 Jul 2026
- Deployed app to Streamlit Cloud: share.streamlit.io → connected vaibhavi-03 GitHub
- Set main file: app.py, branch: main — deployment successful
- Updated README: added live demo badge, results table, full project structure
- Project complete: 18 days, 3 notebooks, 1 deployed app

## Day 20 — 16 Jul 2026
- Reviewed full project end to end — data inspection → EDA → preprocessing → deployment
- Wrote project summary for portfolio and LinkedIn post draft
- Practiced 3 SQL problems: window functions + CASE WHEN
- Next project planning: starting a new ML project for placement prep

## Day 21 — 17 Jul 2026
- Final review of ChurnSight app — tested edge cases in sidebar inputs
- Updated GitHub profile README: progress bar now shows Customer Churn 5/5 complete
- Reflected on 21-day streak: consistency > perfection
- Ready for next project and placement season