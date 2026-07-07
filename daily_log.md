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