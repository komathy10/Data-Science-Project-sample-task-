📌 Project Overview
This project builds intelligence into payroll systems using Machine Learning.
It applies data analysis and ML models to provide salary prediction, attrition risk analysis, anomaly detection, and actionable HR insights.
The goal is to transform payroll from a record-keeping system into a data-driven decision support system.

🎯 Objective
To enhance payroll management by using Machine Learning models to:
Predict employee salaries
Identify attrition risk
Detect payroll anomalies
Provide business insights for HR decision-making

🛠 Tools & Technologies
Python
Pandas, NumPy – Data manipulation
Matplotlib, Seaborn – Data visualization
Scikit-learn – Machine Learning models
Jupyter Notebook – Analysis & reporting

📂 Dataset Description
The dataset contains payroll and employee-related attributes:
EmpID – Employee ID
Name – Employee Name
Department – Engineering, Sales, HR, Finance, etc.
Salary – Monthly salary (INR)
Attendance – Attendance percentage
Leaves – Number of leaves taken
Performance – Performance rating
YearsExperience – Years of experience
TaxSlab, MonthlyTax, PhoneNo
Additional engineered features were created to improve model performance.

🔍 Exploratory Data Analysis (EDA)
EDA was performed to understand payroll patterns and trends:
Salary distribution analysis
Department-wise salary comparison
Attendance and performance analysis
Correlation analysis
Visualizations including histograms, bar charts, scatter plots, and heatmaps
All EDA results are included in the Jupyter Notebook and exported as images.

⚙️ Feature Engineering
New features were created to add business intelligence:
Salary per year of experience
Experience category (Junior, Mid, Senior, Expert)
Performance category
Low attendance and high leave flags
Productivity score
Attrition risk (target variable)

🤖 Machine Learning Models

1️⃣ Salary Prediction Model (Regression)
Algorithm: Random Forest Regressor
Features: Experience, attendance, leaves, performance, department
Evaluation Metrics:
MAE (Mean Absolute Error)
RMSE (Root Mean Squared Error)
R² Score

2️⃣ Attrition Prediction Model (Classification)
Algorithm: Random Forest Classifier
Predicts the probability of employee attrition
Evaluation Metrics:
Accuracy
Precision
Recall
F1 Score

🚨 Anomaly Detection
Payroll anomalies were detected using:
Statistical thresholds
IQR method
Isolation Forest
This helps identify unusual salaries, excessive leaves, or abnormal attendance patterns.

📊 Business Insights (What AI Adds to Payroll)
Enables fair and data-driven salary decisions
Predicts attrition risk for proactive HR action
Identifies top performers and productivity patterns
Detects payroll anomalies and potential fraud
Supports strategic HR planning and budgeting.

📈 Key Outcomes
Improved payroll transparency
Reduced attrition risk through early detection
Data-backed salary planning

Enhanced HR decision-making using AI
