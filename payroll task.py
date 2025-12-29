#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_excel(r"C:\Users\ADMIN\Desktop\data set\payroll_dataset.xlsx")


# In[3]:


print(f"\nDataset Shape: {df.shape}")
print(f"Total Employees: {df.shape[0]}")


# In[4]:


print(df.head())


# In[5]:


print(df.dtypes)


# In[6]:


print(df.isnull().sum())


# In[7]:


#Exploratory Data Analysis (EDA)


# In[8]:


print("\n1. STATISTICAL SUMMARY:")
print(df.describe())


# In[9]:


print("\n2. SALARY ANALYSIS:")
print(f"   Average Salary: ₹{df['Salary'].mean():,.2f}")
print(f"   Median Salary: ₹{df['Salary'].median():,.2f}")
print(f"   Min Salary: ₹{df['Salary'].min():,.2f}")
print(f"   Max Salary: ₹{df['Salary'].max():,.2f}")


# In[10]:


print("\n3. ATTENDANCE ANALYSIS:")
print(f"   Average Attendance: {df['Attendance'].mean():.2f}%")


# In[11]:


print("\n4. CORRELATION WITH SALARY:")
numeric_cols = ['Salary', 'Attendance', 'Leaves', 'Performance', 'YearsExperience']
correlation = df[numeric_cols].corr()['Salary'].sort_values(ascending=False)
print(correlation)


# In[12]:


# Visualizations
plt.figure(figsize=(35,30))


# In[18]:


# Plot 1: Salary Distribution
plt.hist(df['Salary'], bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Salary')
plt.ylabel('Frequency')
plt.title('Salary Distribution')


# In[19]:


# Plot 2: Salary by Department
df.groupby('Department')['Salary'].mean().plot(kind='bar', color='coral')
plt.xlabel('Department')
plt.ylabel('Average Salary')
plt.title('Average Salary by Department')


# In[20]:


# Plot 3: Attendance Distribution
plt.hist(df['Attendance'], bins=15, color='lightgreen', edgecolor='black')
plt.xlabel('Attendance %')
plt.ylabel('Frequency')
plt.title('Attendance Distribution')


# In[21]:


# Plot 4: Experience vs Salary
plt.scatter(df['YearsExperience'], df['Salary'], alpha=0.5, color='purple')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Experience vs Salary')


# In[22]:


# Plot 5: Performance vs Salary
plt.scatter(df['Performance'], df['Salary'], alpha=0.5, color='orange')
plt.xlabel('Performance Rating')
plt.ylabel('Salary')
plt.title('Performance vs Salary')


# In[23]:


#FEATURE ENGINEERING


# In[24]:


df_fe = df.copy()


# In[25]:


# Feature 1: Salary per year of experience
df_fe['SalaryPerExp'] = df_fe['Salary'] / (df_fe['YearsExperience'] + 1)


# In[26]:


# Feature 2: Experience Category
df_fe['ExpCategory'] = pd.cut(df_fe['YearsExperience'], 
                                bins=[0, 2, 5, 10, 20], 
                                labels=['Junior', 'Mid', 'Senior', 'Expert'])


# In[28]:


# Feature 3: Performance Category
df_fe['PerfCategory'] = pd.cut(df_fe['Performance'],
                                 bins=[0, 60, 75, 90, 100],
                                 labels=['Poor', 'Average', 'Good', 'Excellent'])


# In[29]:


# Feature 4: Low Attendance Flag
df_fe['LowAttendance'] = (df_fe['Attendance'] < 85).astype(int)


# In[31]:


# Feature 5: Attrition Risk (Target Variable)
# Employees at risk: Low attendance OR high leaves OR low performance
df_fe['AttritionRisk'] = (
    (df_fe['Attendance'] < 85) | 
    (df_fe['Leaves'] > 15) | 
    (df_fe['Performance'] < 60) |
    (df_fe['YearsExperience'] < 1)
).astype(int)


# In[33]:


print("\n✅ NEW FEATURES CREATED:")
new_features = ['SalaryPerExp', 'ExpCategory', 'PerfCategory', 
                'LowAttendance', 'AttritionRisk']

for feat in new_features:
    print(f"   • {feat}")


# In[34]:


df_fe.head()


# In[35]:


#Anomaly Detection


# In[36]:


stat_anomalies = df_fe[
    (df_fe['Attendance'] < 80) |
    (df_fe['Leaves'] > 20) |
    (df_fe['Salary'] < df_fe['Salary'].quantile(0.05)) |
    (df_fe['Salary'] > df_fe['Salary'].quantile(0.95))
].copy()

print(f"\n1. STATISTICAL ANOMALIES: {len(stat_anomalies)} found")


# In[37]:


#Salary Prediction Model


# In[47]:


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score




le_dept = LabelEncoder()
df_model = df_fe.copy()
df_model['DepartmentEncoded'] = le_dept.fit_transform(df_model['Department'])

# Select features for prediction
features = ['YearsExperience', 'Attendance', 'Leaves', 
            'Performance', 'DepartmentEncoded']
X = df_model[features]
y = df_model['Salary']

# Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest Model
rf_salary = RandomForestRegressor(n_estimators=100, random_state=42)
rf_salary.fit(X_train_scaled, y_train)

# Make predictions
y_pred = rf_salary.predict(X_test_scaled)

# Evaluate model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nMODEL PERFORMANCE:")
print(f"   MAE (Mean Absolute Error): ₹{mae:,.2f}")
print(f"   RMSE (Root Mean Squared Error): ₹{rmse:,.2f}")
print(f"   R² Score: {r2:.4f}")
print(f"   Accuracy: {r2 * 100:.2f}%")


# In[48]:


# Visualization: Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Salary')
plt.ylabel('Predicted Salary')
plt.title('Salary Prediction: Actual vs Predicted')
plt.legend()


# In[49]:


#Attrition Prediction Model


# In[53]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


features_attr = ['YearsExperience', 'Attendance', 'Leaves', 
                 'Performance', 'Salary', 'DepartmentEncoded']
X_attr = df_model[features_attr]
y_attr = df_model['AttritionRisk']

print(f"\nTotal employees: {len(y_attr)}")
print(f"At risk: {y_attr.sum()} ({y_attr.mean() * 100:.2f}%)")
print(f"Safe: {(y_attr == 0).sum()} ({(y_attr == 0).mean() * 100:.2f}%)")

# Split data
X_train_attr, X_test_attr, y_train_attr, y_test_attr = train_test_split(
    X_attr, y_attr, test_size=0.2, random_state=42, stratify=y_attr
)

# Scale features
scaler_attr = StandardScaler()
X_train_attr_scaled = scaler_attr.fit_transform(X_train_attr)
X_test_attr_scaled = scaler_attr.transform(X_test_attr)

# Train Random Forest Classifier
rf_attrition = RandomForestClassifier(n_estimators=100, random_state=42)
rf_attrition.fit(X_train_attr_scaled, y_train_attr)

# Make predictions
y_pred_attr = rf_attrition.predict(X_test_attr_scaled)
y_prob_attr = rf_attrition.predict_proba(X_test_attr_scaled)[:, 1]

# Evaluate model
accuracy = accuracy_score(y_test_attr, y_pred_attr)
precision = precision_score(y_test_attr, y_pred_attr)
recall = recall_score(y_test_attr, y_pred_attr)
f1 = f1_score(y_test_attr, y_pred_attr)

print("\nMODEL PERFORMANCE:")
print(f"   Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"   Precision: {precision:.4f}")
print(f"   Recall: {recall:.4f}")
print(f"   F1 Score: {f1:.4f}")


# In[55]:


print("\n🎯 FEATURE IMPORTANCE FOR ATTRITION:")
# Feature Importance
feature_importance_attr = pd.DataFrame({
    'Feature': features_attr,
    'Importance': rf_attrition.feature_importances_
}).sort_values('Importance', ascending=False)
print(feature_importance_attr.to_string(index=False))

# Identify high-risk employees
df_model['AttritionProbability'] = rf_attrition.predict_proba(
    scaler_attr.transform(df_model[features_attr])
)[:, 1]

high_risk = df_model[df_model['AttritionProbability'] > 0.7].sort_values(
    'AttritionProbability', ascending=False
)

print(f"\n⚠️ HIGH RISK EMPLOYEES (Probability > 70%): {len(high_risk)}")
if len(high_risk) > 0:
    print(high_risk[['EmpID', 'Name', 'Department', 'AttritionProbability']].head(10))


# In[57]:


# Example new employee
new_employee = {
    'Name': 'Rajesh Verma',
    'Department': 'Engineering',
    'YearsExperience': 5.5,
    'Attendance': 92.0,
    'Leaves': 8,
    'Performance': 82.0
}

print("\n👤 NEW EMPLOYEE PROFILE:")
for key, value in new_employee.items():
    print(f"   {key}: {value}")

# Prepare features for prediction
dept_encoded = le_dept.transform([new_employee['Department']])[0]

# Salary prediction - USE DATAFRAME to avoid warnings
salary_features_df = pd.DataFrame([[
    new_employee['YearsExperience'],
    new_employee['Attendance'],
    new_employee['Leaves'],
    new_employee['Performance'],
    dept_encoded
]], columns=features)  # Use the same feature names from training

salary_features_scaled = scaler.transform(salary_features_df)
predicted_salary = rf_salary.predict(salary_features_scaled)[0]

# Attrition prediction - USE DATAFRAME to avoid warnings
attrition_features_df = pd.DataFrame([[
    new_employee['YearsExperience'],
    new_employee['Attendance'],
    new_employee['Leaves'],
    new_employee['Performance'],
    predicted_salary,  # Use predicted salary
    dept_encoded
]], columns=features_attr)  # Use the same feature names from training

attrition_features_scaled = scaler_attr.transform(attrition_features_df)
attrition_prob = rf_attrition.predict_proba(attrition_features_scaled)[0][1]
attrition_risk = "HIGH" if attrition_prob > 0.5 else "LOW"

print("\n🔮 PREDICTIONS:")
print(f"   💰 Predicted Salary: ₹{predicted_salary:,.2f}")
print(f"   🚨 Attrition Probability: {attrition_prob:.2%}")
print(f"   ⚠️ Attrition Risk: {attrition_risk}")

if attrition_risk == "HIGH":
    print("\n💡 RECOMMENDATIONS:")
    if new_employee['Attendance'] < 85:
        print("   • Improve attendance (currently below 85%)")
    if new_employee['Leaves'] > 15:
        print("   • Reduce excessive leave usage")
    if new_employee['Performance'] < 60:
        print("   • Performance improvement plan needed")
    if new_employee['YearsExperience'] < 2:
        print("   • Provide mentorship for junior employee")



# # Business Insights: What AI Adds to Payroll
# 
# 1.Intelligent Salary Prediction & Fair Pay
# Traditional payroll systems rely on fixed rules and manual decisions. By using machine learning, salary predictions are generated based on experience, performance, attendance, and department. This ensures fair, consistent, and data-driven compensation decisions, reducing bias and improving employee trust.
# 
# 2.Early Attrition Risk Identification
# The attrition prediction model identifies employees who are at high risk of leaving based on behavioral and performance patterns. This allows HR teams to take proactive retention measures such as employee engagement, training programs, or compensation adjustments, reducing turnover costs.
# 
# 3.Productivity and Performance Intelligence
# By combining attendance, performance ratings, and leave patterns into a productivity score, AI provides a holistic view of employee contribution. This helps management identify top performers, detect burnout risks, and make informed decisions for promotions and incentives.
# 
# 4.Payroll Anomaly and Compliance Detection
# Machine learning techniques detect abnormal payroll patterns such as unusually high or low salaries, excessive leaves, or irregular attendance. This improves payroll accuracy, prevents financial leakage, and strengthens internal audit and compliance processes.
# 
# 5.Data-Driven HR Decision Support
# AI-powered payroll insights support strategic HR decisions such as workforce planning, budgeting, and hiring. Predicting salary expectations and attrition probability for new employees helps organizations plan resources more effectively and make confident, data-backed decisions.
# 
