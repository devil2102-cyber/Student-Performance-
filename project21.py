import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import streamlit as st

#dataset named student_data1.csv made using pandas and random module of python 
#this dataset contains information about students' marks, attendance, and logins

# Load dataset (replace with actual file)
# For portability, use a relative path instead of an absolute path:
data = pd.read_csv("student_data1.csv")

# Clean column names: remove spaces and lowercase
data.columns = [col.strip().lower().replace(" ", "") for col in data.columns]
# Print columns for debugging
# print("Columns:", data.columns.tolist())

# Fix attendance column: handle 'attendance(%)' and standardize to 'attendance'
if "attendance(%)" in data.columns:
    data = data.rename(columns={"attendance(%)": "attendance"})

# Calculate average marks, attendance, and logins
marks_avg = data["marks"].mean()
attendance_avg = data["attendance"].mean()
logins_avg = data["logins"].mean()
print(f"Average Marks: {marks_avg:.2f}")
print(f"Average Attendance: {attendance_avg:.2f}")
print(f"Average Logins: {logins_avg:.2f}")

# Correlation matrix
corr = data[["marks", "attendance", "logins"]].corr()
print("\nCorrelation Matrix:")
print(corr)

# correlation heatmap
plt.figure(figsize=(7,5))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Between Features")
plt.tight_layout()
plt.show()

#Define risk based on performance thresholds
data["risk"] = np.where((data["marks"] < 40) | (data["attendance"] < 60), 1, 0)

#absentee impact (barplot: risk vs attendance)
plt.figure(figsize=(7,5))
sns.barplot(x="risk", y="attendance", data=data, hue="risk", palette="Reds", legend=False)
plt.xlabel("Risk Level (0 = Safe, 1 = At Risk)")
plt.ylabel("Attendance")
plt.title("Absentee Impact on Student Risk")
plt.tight_layout()
plt.show()

# top vs struggling students (barplot: risk vs marks)
plt.figure(figsize=(7,5))
sns.barplot(x="risk", y="marks", data=data, hue="risk", palette="Blues", legend=False)
plt.xlabel("Risk Level (0 = Safe, 1 = At Risk)")
plt.ylabel("Marks")
plt.title("Performance Comparison of Students")
plt.tight_layout()
plt.show()

# Select features and target
X = data[["marks", "attendance", "logins"]]
y = data["risk"]

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Train Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Predict on test data
y_pred = model.predict(X_test_scaled)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))

# Dashboard: Academic Intervention Support for At-Risk Students

# number and percentage of at-risk students
num_at_risk = data["risk"].sum()
total_students = len(data)
percent_at_risk = (num_at_risk / total_students) * 100
print(f"\nNumber of At-Risk Students: {num_at_risk} ({percent_at_risk:.1f}%)")

# top 10 at-risk students with lowest marks and attendance
at_risk_students = data[data["risk"] == 1].sort_values(by=["marks", "attendance"])
print("\nTop 10 At-Risk Students (Lowest Marks & Attendance):")
print(at_risk_students[["marks", "attendance", "logins"]].head(10))

# plot of  distribution of marks and attendance for at-risk vs safe students
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
sns.boxplot(x="risk", y="marks", data=data, hue="risk", palette="Set2", legend=False)
plt.title("Marks Distribution by Risk")
plt.xlabel("Risk (0=Safe, 1=At Risk)")
plt.subplot(1,2,2)
sns.boxplot(x="risk", y="attendance", data=data, hue="risk", palette="Set1", legend=False)
plt.title("Attendance Distribution by Risk")
plt.xlabel("Risk (0=Safe, 1=At Risk)")
plt.tight_layout()
plt.show()

# Bar chart: Count of at-risk vs safe students
plt.figure(figsize=(5,4))
sns.countplot(x="risk", data=data, hue="risk", palette="pastel", legend=False)
plt.title("Count of Safe vs At-Risk Students")
plt.xlabel("Risk (0=Safe, 1=At Risk)")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.show()

# plot of time-series of average marks, attendance, and logins by semester 
if "semester" in data.columns:
    plt.figure(figsize=(10,5))
    data_grouped = data.groupby("semester")[["marks", "attendance", "logins"]].mean()
    data_grouped.plot(marker='o', ax=plt.gca())
    plt.title("Trend Analysis: Average Metrics by Semester")
    plt.xlabel("Semester")
    plt.ylabel("Average Value")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# plot of engagement score: weighted sum of attendance and logins

data["engagement_score"] = 0.7 * data["attendance"] + 0.3 * (data["logins"] / data["logins"].max() * 100)
print("\nSample Engagement Scores:")
print(data[["marks", "attendance", "logins", "engagement_score"]].head())

# plot of engagement score distribution
plt.figure(figsize=(7,4))
sns.histplot(data["engagement_score"], bins=20, kde=True, color="purple")
plt.title("Distribution of Engagement Scores")
plt.xlabel("Engagement Score")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.show()

#Recommended actions for at-risk students
def suggest_intervention(row):
    suggestions = []
    if row["marks"] < 40:
        suggestions.append("Academic coaching")
    if row["attendance"] < 60:
        suggestions.append("Attendance counseling")
    if row["engagement_score"] < 60:
        suggestions.append("Mentorship program")
    return ", ".join(suggestions) if suggestions else "No intervention needed"

at_risk_students = data[data["risk"] == 1].copy()
at_risk_students["suggestion"] = at_risk_students.apply(suggest_intervention, axis=1)
print("\nIntervention Suggestions for Top 10 At-Risk Students:")
print(at_risk_students[["marks", "attendance", "logins", "engagement_score", "suggestion"]].sort_values(by=["marks", "attendance"]).head(10))

# Streamlit Dashboard
# Only run Streamlit dashboard if executed with Streamlit, not as a normal script
if __name__ == "__main__" or "streamlit" in __import__("sys").argv[0]:
    # Streamlit performance tips:
    st.set_page_config(page_title="Student Academic Risk Dashboard", layout="wide")

    # Use @st.cache_resource for expensive operations (like model training) to speed up reruns
    @st.cache_resource
    def train_model(X_train, y_train):
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model

    st.title("📊 Student Academic Risk Dashboard")

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Average Marks", f"{marks_avg:.2f}")
    col2.metric("Average Attendance", f"{attendance_avg:.2f}")
    col3.metric("Average Logins", f"{logins_avg:.2f}")

    # At-risk stats
    st.subheader("At-Risk Students Overview")
    st.write(f"**Number of At-Risk Students:** {num_at_risk} ({percent_at_risk:.1f}%)")

    # Show top at-risk students
    if not at_risk_students.empty:
        st.write("**Top 10 At-Risk Students (Lowest Marks & Attendance):**")
        st.dataframe(at_risk_students[["marks", "attendance", "logins", "engagement_score", "suggestion"]].sort_values(by=["marks", "attendance"]).head(10))
    else:
        st.info("No at-risk students found in the current dataset.")

    # Correlation heatmap
    st.subheader("Correlation Between Features")
    fig1, ax1 = plt.subplots(figsize=(7,5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax1)
    st.pyplot(fig1)

    # Absentee impact
    st.subheader("Absentee Impact on Student Risk")
    fig2, ax2 = plt.subplots(figsize=(7,5))
    sns.barplot(x="risk", y="attendance", data=data, hue="risk", palette="Reds", legend=False, ax=ax2)
    ax2.set_xlabel("Risk Level (0 = Safe, 1 = At Risk)")
    ax2.set_ylabel("Attendance")
    st.pyplot(fig2)

    # Performance comparison
    st.subheader("Performance Comparison of Students")
    fig3, ax3 = plt.subplots(figsize=(7,5))
    sns.barplot(x="risk", y="marks", data=data, hue="risk", palette="Blues", legend=False, ax=ax3)
    ax3.set_xlabel("Risk Level (0 = Safe, 1 = At Risk)")
    ax3.set_ylabel("Marks")
    st.pyplot(fig3)

    # Distribution plots
    st.subheader("Distribution of Marks and Attendance by Risk")
    fig4, (ax4a, ax4b) = plt.subplots(1,2, figsize=(12,5))
    sns.boxplot(x="risk", y="marks", data=data, hue="risk", palette="Set2", legend=False, ax=ax4a)
    ax4a.set_title("Marks Distribution by Risk")
    ax4a.set_xlabel("Risk (0=Safe, 1=At Risk)")
    sns.boxplot(x="risk", y="attendance", data=data, hue="risk", palette="Set1", legend=False, ax=ax4b)
    ax4b.set_title("Attendance Distribution by Risk")
    ax4b.set_xlabel("Risk (0=Safe, 1=At Risk)")
    st.pyplot(fig4)

    # Count of at-risk vs safe students
    st.subheader("Count of Safe vs At-Risk Students")
    fig5, ax5 = plt.subplots(figsize=(5,4))
    sns.countplot(x="risk", data=data, hue="risk", palette="pastel", legend=False, ax=ax5)
    ax5.set_xlabel("Risk (0=Safe, 1=At Risk)")
    ax5.set_ylabel("Number of Students")
    st.pyplot(fig5)

    # Trend analysis by semester
    if "semester" in data.columns:
        st.subheader("Trend Analysis: Average Metrics by Semester")
        fig6, ax6 = plt.subplots(figsize=(10,5))
        data_grouped = data.groupby("semester")[["marks", "attendance", "logins"]].mean()
        data_grouped.plot(marker='o', ax=ax6)
        ax6.set_xlabel("Semester")
        ax6.set_ylabel("Average Value")
        ax6.set_title("Trend Analysis: Average Metrics by Semester")
        ax6.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig6)

    # Engagement score distribution
    st.subheader("Distribution of Engagement Scores")
    fig7, ax7 = plt.subplots(figsize=(7,4))
    sns.histplot(data["engagement_score"], bins=20, kde=True, color="purple", ax=ax7)
    ax7.set_xlabel("Engagement Score")
    ax7.set_ylabel("Number of Students")
    st.pyplot(fig7)

    # Model performance
    st.subheader("Model Performance")
    st.text(f"Model Accuracy: {accuracy:.2f}")
    st.text(classification_report(y_test, y_pred))

# NOTE:
# The warning "missing ScriptRunContext! This warning can be ignored when running in bare mode."
# means you are running the script as a normal Python script, not with Streamlit.
# To use the dashboard, run this command in your terminal:
#     streamlit run "c:/Users/HP/Python/Data Science Projects/project2.py"
# This will open the dashboard in your browser and remove the warning.
