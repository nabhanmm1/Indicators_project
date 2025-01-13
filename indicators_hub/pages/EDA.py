# pages/1_EDA_SubApp.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os

# Set the page configuration
st.set_page_config(page_title="Customer Satisfaction EDA", layout="wide")

# Title
st.title("🔍 Exploratory Data Analysis: Customer Satisfaction")
st.markdown("## تحليل استكشافي لبيانات رضا العملاء")

# Load Data with Error Handling
@st.cache_data
def load_data():
    file_path = 'indicators_hub/data/customer_satisfaction.csv'
    if not os.path.exists(file_path):
        st.error(f"Data file not found at path: {file_path}. Please ensure the file exists in the 'data/' directory.")
        return None
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        # Drop the 'ID' column as it's not needed for EDA
        if 'ID' in df.columns:
            df = df.drop(columns=['ID'])
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

df = load_data()

if df is None:
    st.stop()

# Sidebar for Filters
st.sidebar.header("🔧 Filters")

# A. Adding Interactive Filters

# Filter by Customer Type
customer_types = st.sidebar.multiselect(
    "Select Customer Type:",
    options=df['Customer type'].unique(),
    default=df['Customer type'].unique()
)
filtered_df = df[df['Customer type'].isin(customer_types)]

# Filter by Gender
genders = st.sidebar.multiselect(
    "Select Gender:",
    options=df['Gender'].unique(),
    default=df['Gender'].unique()
)
filtered_df = filtered_df[filtered_df['Gender'].isin(genders)]

# Filter by Age Group
age_groups = st.sidebar.multiselect(
    "Select Age Group:",
    options=df['Age'].unique(),
    default=df['Age'].unique()
)
filtered_df = filtered_df[filtered_df['Age'].isin(age_groups)]

# Filter by City
cities = st.sidebar.multiselect(
    "Select City:",
    options=df['City'].unique(),
    default=df['City'].unique()
)
filtered_df = filtered_df[filtered_df['City'].isin(cities)]

# Filter by Multiple Visits
multiple_visits = st.sidebar.multiselect(
    "Multiple Visits:",
    options=df['multiple visits'].unique(),
    default=df['multiple visits'].unique()
)
filtered_df = filtered_df[filtered_df['multiple visits'].isin(multiple_visits)]

# Filter by Nationality
nationalities = st.sidebar.multiselect(
    "Select Nationality:",
    options=df['nationality'].unique(),
    default=df['nationality'].unique()
)
filtered_df = filtered_df[filtered_df['nationality'].isin(nationalities)]

# Filter by Duration of Stay
st.sidebar.markdown("### Duration of Stay (hours)")
min_duration = int(filtered_df['duration of stay'].min())
max_duration = int(filtered_df['duration of stay'].max())
duration_range = st.sidebar.slider(
    "Select Duration of Stay:",
    min_value=min_duration,
    max_value=max_duration,
    value=(min_duration, max_duration)
)
filtered_df = filtered_df[
    (filtered_df['duration of stay'] >= duration_range[0]) &
    (filtered_df['duration of stay'] <= duration_range[1])
]

# Filter by Satisfaction Score
st.sidebar.markdown("### Satisfaction Score")
min_satisfaction = float(filtered_df['satisfaction'].min())
max_satisfaction = float(filtered_df['satisfaction'].max())
satisfaction_range = st.sidebar.slider(
    "Select Satisfaction Score:",
    min_value=min_satisfaction,
    max_value=max_satisfaction,
    value=(min_satisfaction, max_satisfaction)
)
filtered_df = filtered_df[
    (filtered_df['satisfaction'] >= satisfaction_range[0]) &
    (filtered_df['satisfaction'] <= satisfaction_range[1])
]

# Optionally, add a download button for the filtered data
csv = filtered_df.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(
    label="📥 Download Filtered Data as CSV",
    data=csv,
    file_name='filtered_customer_satisfaction.csv',
    mime='text/csv',
)

# Main Content Area

# 1. Data Overview
st.header("📄 Data Overview")
st.markdown("## نظرة عامة على البيانات")

st.subheader("Shape of the Dataset")
st.write(f"**Rows:** {filtered_df.shape[0]} | **Columns:** {filtered_df.shape[1]}")
st.markdown("**الصفوف:** {} | **الأعمدة:** {}".format(filtered_df.shape[0], filtered_df.shape[1]))

st.subheader("Column Data Types")
st.write(pd.DataFrame(filtered_df.dtypes, columns=["Data Type"]))
st.markdown("**أنواع البيانات لكل عمود:**")
st.write(pd.DataFrame(filtered_df.dtypes, columns=["أنواع البيانات"]))

st.subheader("Missing Values")
st.write(filtered_df.isnull().sum())
st.markdown("**القيم المفقودة لكل عمود:**")
st.write(filtered_df.isnull().sum())

# 2. Descriptive Statistics
st.header("📈 Descriptive Statistics")
st.markdown("## الإحصائيات الوصفية")

st.subheader("Numerical Features")
st.write(filtered_df.select_dtypes(include=[np.number]).describe())
st.markdown("**الخصائص العددية:**")
st.write(filtered_df.select_dtypes(include=[np.number]).describe())

st.subheader("Categorical Features")
st.markdown("**الخصائص الفئوية:**")
categorical_cols = filtered_df.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    st.write(f"**{col}**")
    st.write(filtered_df[col].value_counts())
    st.write("")

# 3. Visualizations
st.header("📊 Data Visualizations")
st.markdown("## التصورات البيانية")

# B. Dynamic Visualizations Based on Filters

# Satisfaction Distribution
st.subheader("🎯 Satisfaction Distribution")
st.markdown("**توزيع درجات الرضا:**")
fig1, ax1 = plt.subplots()
sns.histplot(filtered_df['satisfaction'], bins=30, kde=True, ax=ax1, color='skyblue')
ax1.set_xlabel("Satisfaction Score")
ax1.set_ylabel("Frequency")
st.pyplot(fig1)

# Satisfaction by Customer Type
st.subheader("🧾 Satisfaction by Customer Type")
st.markdown("**الرضا حسب نوع العميل:**")
fig2, ax2 = plt.subplots()
sns.boxplot(x='Customer type', y='satisfaction', data=filtered_df, ax=ax2)
ax2.set_xlabel("Customer Type")
ax2.set_ylabel("Satisfaction Score")
st.pyplot(fig2)

# Satisfaction by Gender
st.subheader("👥 Satisfaction by Gender")
st.markdown("**الرضا حسب الجنس:**")
fig3, ax3 = plt.subplots()
sns.boxplot(x='Gender', y='satisfaction', data=filtered_df, ax=ax3)
ax3.set_xlabel("Gender")
ax3.set_ylabel("Satisfaction Score")
st.pyplot(fig3)

# Satisfaction by Age Group
st.subheader("📅 Satisfaction by Age Group")
st.markdown("**الرضا حسب فئة العمر:**")
fig4, ax4 = plt.subplots()
sns.boxplot(x='Age', y='satisfaction', data=filtered_df, ax=ax4)
ax4.set_xlabel("Age Group")
ax4.set_ylabel("Satisfaction Score")
st.pyplot(fig4)

# Satisfaction by City
st.subheader("🌆 Satisfaction by City")
st.markdown("**الرضا حسب المدينة:**")
fig5, ax5 = plt.subplots()
sns.boxplot(x='City', y='satisfaction', data=filtered_df, ax=ax5)
ax5.set_xlabel("City")
ax5.set_ylabel("Satisfaction Score")
st.pyplot(fig5)

# C. Displaying Top Nationalities Dynamically
st.subheader("🌍 Satisfaction by Nationality (Top 10)")
st.markdown("**الرضا حسب الجنسية (أفضل 10):**")

# Determine the Top 10 Nationalities in the filtered data
top_nationalities = filtered_df['nationality'].value_counts().nlargest(10).index
fig6, ax6 = plt.subplots(figsize=(10,6))
sns.boxplot(x='nationality', y='satisfaction', data=filtered_df[filtered_df['nationality'].isin(top_nationalities)], ax=ax6)
ax6.set_xlabel("Nationality")
ax6.set_ylabel("Satisfaction Score")
ax6.set_xticklabels(ax6.get_xticklabels(), rotation=45)
st.pyplot(fig6)

# Scatter Plot: Duration of Stay vs. Satisfaction
st.subheader("⏰ Duration of Stay vs. Satisfaction")
st.markdown("**مدة الإقامة مقابل درجة الرضا:**")
fig7 = px.scatter(
    filtered_df, 
    x='duration of stay', 
    y='satisfaction', 
    color='Age', 
    title='Duration of Stay vs. Satisfaction',
    labels={'duration of stay':'Duration of Stay (hours)', 'satisfaction':'Satisfaction Score'},
    hover_data=['Customer type', 'City', 'nationality']
)
st.plotly_chart(fig7, use_container_width=True)

# Correlation Heatmap
st.subheader("📈 Correlation Heatmap")
st.markdown("**خريطة الحرارة للارتباط:**")
corr = filtered_df.select_dtypes(include=[np.number]).corr()
fig8, ax8 = plt.subplots(figsize=(10,8))
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax8)
st.pyplot(fig8)

# 4. Correlation Analysis
st.header("🔗 Correlation Analysis")
st.markdown("## تحليل الارتباط")

st.subheader("📈 Pearson Correlation Matrix")
st.markdown("**مصفوفة ارتباط بيرسون:**")
corr_matrix = filtered_df.select_dtypes(include=[np.number]).corr()
fig, ax = plt.subplots(figsize=(12,10))
sns.heatmap(corr_matrix, annot=True, cmap='viridis', ax=ax)
st.pyplot(fig)

st.subheader("📊 Pairplot")
st.markdown("**مخطط التوزيع المتعدد للمتغيرات العددية:**")
pairplot_fig = sns.pairplot(filtered_df.select_dtypes(include=[np.number]))
st.pyplot(pairplot_fig)

# 5. Insights
st.header("💡 Key Insights")
st.markdown("## الرؤى الرئيسية")

# D. Adding Summary Statistics Based on Filters
st.markdown("""
### **A. Overall Satisfaction**
- **Average Satisfaction:** {:.2f}
- **Median Satisfaction:** {:.2f}
- **Standard Deviation:** {:.2f}
- **Observation:** Satisfaction scores are widely spread, indicating varied customer experiences.

### **أ. الرضا العام**
- **متوسط الرضا:** {:.2f}
- **الوسيط:** {:.2f}
- **الانحراف المعياري:** {:.2f}
- **ملاحظة:** درجات الرضا منتشرة على نطاق واسع، مما يشير إلى تجارب عملاء متنوعة.
""".format(
    filtered_df['satisfaction'].mean(),
    filtered_df['satisfaction'].median(),
    filtered_df['satisfaction'].std(),
    filtered_df['satisfaction'].mean(),
    filtered_df['satisfaction'].median(),
    filtered_df['satisfaction'].std()
))

st.markdown("""
### **B. Customer Type Influence**
- **Dominant Type:** `{}` customers constitute {:.2f}%.
- **Insight:** Focus on maintaining high satisfaction levels among `{}` can significantly impact overall metrics.

### **ب. تأثير نوع العميل**
- **النوع السائد:** يشكل عملاء `{}` {:.2f}%.
- **رؤية:** التركيز على الحفاظ على مستويات عالية من الرضا بين `{}` يمكن أن يؤثر بشكل كبير على المقاييس العامة.
""".format(
    filtered_df['Customer type'].mode()[0],
    (filtered_df['Customer type'].value_counts(normalize=True) * 100).max(),
    filtered_df['Customer type'].mode()[0]
))

st.markdown("""
### **C. Gender Differences**
- **Male:** {} ({:.2f}%).
- **Female:** {} ({:.2f}%).
- **Insight:** Analyze if gender influences satisfaction and tailor services accordingly.

### **ج. اختلافات الجنس**
- **ذكر:** {} ({:.2f}%).
- **أنثى:** {} ({:.2f}%).
- **رؤية:** تحليل ما إذا كان الجنس يؤثر على الرضا وتخصيص الخدمات بناءً على ذلك.
""".format(
    filtered_df['Gender'].value_counts().get('m', 0),
    (filtered_df['Gender'].value_counts(normalize=True) * 100).get('m', 0),
    filtered_df['Gender'].value_counts().get('f', 0),
    (filtered_df['Gender'].value_counts(normalize=True) * 100).get('f', 0)
))

st.markdown("""
### **D. Age Group Trends**
- **Majority Age Group:** `{}` at {:.2f}%.
- **Insight:** Understanding the preferences of the `{}` demographic can enhance satisfaction.

### **د. اتجاهات فئات العمر**
- **فئة العمر الغالبة:** `{}` بنسبة {:.2f}%.
- **رؤية:** فهم تفضيلات الفئة العمرية `{}` يمكن أن يعزز الرضا.
""".format(
    filtered_df['Age'].mode()[0],
    (filtered_df['Age'].value_counts(normalize=True) * 100).max(),
    filtered_df['Age'].mode()[0]
))

st.markdown("""
### **E. City-Based Satisfaction**
- **Primary Cities:** `{}` (~{:.2f}%) and `{}` (~{:.2f}%).
- **Observation:** Regional service quality might vary; identify city-specific strengths and weaknesses.

### **هـ. الرضا حسب المدينة**
- **المدن الرئيسية:** `{}` (~{:.2f}%) و`{}` (~{:.2f}%).
- **ملاحظة:** قد تختلف جودة الخدمة الإقليمية؛ تحديد نقاط القوة والضعف الخاصة بكل مدينة.
""".format(
    filtered_df['City'].value_counts().idxmax(),
    (filtered_df['City'].value_counts(normalize=True) * 100).max(),
    filtered_df['City'].value_counts().nlargest(2).index[1],
    (filtered_df['City'].value_counts(normalize=True) * 100).nlargest(2).iloc[1]
))

st.markdown("""
### **F. Nationality Impact**
- **Top Nationality:** `{}` at {:.2f}%.
- **Insight:** Catering to the needs of the largest nationality can improve satisfaction significantly.

### **و. تأثير الجنسية**
- **أعلى جنسية:** `{}` بنسبة {:.2f}%.
- **رؤية:** تلبية احتياجات أكبر جنسية يمكن أن يحسن الرضا بشكل كبير.
""".format(
    top_nationalities[0],
    (filtered_df['nationality'].value_counts(normalize=True) * 100).iloc[0]
))

st.markdown("""
### **G. Multiple Visits and Satisfaction**
- **Multiple Visits:** {} ({:.2f}%).
- **Insight:** Encourage repeat business through loyalty programs to boost satisfaction and retention.

### **ز. الزيارات المتعددة والرضا**
- **الزيارات المتعددة:** {} ({:.2f}%).
- **رؤية:** تشجيع الأعمال المتكررة من خلال برامج الولاء لتعزيز الرضا والاحتفاظ بالعملاء.
""".format(
    filtered_df['multiple visits'].value_counts().get('Yes', 0),
    (filtered_df['multiple visits'].value_counts(normalize=True) * 100).get('Yes', 0)
))

st.markdown("""
### **H. Duration of Stay Correlation**
- **Pearson Correlation:** {:.2f}.
- **Insight:** Assess if longer stays correlate with higher satisfaction or indicate potential issues.

### **ح. ارتباط مدة الإقامة**
- **معامل ارتباط بيرسون:** {:.2f}.
- **رؤية:** تقييم ما إذا كانت الإقامات الأطول ترتبط بزيادة الرضا أو تشير إلى مشكلات محتملة.
""".format(
    filtered_df['duration of stay'].corr(filtered_df['satisfaction'])
))

st.markdown("""
### **I. Outliers and Anomalies**
- **Extremes:** Scores as low as {:.2f} and as high as {:.2f} indicate exceptional experiences or critical failures.
- **Action:** Investigate outliers for actionable feedback.

### **ط. الشواذ والأنماط غير الطبيعية**
- **الحدود القصوى:** درجات تصل إلى {:.2f} وتنخفض إلى {:.2f} تشير إلى تجارب استثنائية أو فشل حرج.
- **إجراء:** التحقيق في القيم الشاذة للحصول على ملاحظات قابلة للتنفيذ.
""".format(
    filtered_df['satisfaction'].min(),
    filtered_df['satisfaction'].max()
))

st.markdown("""
### **J. Regional Nationalities**
- **Service Tailoring:** High satisfaction among specific nationalities suggests effective service strategies that can be replicated or adjusted for others.

### **ي. الجنسيات الإقليمية**
- **تخصيص الخدمة:** الرضا العالي بين جنسيات محددة يشير إلى استراتيجيات خدمة فعالة يمكن تكرارها أو تعديلها للآخرين.
""")

st.markdown("**Note:** These insights are derived from the filtered dataset and should be further validated with additional data or qualitative feedback.")
st.markdown("**ملاحظة:** هذه الرؤى مستمدة من مجموعة البيانات المصفاة ويجب التحقق منها بشكل أكبر باستخدام بيانات إضافية أو ملاحظات نوعية.")
