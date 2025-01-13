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
        st.error(f"لم يتم العثور على ملف البيانات في المسار: {file_path}. يرجى التأكد من وجود الملف في مجلد 'data/'.")
        return None
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        # Drop the 'ID' column as it's not needed for EDA
        if 'ID' in df.columns:
            df = df.drop(columns=['ID'])
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.error(f"خطأ في تحميل البيانات: {e}")
        return None

df = load_data()

if df is None:
    st.stop()

# Sidebar for Filters
st.sidebar.header("🔧 Filters")

# A. Adding Interactive Filters

# Filter by Customer Type
customer_types = st.sidebar.multiselect(
    "Select Customer Type / اختر نوع العميل:",
    options=df['Customer type'].unique(),
    default=df['Customer type'].unique()
)
filtered_df = df[df['Customer type'].isin(customer_types)]

# Filter by Gender
genders = st.sidebar.multiselect(
    "Select Gender / اختر الجنس:",
    options=df['Gender'].unique(),
    default=df['Gender'].unique()
)
filtered_df = filtered_df[filtered_df['Gender'].isin(genders)]

# Filter by Age Group
age_groups = st.sidebar.multiselect(
    "Select Age Group / اختر فئة العمر:",
    options=df['Age'].unique(),
    default=df['Age'].unique()
)
filtered_df = filtered_df[filtered_df['Age'].isin(age_groups)]

# Filter by City
cities = st.sidebar.multiselect(
    "Select City / اختر المدينة:",
    options=df['City'].unique(),
    default=df['City'].unique()
)
filtered_df = filtered_df[filtered_df['City'].isin(cities)]

# Filter by Multiple Visits
multiple_visits = st.sidebar.multiselect(
    "Multiple Visits / زيارات متعددة:",
    options=df['multiple visits'].unique(),
    default=df['multiple visits'].unique()
)
filtered_df = filtered_df[filtered_df['multiple visits'].isin(multiple_visits)]

# Filter by Nationality
nationalities = st.sidebar.multiselect(
    "Select Nationality / اختر الجنسية:",
    options=df['nationality'].unique(),
    default=df['nationality'].unique()
)
filtered_df = filtered_df[filtered_df['nationality'].isin(nationalities)]

# Filter by Duration of Stay
st.sidebar.markdown("### Duration of Stay (hours) / مدة الإقامة (ساعات)")
min_duration = int(filtered_df['duration of stay'].min())
max_duration = int(filtered_df['duration of stay'].max())
duration_range = st.sidebar.slider(
    "Select Duration of Stay / اختر مدة الإقامة:",
    min_value=min_duration,
    max_value=max_duration,
    value=(min_duration, max_duration)
)
filtered_df = filtered_df[
    (filtered_df['duration of stay'] >= duration_range[0]) &
    (filtered_df['duration of stay'] <= duration_range[1])
]

# Filter by Satisfaction Score
st.sidebar.markdown("### Satisfaction Score / درجة الرضا")
min_satisfaction = float(filtered_df['satisfaction'].min())
max_satisfaction = float(filtered_df['satisfaction'].max())
satisfaction_range = st.sidebar.slider(
    "Select Satisfaction Score / اختر درجة الرضا:",
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
    label="📥 Download Filtered Data as CSV / تنزيل البيانات المصفاة كملف CSV",
    data=csv,
    file_name='filtered_customer_satisfaction.csv',
    mime='text/csv',
)

# Main Content Area

# 1. Data Overview
st.header("📄 Data Overview / نظرة عامة على البيانات")

st.subheader("Shape of the Dataset / شكل مجموعة البيانات")
st.write(f"**Rows / الصفوف:** {filtered_df.shape[0]} | **Columns / الأعمدة:** {filtered_df.shape[1]}")
st.markdown("**الصفوف:** {} | **الأعمدة:** {}".format(filtered_df.shape[0], filtered_df.shape[1]))

st.subheader("Column Data Types / أنواع البيانات لكل عمود")
st.write(pd.DataFrame(filtered_df.dtypes, columns=["Data Type"]))
st.markdown("**أنواع البيانات لكل عمود:**")
st.write(pd.DataFrame(filtered_df.dtypes, columns=["أنواع البيانات"]))

st.subheader("Missing Values / القيم المفقودة")
st.write(filtered_df.isnull().sum())
st.markdown("**القيم المفقودة لكل عمود:**")
st.write(filtered_df.isnull().sum())

# 2. Descriptive Statistics
st.header("📈 Descriptive Statistics / الإحصائيات الوصفية")

st.subheader("Numerical Features / الخصائص العددية")
st.write(filtered_df.select_dtypes(include=[np.number]).describe())
st.markdown("**الخصائص العددية:**")
st.write(filtered_df.select_dtypes(include=[np.number]).describe())

st.subheader("Categorical Features / الخصائص الفئوية")
st.markdown("**الخصائص الفئوية:**")
categorical_cols = filtered_df.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    st.write(f"**{col}** / **{col}**")
    st.write(filtered_df[col].value_counts())
    st.write("")

# 3. Visualizations
st.header("📊 Data Visualizations / التصورات البيانية")

# B. Dynamic Visualizations Based on Filters

# Satisfaction Distribution
st.subheader("🎯 Satisfaction Distribution / توزيع درجات الرضا")
st.markdown("**توزيع درجات الرضا:**")
fig1, ax1 = plt.subplots()
sns.histplot(filtered_df['satisfaction'], bins=30, kde=True, ax=ax1, color='skyblue')
ax1.set_xlabel("Satisfaction Score / درجة الرضا")
ax1.set_ylabel("Frequency / التكرار")
st.pyplot(fig1)

# Satisfaction by Customer Type
st.subheader("🧾 Satisfaction by Customer Type / الرضا حسب نوع العميل")
st.markdown("**الرضا حسب نوع العميل:**")
fig2, ax2 = plt.subplots()
sns.boxplot(x='Customer type', y='satisfaction', data=filtered_df, ax=ax2)
ax2.set_xlabel("Customer Type / نوع العميل")
ax2.set_ylabel("Satisfaction Score / درجة الرضا")
st.pyplot(fig2)

# Satisfaction by Gender
st.subheader("👥 Satisfaction by Gender / الرضا حسب الجنس")
st.markdown("**الرضا حسب الجنس:**")
fig3, ax3 = plt.subplots()
sns.boxplot(x='Gender', y='satisfaction', data=filtered_df, ax=ax3)
ax3.set_xlabel("Gender / الجنس")
ax3.set_ylabel("Satisfaction Score / درجة الرضا")
st.pyplot(fig3)

# Satisfaction by Age Group
st.subheader("📅 Satisfaction by Age Group / الرضا حسب فئة العمر")
st.markdown("**الرضا حسب فئة العمر:**")
fig4, ax4 = plt.subplots()
sns.boxplot(x='Age', y='satisfaction', data=filtered_df, ax=ax4)
ax4.set_xlabel("Age Group / فئة العمر")
ax4.set_ylabel("Satisfaction Score / درجة الرضا")
st.pyplot(fig4)

# Satisfaction by City
st.subheader("🌆 Satisfaction by City / الرضا حسب المدينة")
st.markdown("**الرضا حسب المدينة:**")
fig5, ax5 = plt.subplots()
sns.boxplot(x='City', y='satisfaction', data=filtered_df, ax=ax5)
ax5.set_xlabel("City / المدينة")
ax5.set_ylabel("Satisfaction Score / درجة الرضا")
st.pyplot(fig5)

# C. Displaying Top Nationalities Dynamically
st.subheader("🌍 Satisfaction by Nationality (Top 10) / الرضا حسب الجنسية (أفضل 10)")
st.markdown("**الرضا حسب الجنسية (أفضل 10):**")

# Determine the Top 10 Nationalities in the filtered data
top_nationalities = filtered_df['nationality'].value_counts().nlargest(10).index
fig6, ax6 = plt.subplots(figsize=(10,6))
sns.boxplot(x='nationality', y='satisfaction', data=filtered_df[filtered_df['nationality'].isin(top_nationalities)], ax=ax6)
ax6.set_xlabel("Nationality / الجنسية")
ax6.set_ylabel("Satisfaction Score / درجة الرضا")
ax6.set_xticklabels(ax6.get_xticklabels(), rotation=45)
st.pyplot(fig6)

# Scatter Plot: Duration of Stay vs. Satisfaction
st.subheader("⏰ Duration of Stay vs. Satisfaction / مدة الإقامة مقابل درجة الرضا")
st.markdown("**مدة الإقامة مقابل درجة الرضا:**")
fig7 = px.scatter(
    filtered_df, 
    x='duration of stay', 
    y='satisfaction', 
    color='Age', 
    title='Duration of Stay vs. Satisfaction / مدة الإقامة مقابل درجة الرضا',
    labels={'duration of stay':'Duration of Stay (hours) / مدة الإقامة (ساعات)', 'satisfaction':'Satisfaction Score / درجة الرضا'},
    hover_data=['Customer type', 'City', 'nationality']
)
st.plotly_chart(fig7, use_container_width=True)

# Correlation Heatmap
st.subheader("📈 Correlation Heatmap / خريطة الحرارة للارتباط")
st.markdown("**خريطة الحرارة للارتباط:**")
corr = filtered_df.select_dtypes(include=[np.number]).corr()
fig8, ax8 = plt.subplots(figsize=(10,8))
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax8)
st.pyplot(fig8)

# 4. Correlation Analysis
st.header("🔗 Correlation Analysis / تحليل الارتباط")
st.markdown("## تحليل الارتباط")

st.subheader("📈 Pearson Correlation Matrix / مصفوفة ارتباط بيرسون")
st.markdown("**مصفوفة ارتباط بيرسون:**")
corr_matrix = filtered_df.select_dtypes(include=[np.number]).corr()
fig, ax = plt.subplots(figsize=(12,10))
sns.heatmap(corr_matrix, annot=True, cmap='viridis', ax=ax)
st.pyplot(fig)

st.subheader("📊 Pairplot / مخطط التوزيع المتعدد للمتغيرات العددية")
st.markdown("**مخطط التوزيع المتعدد للمتغيرات العددية:**")
pairplot_fig = sns.pairplot(filtered_df.select_dtypes(include=[np.number]))
st.pyplot(pairplot_fig)

# 5. Insights
st.header("💡 Key Insights / الرؤى الرئيسية")
st.markdown("## الرؤى الرئيسية")

# D. Adding Summary Statistics Based on Filters

# A. Overall Satisfaction
average_satisfaction = filtered_df['satisfaction'].mean()
median_satisfaction = filtered_df['satisfaction'].median()
std_satisfaction = filtered_df['satisfaction'].std()

st.markdown(f"""
### **A. Overall Satisfaction / الرضا العام**
- **Average Satisfaction / متوسط الرضا:** {average_satisfaction:.2f}
- **Median Satisfaction / الوسيط:** {median_satisfaction:.2f}
- **Standard Deviation / الانحراف المعياري:** {std_satisfaction:.2f}
- **Observation / ملاحظة:** Satisfaction scores are widely spread, indicating varied customer experiences.
- **ملاحظة:** درجات الرضا منتشرة على نطاق واسع، مما يشير إلى تجارب عملاء متنوعة.
""")

# B. Customer Type Influence
dominant_type = filtered_df['Customer type'].mode()[0]
dominant_type_percentage = (filtered_df['Customer type'].value_counts(normalize=True) * 100).max()

st.markdown(f"""
### **B. Customer Type Influence / تأثير نوع العميل**
- **Dominant Type / النوع السائد:** `{dominant_type}` customers constitute {dominant_type_percentage:.2f}%.
- **Insight / رؤية:** Focus on maintaining high satisfaction levels among `{dominant_type}` can significantly impact overall metrics.
- **رؤية:** التركيز على الحفاظ على مستويات عالية من الرضا بين `{dominant_type}` يمكن أن يؤثر بشكل كبير على المقاييس العامة.
""")

# C. Gender Differences
male_count = filtered_df['Gender'].value_counts().get('m', 0)
male_percentage = (filtered_df['Gender'].value_counts(normalize=True) * 100).get('m', 0)
female_count = filtered_df['Gender'].value_counts().get('f', 0)
female_percentage = (filtered_df['Gender'].value_counts(normalize=True) * 100).get('f', 0)

st.markdown(f"""
### **C. Gender Differences / اختلافات الجنس**
- **Male / ذكر:** {male_count} ({male_percentage:.2f}%%).
- **Female / أنثى:** {female_count} ({female_percentage:.2f}%%).
- **Insight / رؤية:** Analyze if gender influences satisfaction and tailor services accordingly.
- **رؤية:** تحليل ما إذا كان الجنس يؤثر على الرضا وتخصيص الخدمات بناءً على ذلك.
""")

# D. Age Group Trends
majority_age_group = filtered_df['Age'].mode()[0]
majority_age_percentage = (filtered_df['Age'].value_counts(normalize=True) * 100).max()

st.markdown(f"""
### **D. Age Group Trends / اتجاهات فئات العمر**
- **Majority Age Group / فئة العمر الغالبة:** `{majority_age_group}` at {majority_age_percentage:.2f}%.
- **Insight / رؤية:** Understanding the preferences of the `{majority_age_group}` demographic can enhance satisfaction.
- **رؤية:** فهم تفضيلات الفئة العمرية `{majority_age_group}` يمكن أن يعزز الرضا.
""")

# E. City-Based Satisfaction
primary_city = filtered_df['City'].value_counts().idxmax()
primary_city_percentage = (filtered_df['City'].value_counts(normalize=True) * 100).max()
second_city = filtered_df['City'].value_counts().nlargest(2).index[1]
second_city_percentage = (filtered_df['City'].value_counts(normalize=True) * 100).nlargest(2).iloc[1]

st.markdown(f"""
### **E. City-Based Satisfaction / الرضا حسب المدينة**
- **Primary Cities / المدن الرئيسية:** `{primary_city}` (~{primary_city_percentage:.2f}%%) and `{second_city}` (~{second_city_percentage:.2f}%%).
- **Observation / ملاحظة:** Regional service quality might vary; identify city-specific strengths and weaknesses.
- **ملاحظة:** قد تختلف جودة الخدمة الإقليمية؛ تحديد نقاط القوة والضعف الخاصة بكل مدينة.
""")

# F. Nationality Impact
top_nationality = top_nationalities[0]
top_nationality_percentage = (filtered_df['nationality'].value_counts(normalize=True) * 100).iloc[0]

st.markdown(f"""
### **F. Nationality Impact / تأثير الجنسية**
- **Top Nationality / أعلى جنسية:** `{top_nationality}` at {top_nationality_percentage:.2f}%%.
- **Insight / رؤية:** Catering to the needs of the largest nationality can improve satisfaction significantly.
- **رؤية:** تلبية احتياجات أكبر جنسية يمكن أن يحسن الرضا بشكل كبير.
""")

# G. Multiple Visits and Satisfaction
multiple_visits_yes = filtered_df['multiple visits'].value_counts().get('Yes', 0)
multiple_visits_yes_percentage = (filtered_df['multiple visits'].value_counts(normalize=True) * 100).get('Yes', 0)

st.markdown(f"""
### **G. Multiple Visits and Satisfaction / الزيارات المتعددة والرضا**
- **Multiple Visits / الزيارات المتعددة:** {multiple_visits_yes} ({multiple_visits_yes_percentage:.2f}%%).
- **Insight / رؤية:** Encourage repeat business through loyalty programs to boost satisfaction and retention.
- **رؤية:** تشجيع الأعمال المتكررة من خلال برامج الولاء لتعزيز الرضا والاحتفاظ بالعملاء.
""")

# H. Duration of Stay Correlation
duration_corr = filtered_df['duration of stay'].corr(filtered_df['satisfaction'])

st.markdown(f"""
### **H. Duration of Stay Correlation / ارتباط مدة الإقامة**
- **Pearson Correlation / معامل ارتباط بيرسون:** {duration_corr:.2f}.
- **Insight / رؤية:** Assess if longer stays correlate with higher satisfaction or indicate potential issues.
- **رؤية:** تقييم ما إذا كانت الإقامات الأطول ترتبط بزيادة الرضا أو تشير إلى مشكلات محتملة.
""")

# I. Outliers and Anomalies
min_satisfaction_score = filtered_df['satisfaction'].min()
max_satisfaction_score = filtered_df['satisfaction'].max()

st.markdown(f"""
### **I. Outliers and Anomalies / الشواذ والأنماط غير الطبيعية**
- **Extremes / الحدود القصوى:** Scores as low as {min_satisfaction_score:.2f} and as high as {max_satisfaction_score:.2f} indicate exceptional experiences or critical failures.
- **Action / إجراء:** Investigate outliers for actionable feedback.
- **رؤية:** درجات تصل إلى {min_satisfaction_score:.2f} وتنخفض إلى {max_satisfaction_score:.2f} تشير إلى تجارب استثنائية أو فشل حرج.
- **إجراء:** التحقيق في القيم الشاذة للحصول على ملاحظات قابلة للتنفيذ.
""")

# J. Regional Nationalities
st.markdown(f"""
### **J. Regional Nationalities / الجنسيات الإقليمية**
- **Service Tailoring / تخصيص الخدمة:** High satisfaction among specific nationalities suggests effective service strategies that can be replicated or adjusted for others.
- **رؤية:** الرضا العالي بين جنسيات محددة يشير إلى استراتيجيات خدمة فعالة يمكن تكرارها أو تعديلها للآخرين.
""")

st.markdown("**Note / ملاحظة:** These insights are derived from the filtered dataset and should be further validated with additional data or qualitative feedback.")
st.markdown("**ملاحظة:** هذه الرؤى مستمدة من مجموعة البيانات المصفاة ويجب التحقق منها بشكل أكبر باستخدام بيانات إضافية أو ملاحظات نوعية.")
