# pages/2_DOE_Analysis.py

import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from itertools import combinations
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Set the page configuration
st.set_page_config(page_title="DOE/Regression Analysis", layout="wide")

# Title
st.title("🔬 DOE/Regression Analysis: Factors Affecting Satisfaction")
st.markdown("## تحليل التصميم التجريبي/الانحدار: العوامل المؤثرة على الرضا")

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
        # Drop the 'ID' column as it's not needed for analysis
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

# Sidebar for Factor and Interaction Selection
st.sidebar.header("🔧 Selection Panel / لوحة الاختيار")

# **A. Factor Selection**
st.sidebar.subheader("1. Select Factors / اختر العوامل:")
available_factors = df.columns.tolist()

# Remove response variable
response_variable = 'satisfaction'
if response_variable in available_factors:
    available_factors.remove(response_variable)

selected_factors = st.sidebar.multiselect(
    "Select Factors Influencing Satisfaction / اختر العوامل المؤثرة على الرضا:",
    options=available_factors,
    default=available_factors  # Default select all factors
)

# **B. Interaction Selection**
st.sidebar.subheader("2. Select Two-Way Interactions / اختر التفاعلات الثنائية:")
# Generate all possible two-way interactions from selected factors
if len(selected_factors) < 2:
    st.sidebar.warning("Please select at least two factors to analyze interactions.")
    st.sidebar.warning("يرجى اختيار عاملين على الأقل لتحليل التفاعلات.")
    selected_interactions = []
else:
    possible_interactions = [f"{pair[0]}*{pair[1]}" for pair in combinations(selected_factors, 2)]
    selected_interactions = st.sidebar.multiselect(
        "Select Two-Way Interactions to Include / اختر التفاعلات الثنائية التي تريد تضمينها:",
        options=possible_interactions,
        default=[]
    )

# **C. Model Execution Button**
st.sidebar.subheader("3. Run Analysis / تشغيل التحليل")
run_analysis = st.sidebar.button("🔍 Run DOE Analysis / تشغيل تحليل التصميم التجريبي")

# Helper Functions
def escape_variable(var_name):
    """
    Enclose variable names with backticks to handle spaces and special characters.
    Use C() to indicate categorical variables.
    """
    if df[var_name].dtype == 'object' or pd.api.types.is_categorical_dtype(df[var_name]):
        return f"C(`{var_name}`)"
    else:
        return f"`{var_name}`"

def escape_interaction(interaction):
    """
    Enclose each variable in an interaction term with backticks and C() if categorical.
    """
    vars_in_interaction = interaction.split('*')
    escaped = '*'.join([escape_variable(var) for var in vars_in_interaction])
    return escaped

# Main Content Area
if run_analysis:
    if not selected_factors:
        st.error("No factors selected. Please select at least one factor for analysis.")
        st.error("لم يتم اختيار أي عوامل. يرجى اختيار عامل واحد على الأقل للتحليل.")
    else:
        # **1. Prepare the Formula for Regression**
        # Escape variable names
        escaped_factors = [escape_variable(var) for var in selected_factors]
        formula = 'satisfaction ~ ' + ' + '.join(escaped_factors)
        
        # Escape interaction terms
        if selected_interactions:
            escaped_interactions = [escape_interaction(interaction) for interaction in selected_interactions]
            formula += ' + ' + ' + '.join(escaped_interactions)
        
        st.markdown("### **Regression Formula / صيغة الانحدار:**")
        st.code(formula, language='python')
        
        # **2. Fit the Regression Model**
        try:
            model = smf.ols(formula=formula, data=df).fit()
        except Exception as e:
            st.error(f"Error fitting the model: {e}")
            st.error(f"خطأ في تركيب النموذج: {e}")
            st.stop()
        
        # **3. Display Model Summary**
        st.markdown("### **Model Summary / ملخص النموذج:**")
        st.text(model.summary())
    
        # **4. ANOVA Table**
        st.markdown("### **ANOVA Table / جدول تحليل التباين:**")
        try:
            anova_table = sm.stats.anova_lm(model, typ=2)  # Type II ANOVA
            st.dataframe(anova_table)
        except Exception as e:
            st.error(f"Error generating ANOVA table: {e}")
            st.error(f"خطأ في إنشاء جدول تحليل التباين: {e}")
    
        # **5. Interpretation of Results**
        st.markdown("### **Interpretation / تفسير النتائج:**")
        st.markdown("""
        - **Significant Factors / العوامل ذات الدلالة الإحصائية:** Factors with p-values < 0.05 are considered statistically significant.
        - **Interaction Effects / تأثيرات التفاعل:** Significant interaction terms indicate that the effect of one factor depends on the level of another factor.
        - **Model Fit / ملاءمة النموذج:** R-squared indicates the proportion of variance explained by the model.
        
        ---
        
        - **العوامل ذات الدلالة الإحصائية:** العوامل ذات قيم p أقل من 0.05 تعتبر ذات دلالة إحصائية.
        - **تأثيرات التفاعل:** التفاعلات ذات الدلالة تشير إلى أن تأثير عامل يعتمد على مستوى عامل آخر.
        - **ملاءمة النموذج:** معامل التحديد R-squared يشير إلى نسبة التباين التي يفسرها النموذج.
        """)
        
        # **6. Download ANOVA Table**
        csv_anova = anova_table.to_csv().encode('utf-8')
        st.download_button(
            label="📥 Download ANOVA Table as CSV / تنزيل جدول ANOVA كملف CSV",
            data=csv_anova,
            file_name='anova_table.csv',
            mime='text/csv',
        )
        
        # **7. Visualizing Significant Factors (Optional)**
        st.markdown("### **Visualizing Significant Factors / تصور العوامل ذات الدلالة الإحصائية:**")
        significant_factors = anova_table[anova_table['PR(>F)'] < 0.05].index.tolist()
        if significant_factors:
            for factor in significant_factors:
                if '*' in factor:
                    # Interaction term
                    factors = factor.split('*')
                    factor1 = factors[0].strip('C(``)' )
                    factor2 = factors[1].strip('C(``)' )
                    st.write(f"**Interaction: {factor1} × {factor2} / تفاعل: {factor1} × {factor2}**")
                    fig, ax = plt.subplots()
                    sns.boxplot(x=factor1, y='satisfaction', hue=factor2, data=df)
                    ax.set_title(f"Interaction Effect: {factor1} × {factor2} / تأثير التفاعل: {factor1} × {factor2}")
                    ax.set_xlabel(f"{factor1} / {factor1}")
                    ax.set_ylabel("Satisfaction Score / درجة الرضا")
                    st.pyplot(fig)
                else:
                    # Main effect
                    factor_clean = factor.strip('C(``)' )
                    st.write(f"**Factor: {factor_clean} / العامل: {factor_clean}**")
                    fig, ax = plt.subplots()
                    sns.boxplot(x=factor, y='satisfaction', data=df)
                    ax.set_title(f"Effect of {factor_clean} on Satisfaction / تأثير {factor_clean} على الرضا")
                    ax.set_xlabel(f"{factor_clean} / {factor_clean}")
                    ax.set_ylabel("Satisfaction Score / درجة الرضا")
                    st.pyplot(fig)
        else:
            st.write("No significant factors found based on the selected model.")
            st.write("لم يتم العثور على عوامل ذات دلالة إحصائية بناءً على النموذج المختار.")
