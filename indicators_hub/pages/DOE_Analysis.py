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

# -----------------------------------------
# 1. Data Loading
# -----------------------------------------
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

# -----------------------------------------
# 2. Sidebar: Data Inspection
# -----------------------------------------
st.sidebar.header("🔍 Data Inspection / فحص البيانات")
if st.sidebar.checkbox("📋 Show Available Columns / عرض الأعمدة المتاحة"):
    st.sidebar.write(df.columns.tolist())

# -----------------------------------------
# 3. Sidebar: Factor and Interaction Selection
# -----------------------------------------
st.sidebar.header("🔧 Selection Panel / لوحة الاختيار")

# A. Factor Selection
st.sidebar.subheader("1. Select Factors / اختر العوامل:")
available_factors = df.columns.tolist()

# Remove the response variable from factor list
response_variable = 'satisfaction'
if response_variable in available_factors:
    available_factors.remove(response_variable)

selected_factors = st.sidebar.multiselect(
    "Select Factors Influencing Satisfaction / اختر العوامل المؤثرة على الرضا:",
    options=available_factors,
    default=available_factors  # Default select all
)

# B. Interaction Selection
st.sidebar.subheader("2. Select Two-Way Interactions / اختر التفاعلات الثنائية:")
if len(selected_factors) < 2:
    st.sidebar.warning("Please select at least two factors to analyze interactions.")
    st.sidebar.warning("يرجى اختيار عاملين على الأقل لتحليل التفاعلات.")
    selected_interactions = []
else:
    # Use ':' for interactions
    possible_interactions = [f"{pair[0]}:{pair[1]}" for pair in combinations(selected_factors, 2)]
    selected_interactions = st.sidebar.multiselect(
        "Select Two-Way Interactions to Include / اختر التفاعلات الثنائية التي تريد تضمينها:",
        options=possible_interactions,
        default=[]
    )

# C. Model Execution Button
st.sidebar.subheader("3. Run Analysis / تشغيل التحليل")
run_analysis = st.sidebar.button("🔍 Run DOE Analysis / تشغيل تحليل التصميم التجريبي")

# -----------------------------------------
# Helper Functions
# -----------------------------------------
def escape_variable(var_name):
    """
    Enclose variable names with backticks to handle spaces and special characters.
    """
    return f"`{var_name}`"

def is_categorical(df, var):
    """
    Determine if a variable is categorical based on its data type.
    """
    return df[var].dtype == 'object' or df[var].dtype.name == 'category'

def process_factor(var, is_cat):
    """
    Apply C() to categorical variables and escape variable names.
    """
    if is_cat:
        return f"C({escape_variable(var)})"
    else:
        return f"{escape_variable(var)}"

def process_interaction(interaction, df):
    """
    Process interaction terms by applying C() to categorical variables.
    Interaction uses ':' to denote two-way interaction in statsmodels.
    """
    var1, var2 = interaction.split(':')
    term1 = process_factor(var1, is_categorical(df, var1))
    term2 = process_factor(var2, is_categorical(df, var2))
    # Return the correct statsmodels syntax: factor1:factor2 for interaction only
    # If you prefer main effects + interaction, you could do factor1 * factor2,
    # but here we stay consistent with ':' only for the two-way interaction
    return f"{term1}:{term2}"

# -----------------------------------------
# 4. Main Analysis
# -----------------------------------------
if run_analysis:
    if not selected_factors:
        st.error("No factors selected. Please select at least one factor for analysis.")
        st.error("لم يتم اختيار أي عوامل. يرجى اختيار عامل واحد على الأقل للتحليل.")
        st.stop()
    else:
        # 4.1 Prepare the Formula for Regression
        escaped_factors = [process_factor(var, is_categorical(df, var)) for var in selected_factors]
        formula = 'satisfaction ~ ' + ' + '.join(escaped_factors)

        # Add interaction terms
        if selected_interactions:
            escaped_interactions = [process_interaction(interaction, df) for interaction in selected_interactions]
            formula += ' + ' + ' + '.join(escaped_interactions)

        # Debug: Show the constructed formula
        st.markdown("### **Regression Formula / صيغة الانحدار:**")
        st.code(formula, language='python')
        st.markdown("**🔎 Constructed Formula / الصيغة المُركبة:**")
        st.write(formula)

        # 4.2 Fit the Regression Model
        try:
            model = smf.ols(formula=formula, data=df).fit()
        except Exception as e:
            st.error(f"Error fitting the model: {e}")
            st.error(f"خطأ في تركيب النموذج: {e}")
            st.stop()

        # 4.3 Display Model Summary
        st.markdown("### **Model Summary / ملخص النموذج:**")
        st.text(model.summary())

        # 4.4 ANOVA Table
        st.markdown("### **ANOVA Table / جدول تحليل التباين:**")
        try:
            anova_table = sm.stats.anova_lm(model, typ=2)  # Type II ANOVA
            st.dataframe(anova_table)
        except Exception as e:
            st.error(f"Error generating ANOVA table: {e}")
            st.error(f"خطأ في إنشاء جدول تحليل التباين: {e}")
            st.stop()

        # 4.5 Interpretation of Results
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

        # 4.6 Download ANOVA Table
        csv_anova = anova_table.to_csv().encode('utf-8')
        st.download_button(
            label="📥 Download ANOVA Table as CSV / تنزيل جدول ANOVA كملف CSV",
            data=csv_anova,
            file_name='anova_table.csv',
            mime='text/csv',
        )

        # 4.7 Visualize Significant Factors
        st.markdown("### **Visualizing Significant Factors / تصور العوامل ذات الدلالة الإحصائية:**")
        significant_factors = anova_table[anova_table['PR(>F)'] < 0.05].index.tolist()

        if significant_factors:
            for factor in significant_factors:
                # Check if it's an interaction (':') or a main effect
                if ':' in factor:
                    # Interaction term
                    var1, var2 = factor.split(':')
                    # Remove 'C(`' and '`)' to clean up variable names
                    var1_clean = var1.replace('C(`', '').replace('`)', '')
                    var2_clean = var2.replace('C(`', '').replace('`)', '')

                    st.write(f"**Interaction: {var1_clean} × {var2_clean} / تفاعل: {var1_clean} × {var2_clean}**")
                    fig, ax = plt.subplots()
                    # Visualize with a boxplot or any relevant plot
                    sns.boxplot(x=var1_clean, y='satisfaction', hue=var2_clean, data=df)
                    ax.set_title(f"Interaction Effect: {var1_clean} × {var2_clean}")
                    ax.set_xlabel(f"{var1_clean} / {var1_clean}")
                    ax.set_ylabel("Satisfaction Score / درجة الرضا")
                    st.pyplot(fig)
                else:
                    # Main effect
                    var_clean = factor.replace('C(`', '').replace('`)', '')

                    st.write(f"**Factor: {var_clean} / العامل: {var_clean}**")
                    fig, ax = plt.subplots()
                    sns.boxplot(x=var_clean, y='satisfaction', data=df)
                    ax.set_title(f"Effect of {var_clean} on Satisfaction / تأثير {var_clean} على الرضا")
                    ax.set_xlabel(f"{var_clean} / {var_clean}")
                    ax.set_ylabel("Satisfaction Score / درجة الرضا")
                    st.pyplot(fig)
        else:
            st.write("No significant factors found based on the selected model.")
            st.write("لم يتم العثور على عوامل ذات دلالة إحصائية بناءً على النموذج المختار.")
