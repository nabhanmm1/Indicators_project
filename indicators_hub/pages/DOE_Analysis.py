#############################
# 1. Imports and Setup
#############################
import streamlit as st
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from itertools import combinations
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Configure Streamlit Page
st.set_page_config(page_title="DOE/Regression Analysis", layout="wide")

#############################
# 2. Load Data
#############################
@st.cache_data
def load_data():
    file_path = 'indicators_hub/data/customer_satisfaction2.csv'
    if not os.path.exists(file_path):
        st.error(f"Data file not found at path: {file_path}.")
        return None
    try:
        df = pd.read_csv(file_path, encoding='utf-8')

        st.write("**DEBUG: Printing all columns with `repr()` to check for hidden characters**")
        for col in df.columns:
            st.write("Column:", repr(col))

        # Example if you still want to rename columns (uncomment if needed):
        # df.rename(columns={
        #     "Customer_type": "Customer_type",
        #     "multiple_visits": "multiple_visits",
        #     "duration_of_stay": "duration_of_stay"
        # }, inplace=True)

        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

df = load_data()
if df is None:
    st.stop()

#############################
# 3. Title and Data Inspection
#############################
st.title("🔬 DOE/Regression Analysis: Factors Affecting Satisfaction")
st.markdown("## تحليل التصميم التجريبي/الانحدار: العوامل المؤثرة على الرضا")

st.sidebar.header("🔍 Data Inspection / فحص البيانات")
if st.sidebar.checkbox("📋 Show Available Columns / عرض الأعمدة المتاحة"):
    st.sidebar.write(df.columns.tolist())

#############################
# 4. Sidebar Selections
#############################
st.sidebar.header("🔧 Selection Panel / لوحة الاختيار")

# A. Factor Selection
response_variable = "satisfaction"  # Make sure this matches your CSV
available_factors = [col for col in df.columns if col != response_variable]

selected_factors = st.sidebar.multiselect(
    "Select Factors Influencing Satisfaction / اختر العوامل المؤثرة على الرضا:",
    options=available_factors,
    default=available_factors  # Select all by default
)

# B. Interaction Selection
st.sidebar.subheader("Select Two-Way Interactions / اختر التفاعلات الثنائية:")
if len(selected_factors) < 2:
    st.sidebar.warning("Please select at least two factors to analyze interactions.")
    selected_interactions = []
else:
    # Use ':' for generating two-way interactions
    possible_interactions = [f"{a}:{b}" for a, b in combinations(selected_factors, 2)]
    selected_interactions = st.sidebar.multiselect(
        "Select Two-Way Interactions to Include / اختر التفاعلات الثنائية:",
        options=possible_interactions,
        default=[]
    )

# Button to run the analysis
run_analysis = st.sidebar.button("🔍 Run DOE Analysis / تشغيل تحليل التصميم التجريبي")

#############################
# 5. Helper Functions
#############################
def is_categorical(df, var):
    """Check if a column is categorical (object or category dtype)."""
    return df[var].dtype == 'object' or df[var].dtype.name == 'category'

def build_formula(response, factors, interactions=None):
    """
    Construct a statsmodels formula string.
    For categorical variables => `C(var)`
    For numeric => var
    For interactions => factor1:factor2
    """
    if interactions is None:
        interactions = []

    # Build main factors
    main_terms = []
    for factor in factors:
        if is_categorical(df, factor):
            main_terms.append(f"C({factor})")
        else:
            main_terms.append(f"{factor}")

    # Build interaction terms
    interaction_terms = []
    for interaction in interactions:
        var1, var2 = interaction.split(':')
        if is_categorical(df, var1):
            term1 = f"C({var1})"
        else:
            term1 = var1

        if is_categorical(df, var2):
            term2 = f"C({var2})"
        else:
            term2 = var2

        # statsmodels uses `var1:var2` to indicate only the interaction
        interaction_terms.append(f"{term1}:{term2}")

    # Combine into a single formula string
    all_terms = main_terms + interaction_terms
    formula_str = f"{response} ~ " + " + ".join(all_terms)
    return formula_str

#############################
# 6. Run Analysis Logic
#############################
if run_analysis:
    if not selected_factors:
        st.error("No factors selected. Please select at least one factor for analysis.")
        st.stop()

    # A. Construct the formula
    formula = build_formula(response_variable, selected_factors, selected_interactions)

    st.markdown("### **Regression Formula / صيغة الانحدار:**")
    st.code(formula, language='python')
    st.markdown("**🔎 Constructed Formula (repr) / الصيغة المُركبة (repr):**")
    st.write(repr(formula))  # <-- This ensures we see *exactly* what's being passed

    # B. Fit the model
    try:
        model = smf.ols(formula=formula, data=df).fit()
    except Exception as e:
        st.error(f"Error fitting the model: {e}")
        st.stop()

    # C. Display Model Summary
    st.markdown("### **Model Summary / ملخص النموذج:**")
    st.text(model.summary())

    # D. ANOVA Table
    st.markdown("### **ANOVA Table / جدول تحليل التباين:**")
    try:
        anova_table = sm.stats.anova_lm(model, typ=2)
        st.dataframe(anova_table)
    except Exception as e:
        st.error(f"Error generating ANOVA table: {e}")

    # E. Interpretation
    st.markdown("### **Interpretation / تفسير النتائج:**")
    st.markdown("""
    - **Significant Factors / العوامل ذات الدلالة الإحصائية:** p-values < 0.05 are generally considered significant.
    - **Interaction Effects / تأثيرات التفاعل:** If an interaction term is significant, the effect of one factor depends on the level of another.
    - **Model Fit / ملاءمة النموذج:** R-squared indicates how much variance is explained by the model.
    """)

    # F. Download ANOVA Table
    csv_anova = anova_table.to_csv().encode('utf-8')
    st.download_button(
        label="📥 Download ANOVA Table as CSV",
        data=csv_anova,
        file_name='anova_table.csv',
        mime='text/csv',
    )

    # G. Visualize Significant Factors
    st.markdown("### **Visualizing Significant Factors**")
    significant_factors = anova_table[anova_table['PR(>F)'] < 0.05].index.tolist()
    if len(significant_factors) == 0:
        st.write("No significant factors found for the selected model.")
    else:
        for factor in significant_factors:
            if ':' in factor:
                # Interaction
                var1, var2 = factor.split(':')
                st.write(f"**Interaction: {var1} × {var2}**")
                fig, ax = plt.subplots()
                sns.boxplot(x=var1, y=response_variable, hue=var2, data=df)
                ax.set_title(f"Interaction: {var1} × {var2}")
                ax.set_xlabel(var1)
                ax.set_ylabel(response_variable)
                st.pyplot(fig)
            else:
                # Main factor
                st.write(f"**Factor: {factor}**")
                fig, ax = plt.subplots()
                sns.boxplot(x=factor, y=response_variable, data=df)
                ax.set_title(f"Effect of {factor} on {response_variable}")
                ax.set_xlabel(factor)
                ax.set_ylabel(response_variable)
                st.pyplot(fig)
