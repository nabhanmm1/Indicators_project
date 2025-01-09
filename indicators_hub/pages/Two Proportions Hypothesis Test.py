import math
import streamlit as st
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

def calculate_test_statistic(p1, p2, n1, n2, finite=False, N1=None, N2=None):
    """
    Calculates the z-test statistic for the difference in proportions.
    If finite=True, applies Finite Population Correction (FPC).
    """
    if finite and N1 and N2:
        # Calculate FPC for each population
        FPC1 = math.sqrt((N1 - n1) / (N1 - 1)) if N1 > 1 else 1
        FPC2 = math.sqrt((N2 - n2) / (N2 - 1)) if N2 > 1 else 1
        # Calculate standard error with FPC
        se = math.sqrt((p1 * (1 - p1) / n1) * FPC1**2 + (p2 * (1 - p2) / n2) * FPC2**2)
    else:
        # Standard standard error without FPC
        se = math.sqrt(p1*(1 - p1)/n1 + p2*(1 - p2)/n2)
    
    if se == 0:
        return 0
    z = (p1 - p2) / se
    return z

def calculate_p_value(z, direction='greater'):
    """
    Calculates the p-value for the one-sided z-test.
    """
    if direction == 'greater':
        p_val = 1 - norm.cdf(z)
    else:
        p_val = norm.cdf(z)
    return p_val

def calculate_sample_size_finite(alpha, E, p, N):
    """
    Calculates required sample size with Finite Population Correction (FPC).
    """
    z = norm.ppf(1 - alpha)
    n0 = (z**2 * p * (1 - p)) / (E**2)
    n = n0 / (1 + (n0 - 1)/N)
    return math.ceil(n)

def plot_OC_curve(p1, n1, p2_range, alpha, direction, n2):
    """
    Plots the Operating Characteristic (OC) curve showing power vs. p2.
    """
    power = []
    z_alpha = norm.ppf(1 - alpha)
    
    for p2 in p2_range:
        se = math.sqrt(p1*(1 - p1)/n1 + p2*(1 - p2)/n2)
        if se == 0:
            z = 0
        else:
            z = (p1 - p2) / se
        if direction == 'greater':
            # Power = P(z >= z_alpha | H1 true)
            power_val = 1 - norm.cdf(z_alpha - z)
        else:
            # Power = P(z <= -z_alpha | H1 true)
            power_val = norm.cdf(z_alpha + z)
        power.append(power_val)
    
    fig, ax = plt.subplots()
    ax.plot(p2_range, power, color='blue', label='Power')
    ax.axhline(y=1 - alpha, color='red', linestyle='--', label=f'Power = 1 - α ({1 - alpha})')
    ax.set_xlabel('True Proportion p₂')
    ax.set_ylabel('Power')
    ax.set_title('Operating Characteristic (OC) Curve')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

def main():
    st.title("One-Sided Difference in Proportion Test with OC Curve")
    
    st.markdown(
        r"""
        ### Perform a One-Sided Difference in Proportion Test

        This app allows you to test whether one population proportion is greater than or less than another. Additionally, it visualizes the **Operating Characteristic (OC) curve**, illustrating the test's power across a range of true proportions.

        ---
        """,
        unsafe_allow_html=True
    )
    
    st.sidebar.header("Parameters")
    
    # Test Direction
    direction = st.sidebar.radio(
        "Select Test Direction:",
        ("p₁ > p₂", "p₁ < p₂"),
        help="Choose 'p₁ > p₂' to test if proportion 1 is greater than proportion 2, or 'p₁ < p₂' to test if proportion 1 is less than proportion 2."
    )
    
    # Map direction to 'greater' or 'less'
    if direction == "p₁ > p₂":
        test_direction = 'greater'
    else:
        test_direction = 'less'
    
    # Proportion Inputs
    p1 = st.sidebar.number_input(
        "Proportion p₁",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.01,
        help="Proportion in the first population (between 0 and 1)."
    )
    p2 = st.sidebar.number_input(
        "Proportion p₂",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        help="Proportion in the second population (between 0 and 1)."
    )
    
    # Sample Size Inputs
    n1 = st.sidebar.number_input(
        "Sample Size n₁",
        min_value=1,
        max_value=1000000,
        value=100,
        step=1,
        help="Sample size for the first population."
    )
    n2 = st.sidebar.number_input(
        "Sample Size n₂",
        min_value=1,
        max_value=1000000,
        value=100,
        step=1,
        help="Sample size for the second population."
    )
    
    # Significance Level
    alpha = st.sidebar.number_input(
        "Significance Level (α)",
        min_value=0.0,
        max_value=1.0,
        value=0.05,
        step=0.01,
        help="Significance level for the test (e.g., 0.05 for 95% confidence)."
    )
    
    # Finite Population Option
    finite_population = st.sidebar.radio(
        "Do you want to specify finite population sizes?",
        ("No", "Yes"),
        help="Select 'Yes' to input finite population sizes for both groups, applying Finite Population Correction."
    )
    
    N1 = None
    N2 = None
    if finite_population == "Yes":
        N1 = st.sidebar.number_input(
            "Population Size N₁",
            min_value=1,
            max_value=10000000,
            value=10000,
            step=1,
            help="Total population size for the first group."
        )
        N2 = st.sidebar.number_input(
            "Population Size N₂",
            min_value=1,
            max_value=10000000,
            value=10000,
            step=1,
            help="Total population size for the second group."
        )
        if N1 < n1:
            st.sidebar.warning("Population size N₁ must be at least equal to sample size n₁.")
        if N2 < n2:
            st.sidebar.warning("Population size N₂ must be at least equal to sample size n₂.")
    
    # Calculate Test Statistic and P-Value
    if finite_population == "Yes" and N1 and N2:
        finite = True
    else:
        finite = False
    
    z = calculate_test_statistic(p1, p2, n1, n2, finite, N1, N2)
    p_val = calculate_p_value(z, direction=test_direction)
    
    # Display Results
    st.subheader("Test Results")
    st.latex(r"""
    z = \frac{p_1 - p_2}{\sqrt{\frac{p_1(1 - p_1)}{n_1} + \frac{p_2(1 - p_2)}{n_2}}}
    """)
    st.markdown(f"- **Test Statistic (z):** {z:.4f}")
    st.markdown(f"- **P-Value:** {p_val:.4f}")
    
    # Decision with English and Arabic Translations
    if test_direction == 'greater':
        if p_val < alpha:
            decision_en = "Reject the null hypothesis. There is sufficient evidence that \( p_1 \) is greater than \( p_2 \)."
            decision_ar = "رفض الفرضية الصفرية. هناك أدلة كافية على أن \( p_1 \) أكبر من \( p_2 \)."
        else:
            decision_en = "Fail to reject the null hypothesis. There is insufficient evidence that \( p_1 \) is greater than \( p_2 \)."
            decision_ar = "عدم رفض الفرضية الصفرية. لا توجد أدلة كافية على أن \( p_1 \) أكبر من \( p_2 \)."
    else:
        if p_val < alpha:
            decision_en = "Reject the null hypothesis. There is sufficient evidence that \( p_1 \) is less than \( p_2 \)."
            decision_ar = "رفض الفرضية الصفرية. هناك أدلة كافية على أن \( p_1 \) أقل من \( p_2 \)."
        else:
            decision_en = "Fail to reject the null hypothesis. There is insufficient evidence that \( p_1 \) is less than \( p_2 \)."
            decision_ar = "عدم رفض الفرضية الصفرية. لا توجد أدلة كافية على أن \( p_1 \) أقل من \( p_2 \)."
    
    # Display Decision with RTL for Arabic
    st.markdown(f"### **Decision:** {decision_en}")
    st.markdown(
        f"""### **القرار:** <div dir="rtl">{decision_ar}</div>""",
        unsafe_allow_html=True
    )
    
    # Expandable Section for Formula Details
    with st.expander("Show Formula Details"):
        if finite_population == "Yes" and N1 and N2:
            st.markdown(
                r"""
                **Formula Used (Finite Population):**
        
                $$
                z = \frac{p_1 - p_2}{\sqrt{\frac{p_1(1 - p_1)}{n_1} \times \frac{N_1 - n_1}{N_1 - 1} + \frac{p_2(1 - p_2)}{n_2} \times \frac{N_2 - n_2}{N_2 - 1}}}
                $$
        
                **Where:**
                - $ p_1 $ = Proportion in the first population
                - $ p_2 $ = Proportion in the second population
                - $ n_1 $ = Sample size of the first population
                - $ n_2 $ = Sample size of the second population
                - $ N_1 $ = Population size of the first group
                - $ N_2 $ = Population size of the second group
        
                **Adjustment with Finite Population Correction (FPC):**
        
                When sampling from finite populations, the standard error is adjusted using FPC:
        
                $$
                SE = \sqrt{\frac{p_1(1 - p_1)}{n_1} \times \frac{N_1 - n_1}{N_1 - 1} + \frac{p_2(1 - p_2)}{n_2} \times \frac{N_2 - n_2}{N_2 - 1}}
                $$
        
                **Calculation Steps:**
                1. **Compute the Finite Population Correction (FPC) for each population:**
                   - For Population 1:
                     $$
                     FPC_1 = \sqrt{\frac{N_1 - n_1}{N_1 - 1}}
                     $$
                   - For Population 2:
                     $$
                     FPC_2 = \sqrt{\frac{N_2 - n_2}{N_2 - 1}}
                     $$
                2. **Calculate the Standard Error (SE) with FPC:**
                   $$
                   SE = \sqrt{\left(\frac{p_1(1 - p_1)}{n_1} \times FPC_1^2\right) + \left(\frac{p_2(1 - p_2)}{n_2} \times FPC_2^2\right)}
                   $$
                3. **Compute the z-test statistic:**
                   $$
                   z = \frac{p_1 - p_2}{SE}
                   $$
                4. **Determine the p-value based on the test direction.**
                5. **Compare the p-value with the significance level \( \alpha \) to make a decision.**
                """
            )
        else:
            st.markdown(
                r"""
                **Formula Used (Infinite Population):**
        
                $$
                z = \frac{p_1 - p_2}{\sqrt{\frac{p_1(1 - p_1)}{n_1} + \frac{p_2(1 - p_2)}{n_2}}}
                $$
        
                **Where:**
                - $ p_1 $ = Proportion in the first population
                - $ p_2 $ = Proportion in the second population
                - $ n_1 $ = Sample size of the first population
                - $ n_2 $ = Sample size of the second population
        
                **Calculation Steps:**
                1. **Calculate the Standard Error (SE):**
                   $$
                   SE = \sqrt{\frac{p_1(1 - p_1)}{n_1} + \frac{p_2(1 - p_2)}{n_2}}
                   $$
                2. **Compute the z-test statistic:**
                   $$
                   z = \frac{p_1 - p_2}{SE}
                   $$
                3. **Determine the p-value based on the test direction.**
                4. **Compare the p-value with the significance level \( \alpha \) to make a decision.**
                """
            )
    
    # Expandable Section for Operating Characteristic (OC) Curve
    with st.expander("Show Operating Characteristic (OC) Curve"):
        st.markdown(
            r"""
            ### Operating Characteristic (OC) Curve
            
            The OC curve illustrates the power of the test across a range of true proportions \( p_2 \). It shows the probability of correctly rejecting the null hypothesis for different values of \( p_2 \).
            
            **Interpretation:**
            - **Power:** The ability of the test to correctly reject the null hypothesis when it is false.
            - **Higher Power:** Indicates a higher probability of detecting a true effect.
            - **Curve Shape:** As \( p_2 \) decreases (for 'greater' test), power increases, and vice versa.
            
            ---
            """
        )
        st.markdown(
            r"""
            ### منحنى الخصائص التشغيلية (OC)
            
            يوضح منحنى OC قوة الاختبار عبر مجموعة من النسب الحقيقية \( p_2 \). يظهر احتمال رفض الفرضية الصفرية بشكل صحيح لمختلف قيم \( p_2 \).
            
            **التفسير:**
            - **القوة:** قدرة الاختبار على رفض الفرضية الصفرية بشكل صحيح عندما تكون خاطئة.
            - **قوة أعلى:** تشير إلى احتمال أكبر لاكتشاف تأثير حقيقي.
            - **شكل المنحنى:** كلما انخفض \( p_2 \) (في اختبار 'أكبر'), زادت القوة، والعكس صحيح.
            
            ---
            """
        )
        
        # Define range for p2 based on test direction
        if test_direction == 'greater':
            p2_min = 0.0
            p2_max = p1
            p2_range = np.linspace(p2_min, p2_max, 100)
        else:
            p2_min = p1
            p2_max = 1.0
            p2_range = np.linspace(p2_min, p2_max, 100)
        
        plot_OC_curve(p1, n1, p2_range, alpha, test_direction, n2)
    
    # Additional Notes
    st.markdown(
        r"""
        ---
        **Note:** If you specified finite population sizes, the calculations adjust the sample sizes accordingly using the Finite Population Correction (FPC). Ensure that your sample sizes do not exceed your population sizes.
        """
    )

if __name__ == "__main__":
    main()
