import math
import streamlit as st
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

def calculate_basic_sample_size(alpha, beta, p1, p2, two_sided=True):
    """
    Basic formula to calculate sample size per group for two proportions.
    """
    if two_sided:
        z_alpha = norm.ppf(1 - alpha/2)  # z_{alpha/2}
    else:
        z_alpha = norm.ppf(1 - alpha)    # z_{alpha}
    
    z_beta = norm.ppf(1 - beta)         # z_{beta}
    
    delta = abs(p1 - p2)
    p_bar = (p1 + p2) / 2.0
    
    if delta == 0:
        return float('inf')  # Can't detect a zero difference
    
    numerator = (z_alpha + z_beta)**2 * (p_bar * (1 - p_bar))
    denominator = (delta**2) / 2.0  # Adjusted for equal sample sizes
    
    n = 2.0 * (numerator / denominator)
    return math.ceil(n)

def calculate_advanced_sample_size(alpha, beta, p1, p2, two_sided=True):
    """
    Advanced formula to calculate sample size per group for two proportions.
    """
    if two_sided:
        z_alpha = norm.ppf(1 - alpha/2)  # z_{alpha/2}
    else:
        z_alpha = norm.ppf(1 - alpha)    # z_{alpha}
    
    z_beta = norm.ppf(1 - beta)         # z_{beta}
    
    q1 = 1 - p1
    q2 = 1 - p2
    
    delta = p1 - p2
    if delta == 0:
        return float('inf')  # Undefined if p1 == p2
    
    term1 = z_alpha * math.sqrt((p1 + p2) * (q1 + q2) / 2)
    term2 = z_beta * math.sqrt(p1 * q1 + p2 * q2)
    
    numerator = (term1 + term2) ** 2
    denominator = (delta) ** 2
    
    n = numerator / denominator
    return math.ceil(n)

def plot_sample_size_vs_delta(alpha, beta, p1, p2, two_sided, method):
    """
    Plots sample size vs difference in proportions.
    """
    # Define a range of delta values around the current difference
    current_delta = abs(p1 - p2)
    max_delta = min(p1, 1 - p2) if p1 > p2 else min(p2, 1 - p1)
    delta_range = np.linspace(0.01, max_delta, 100)
    sample_sizes = []
    
    for delta in delta_range:
        # Adjust p2 based on delta to ensure it stays within [0,1]
        if p1 > p2:
            p2_adjusted = p1 - delta
            if p2_adjusted < 0:
                p2_adjusted = 0.0
        else:
            p2_adjusted = p1 + delta
            if p2_adjusted > 1:
                p2_adjusted = 1.0
        
        # Calculate sample size based on selected method
        if method == "Basic Method":
            p_bar = (p1 + p2_adjusted) / 2.0
            if delta == 0:
                n = float('inf')
            else:
                numerator = (norm.ppf(1 - alpha/2) + norm.ppf(1 - beta))**2 * (p_bar * (1 - p_bar))
                denominator = (delta**2) / 2.0
                n = 2.0 * (numerator / denominator)
                n = math.ceil(n)
        else:
            if delta == 0:
                n = float('inf')
            else:
                z_alpha = norm.ppf(1 - alpha/2) if two_sided else norm.ppf(1 - alpha)
                z_beta = norm.ppf(1 - beta)
                q1 = 1 - p1
                q2 = 1 - p2_adjusted
                term1 = z_alpha * math.sqrt((p1 + p2_adjusted) * (q1 + q2) / 2)
                term2 = z_beta * math.sqrt(p1 * q1 + p2_adjusted * q2)
                numerator = (term1 + term2) ** 2
                denominator = (p1 - p2_adjusted) ** 2
                n = numerator / denominator
                n = math.ceil(n)
        
        sample_sizes.append(n)
    
    # Plotting
    fig, ax = plt.subplots()
    ax.plot(delta_range, sample_sizes, label=method, color='blue')
    ax.set_xlabel('Difference in Proportions (δ)')
    ax.set_ylabel('Sample Size per Group (n)')
    ax.set_title('Sample Size vs. Difference in Proportions')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

def main():
    st.title("Two Proportion Sample Size")
    
    st.markdown(
        """
        This tool calculates the **required sample size** per group to detect a difference
        between two population proportions $p_1$ and $p_2$ with specified:
        
        - Significance level $ \alpha $ (Type I error)
        - Desired Type II error $ \beta $ (or power = 1 - $ \beta $)
        - Whether the test is two-sided or one-sided
        
        Use the sidebar to select the calculation method and input the required parameters.
        """
    )
    
    st.sidebar.header("Parameters")
    
    # Method Selection
    method = st.sidebar.radio(
        "Select Calculation Method:", 
        ["Basic Method", "Advanced Method"],
        index=0,
        help="Choose 'Basic Method' for a basic calculation or 'Advanced Method' for a more precise calculation accounting for different variances."
    )
    
    # User Inputs with Tooltips
    alpha = st.sidebar.number_input(
        "Alpha (α Type I error)", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.05, 
        step=0.01, 
        help="Significance level (Type I error rate). Common values are 0.05 or 0.01."
    )
    beta  = st.sidebar.number_input(
        "Beta (β Type II error)", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.2, 
        step=0.01, 
        help="Type II error rate. Common value is 0.2 (power = 0.8)."
    )
    p1    = st.sidebar.number_input(
        "Proportion p1", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.3, 
        step=0.01, 
        help="Proportion in the first population (between 0 and 1)."
    )
    p2    = st.sidebar.number_input(
        "Proportion p2", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.5, 
        step=0.01, 
        help="Proportion in the second population (between 0 and 1)."
    )
    
    test_side = st.sidebar.radio(
        "Test Type", 
        ["Two-sided", "One-sided"], 
        index=0, 
        help="Choose 'Two-sided' if you want to detect a difference in either direction, or 'One-sided' if only one direction is of interest."
    )
    two_sided = True if test_side == "Two-sided" else False
    
    # Input Validation
    error = False
    if not (0 <= p1 <= 1) or not (0 <= p2 <= 1):
        st.error("Proportions p1 and p2 must be between 0 and 1.")
        error = True
    if not (0 < alpha < 1):
        st.error("Alpha (α) must be between 0 and 1.")
        error = True
    if not (0 < beta < 1):
        st.error("Beta must be between 0 and 1.")
        error = True
    
    if not error:
        # Calculate Sample Size based on selected method
        if method == "Basic Method":
            n_required = calculate_basic_sample_size(alpha, beta, p1, p2, two_sided)
            formula_used = "Basic Method"
        else:
            n_required = calculate_advanced_sample_size(alpha, beta, p1, p2, two_sided)
            formula_used = "Advanced Method"
        
        # Display Calculated Sample Size
        st.subheader("Calculated Sample Size (per group)")
        st.latex(r"n \geq " + str(n_required))
        
        # Display Parameter Summary
        st.markdown(f"""
        - **Method Used**: {formula_used}
        - $\alpha$ = {alpha}
        - $\beta$ = {beta}
        - $p_1$ = {p1}
        - $p_2$ = {p2}
        - **Test Type** = {test_side}
        """)
        
        if n_required == float('inf'):
            st.warning("Cannot detect a zero difference (p1 == p2).")
        
        # Method-specific Information
        if method == "Basic Method":
            st.markdown(
                """
                ### Basic Method Formula
                $$
                n \geq 2 \times \frac{(z_{\alpha/2} + z_{\beta})^2 \times \bar{p} \times (1 - \bar{p})}{\delta^2}
                $$
                where:
                - $\bar{p} = \frac{p_1 + p_2}{2}$
                - $\delta = |p_1 - p_2|$
                """
            )
        else:
            st.markdown(
                """
                ### Advanced Method Formula
                $$
                n = \frac{\left[z_{\alpha / 2} \times \sqrt{\frac{(p_1 + p_2)(q_1 + q_2)}{2}} + z_\beta \times \sqrt{p_1 q_1 + p_2 q_2}\right]^2}{(p_1 - p_2)^2}
                $$
                where:
                - $q_1 = 1 - p_1$
                - $q_2 = 1 - p_2$
                """
            )
        
        # Expandable Section for Formula Details
        with st.expander("Show Formula Details"):
            if method == "Basic Method":
                p_bar = (p1 + p2) / 2.0
                delta = abs(p1 - p2)
                z_alpha = norm.ppf(1 - alpha/2) if two_sided else norm.ppf(1 - alpha)
                z_beta = norm.ppf(1 - beta)
                st.markdown(
                    f"""
                    **Formula Used: Basic Method**
                    $$
                    n \geq 2 \times \frac{{({z_alpha:.4f} + {z_beta:.4f})^2 \times {p_bar:.4f} \times (1 - {p_bar:.4f})}}{{({delta:.4f})^2}}
                    $$
                    where:
                    - $z_{{\alpha/2}}$ = {z_alpha:.4f}
                    - $z_{{\beta}}$ = {z_beta:.4f}
                    - $\bar{{p}} = \frac{{{p1} + {p2}}}{2} = {p_bar:.4f}$
                    - $\delta$ = |{p1} - {p2}| = {delta:.4f}
                    """
                )
            else:
                delta = p1 - p2
                z_alpha = norm.ppf(1 - alpha/2) if two_sided else norm.ppf(1 - alpha)
                z_beta = norm.ppf(1 - beta)
                q1 = 1 - p1
                q2 = 1 - p2
                st.markdown(
                    f"""
                    **Formula Used: Advanced Method**
                    $$
                    n = \frac{{\left[{z_alpha:.4f} \times \sqrt{{\frac{{({p1} + {p2}) \times ({q1} + {q2})}}{2}}} + {z_beta:.4f} \times \sqrt{{{p1} \times {q1} + {p2} \times {q2}}}\right]^2}}{{({delta:.4f})^2}}
                    $$
                    where:
                    - $z_{{\alpha/2}}$ = {z_alpha:.4f}
                    - $z_{{\beta}}$ = {z_beta:.4f}
                    - $q_1 = 1 - p_1 = {q1:.4f} $
                    - $q_2 = 1 - p_2 = {q2:.4f} $
                    - $\delta = p_1 - p_2 = {delta:.4f} $
                    """
                )
        
        # Expandable Section for Visual Aid: Plot Sample Size vs Difference in Proportions
        with st.expander("Show Sample Size vs. Difference Plot"):
            plot_sample_size_vs_delta(alpha, beta, p1, p2, two_sided, method)

if __name__ == "__main__":
    main()
