import math
import streamlit as st
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

def calculate_sample_size_infinite(alpha: float, E: float, p: float) -> int:
    """
    Returns the sample size needed (rounded up) for a desired margin of error (E)
    in a confidence interval for a proportion p, given alpha.

    Formula: n >= (z_{alpha/2} / E)^2 * p(1 - p)
    """
    z_value = norm.ppf(1 - alpha/2)
    n_float = (z_value / E) ** 2 * p * (1 - p)
    return math.ceil(n_float)

def calculate_sample_size_finite(alpha: float, E: float, p: float, N: int) -> int:
    """
    Calculates the required sample size for a finite population using Finite Population Correction (FPC).

    Parameters:
    - alpha: Significance level (e.g., 0.05 for 95% confidence)
    - E: Margin of error (e.g., 0.05 for 5%)
    - p: Estimated proportion (e.g., 0.5 for maximum variability)
    - N: Total population size

    Returns:
    - Required sample size (rounded up)
    """
    z = norm.ppf(1 - alpha/2)
    n0 = (z**2 * p * (1 - p)) / (E**2)
    n = n0 / (1 + (n0 - 1) / N)
    return math.ceil(n)

def plot_sample_size_vs_E(alpha: float, p: float, finite: bool, N: int = None):
    """
    Plots sample size vs Margin of Error (E) for a given alpha and proportion p.
    If finite is True and N is provided, applies Finite Population Correction.
    """
    E_values = np.linspace(0.01, 0.5, 100)
    sample_sizes = []
    
    for E in E_values:
        if finite and N:
            n = calculate_sample_size_finite(alpha, E, p, N)
        else:
            n = calculate_sample_size_infinite(alpha, E, p)
        sample_sizes.append(n)
    
    fig, ax = plt.subplots()
    ax.plot(E_values, sample_sizes, color='green')
    ax.set_xlabel('Margin of Error (E)')
    ax.set_ylabel('Sample Size (n)')
    ax.set_title('Sample Size vs. Margin of Error (E)')
    ax.grid(True)
    st.pyplot(fig)

def main():
    st.title("Sample Size Calculator (Method 1)")

    st.markdown(
        r"""
        ### Determine Sample Size for a Desired Confidence Interval Width

        We use the formula:

        $$
        n \geq \left(\frac{z_{\alpha/2}}{E}\right)^2 \, p(1 - p)
        $$

        - **$ \alpha $**: Significance level (e.g., 0.05 for 95% CI)  
        - **$ E $**: Margin of error (half-width of the CI)  
        - **$ p $**: Estimated (or worst-case) population proportion  
        ---
        """,
        unsafe_allow_html=True
    )

    st.sidebar.header("Parameters")
    
    # Option to specify population size
    finite_population = st.sidebar.radio(
        "Do you want to specify the population size (N)?",
        ("No", "Yes"),
        help="Select 'Yes' if you have a finite population size to adjust the sample size calculation accordingly."
    )
    
    # User Inputs with Tooltips
    alpha = st.sidebar.number_input(
        "Alpha (α)", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.05, 
        step=0.01, 
        help="Significance level (Type I error rate). Common values are 0.05 for a 95% confidence interval."
    )
    E = st.sidebar.number_input(
        "Margin of Error (E)", 
        min_value=0.001, 
        max_value=1.0, 
        value=0.05, 
        step=0.001, 
        help="Margin of error (half-width of the confidence interval). Smaller values require larger sample sizes."
    )
    p_val = st.sidebar.number_input(
        "Proportion (p)", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.5, 
        step=0.01, 
        help="Estimated population proportion. Use 0.5 for maximum variability if unsure."
    )
    
    N = None
    if finite_population == "Yes":
        N = st.sidebar.number_input(
            "Population Size (N)", 
            min_value=1, 
            max_value=1_000_000, 
            value=10000, 
            step=1, 
            help="Total size of the population from which the sample is drawn."
        )
        if N < 1:
            st.error("Population size (N) must be at least 1.")
    
    # Calculate required sample size
    if finite_population == "Yes" and N:
        n_required = calculate_sample_size_finite(alpha, E, p_val, N)
    else:
        n_required = calculate_sample_size_infinite(alpha, E, p_val)
    
    z_value = norm.ppf(1 - alpha/2)
    
    st.subheader("Results:")
    st.latex(r"""
    n \geq \left(\frac{z_{\alpha/2}}{E}\right)^2 \, p(1 - p)
    """)
    
    st.markdown(f"- $z_{{\\alpha/2}}$ = {z_value:.4f}")
    st.markdown(f"- $\\alpha$ = {alpha}")
    st.markdown(f"- $E$ = {E}")
    st.markdown(f"- $p$ = {p_val}")
    
    if finite_population == "Yes" and N:
        st.markdown(f"- $N$ = {N}")
    
    st.markdown(f"### **Required Sample Size:** {n_required}")
    
    if finite_population == "Yes" and N:
        if n_required > N:
            st.warning(f"Calculated sample size ({n_required}) exceeds the population size ({N}). Please adjust your inputs.")
        elif n_required == N:
            st.info(f"Sample size equals the population size. Consider conducting a census.")
    
    # Expandable Section for Formula Details
    with st.expander("Show Formula Details"):
        if finite_population == "Yes" and N:
            st.markdown(
                rf"""
                **Formula Used (Finite Population):**
        
                $$
                n \geq \frac{{\left(\frac{{z_{{\alpha/2}}}}{{E}}\right)^2 \times p(1 - p)}}{{1 + \left(\frac{{\left(\frac{{z_{{\alpha/2}}}}{{E}}\right)^2 \times p(1 - p) - 1}}{{N}}\right)}}
                $$
        
                **Where:**
                - $ z_{{\alpha/2}} $ = {z_value:.4f}
                - $ \alpha $ = {alpha}
                - $ E $ = {E}
                - $ p $ = {p_val}
                - $ N $ = {N}
        
                **Calculation Steps:**
                1. Compute the initial sample size ($ n_0 $) assuming an infinite population:
                   $$
                   n_0 = \frac{{z_{{\alpha/2}}^2 \times p(1 - p)}}{{E^2}}
                   $$
                2. Adjust for finite population:
                   $$
                   n = \frac{{n_0}}{{1 + \left(\frac{{n_0 - 1}}{{N}}\right)}}
                   $$
                3. Round up $ n $ to the nearest whole number as sample size must be an integer.
                """
            )
        else:
            st.markdown(
                rf"""
                **Formula Used (Infinite Population):**
        
                $$
                n \geq \left(\frac{{z_{{\alpha/2}}}}{{E}}\right)^2 \times p(1 - p)
                $$
        
                **Where:**
                - $ z_{{\alpha/2}} $ = {z_value:.4f}
                - $ \alpha $ = {alpha}
                - $ E $ = {E}
                - $ p $ = {p_val}
        
                **Calculation Steps:**
                1. Compute the sample size using the formula.
                2. Round up $ n $ to the nearest whole number as sample size must be an integer.
                """
            )
    
    # Expandable Section for Visualizations
    with st.expander("Show Visualizations"):
        st.markdown("#### Sample Size vs. Margin of Error (E)")
        plot_sample_size_vs_E(alpha, p_val, finite_population == "Yes", N)
        
        if finite_population == "Yes" and N:
            st.markdown("#### Sample Size vs. Population Size (N)")
            # Define a range for N if needed, or plot how sample size changes as N varies
            # For simplicity, we'll plot up to 10 times the initial N
            plot_sample_size_vs_E(alpha, p_val, finite=True, N=N)

if __name__ == "__main__":
    main()
