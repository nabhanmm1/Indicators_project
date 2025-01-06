import math
import streamlit as st
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

def calculate_sample_size_finite(alpha: float, E: float, p: float, N: int) -> int:
    """
    Calculates the required sample size for a finite population.
    
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

def plot_sample_size_vs_N(alpha: float, E: float, p: float, max_N: int):
    """
    Plots sample size vs Population size for given alpha, E, and p.
    """
    N_values = np.linspace(100, max_N, 100)
    sample_sizes = [calculate_sample_size_finite(alpha, E, p, int(N)) for N in N_values]
    
    fig, ax = plt.subplots()
    ax.plot(N_values, sample_sizes, color='blue')
    ax.set_xlabel('Population Size (N)')
    ax.set_ylabel('Sample Size (n)')
    ax.set_title('Sample Size vs. Population Size (N)')
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
        - **$ N $**: Total population size
    
        ---
        """,
        unsafe_allow_html=True
    )

    st.sidebar.header("Parameters")
    
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
    N = st.sidebar.number_input(
        "Population Size (N)", 
        min_value=1, 
        max_value=1_000_000, 
        value=10_000, 
        step=1, 
        help="Total size of the population from which the sample is drawn."
    )

    n_required = calculate_sample_size_finite(alpha, E, p_val, N)
    z_value = norm.ppf(1 - alpha/2)

    st.subheader("Results:")
    st.latex(r"""
    n \geq \left(\frac{z_{\alpha/2}}{E}\right)^2 \, p(1 - p)
    """)
    
    st.markdown(f"- $z_{{\\alpha/2}}$ = {z_value:.4f}")
    st.markdown(f"- $\\alpha$ = {alpha}")
    st.markdown(f"- $E$ = {E}")
    st.markdown(f"- $p$ = {p_val}")
    st.markdown(f"- $N$ = {N}")

    st.markdown(f"### **Required Sample Size:** {n_required}")

    if n_required == float('inf'):
        st.warning("Cannot detect a zero difference ($p = 0$ or $p = 1$). Please adjust your inputs.")
    
    # Expandable Section for Formula Details
    with st.expander("Show Formula Details"):
        st.markdown(
            rf"""
            **Formula Used:**
    
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
    
    # Expandable Section for Visualizations
    with st.expander("Show Visualizations"):
        st.markdown("#### Sample Size vs. Margin of Error (E)")
        plot_sample_size_vs_E(alpha, p_val)
        
        st.markdown("#### Sample Size vs. Population Size (N)")
        plot_sample_size_vs_N(alpha, E, p_val, N)

if __name__ == "__main__":
    main()
