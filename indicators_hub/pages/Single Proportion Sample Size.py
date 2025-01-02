import streamlit as st
import math
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

def calculate_sample_size(alpha: float, E: float, p: float) -> int:
    """
    Returns the sample size needed (rounded up) for a desired margin of error (E)
    in a confidence interval for a proportion p, given alpha.

    Formula: n >= (z_{alpha/2} / E)^2 * p(1 - p)
    """
    z_value = norm.ppf(1 - alpha/2)
    n_float = (z_value / E) ** 2 * p * (1 - p)
    return math.ceil(n_float)

def plot_sample_size_vs_E(alpha: float, p: float):
    """
    Plots sample size vs Margin of Error (E) for a fixed alpha and proportion p.
    """
    E_values = np.linspace(0.01, 0.5, 100)
    sample_sizes = [calculate_sample_size(alpha, E, p) for E in E_values]
    
    fig, ax = plt.subplots()
    ax.plot(E_values, sample_sizes, color='green')
    ax.set_xlabel('Margin of Error (E)')
    ax.set_ylabel('Sample Size (n)')
    ax.set_title('Sample Size vs. Margin of Error (E)')
    ax.grid(True)
    st.pyplot(fig)

def plot_sample_size_vs_p(alpha: float, E: float):
    """
    Plots sample size vs Proportion (p) for a fixed alpha and margin of error E.
    """
    p_values = np.linspace(0.01, 0.99, 100)
    sample_sizes = [calculate_sample_size(alpha, E, p) for p in p_values]
    
    fig, ax = plt.subplots()
    ax.plot(p_values, sample_sizes, color='purple')
    ax.set_xlabel('Proportion (p)')
    ax.set_ylabel('Sample Size (n)')
    ax.set_title('Sample Size vs. Proportion (p)')
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

    n_required = calculate_sample_size(alpha, E, p_val)
    z_value = norm.ppf(1 - alpha/2)

    st.subheader("Results:")
    st.latex(r"""
    n \geq \left(\frac{z_{\alpha/2}}{E}\right)^2 \, p(1 - p)
    """)

    st.markdown(f"- $z_{{\\alpha/2}}$ = {z_value:.4f}")
    st.markdown(f"- $\\alpha$ = {alpha}")
    st.markdown(f"- $E$ = {E}")
    st.markdown(f"- $p$ = {p_val}")

    st.markdown(f"### **Required Sample Size:** {n_required}")

    if n_required == float('inf'):
        st.warning("Cannot detect a zero difference ($p = 0$ or $p = 1$). Please adjust your inputs.")
    
    # Expandable Section for Formula Details
    with st.expander("Show Formula Details"):
        st.markdown(
            rf"""
            **Formula Used:**

            $$
            n \geq \left(\frac{{z_{{\alpha/2}}}}{{E}}\right)^2 \times p(1 - p)
            $$

            **Where:**
            - $ z_{{\alpha/2}} $ = {z_value:.4f}
            - $ \alpha $ = {alpha}
            - $ E $ = {E}
            - $ p $ = {p_val}

            **Calculation Steps:**
            1. Compute $ z_{{\alpha/2}} $ using the standard normal distribution.
            2. Plug the values into the formula to calculate $ n $.
            3. Round up $ n $ to the nearest whole number as sample size must be an integer.
            """
        )
    
    # Expandable Section for Visualizations
    with st.expander("Show Visualizations"):
        st.markdown("#### Sample Size vs. Margin of Error (E)")
        plot_sample_size_vs_E(alpha, p_val)
        
        st.markdown("#### Sample Size vs. Proportion (p)")
        plot_sample_size_vs_p(alpha, E)

if __name__ == "__main__":
    main()
