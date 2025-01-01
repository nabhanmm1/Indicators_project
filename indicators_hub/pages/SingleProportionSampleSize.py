import streamlit as st
import math
from scipy.stats import norm

def calculate_sample_size(alpha: float, E: float, p: float) -> int:
    """
    Returns the sample size needed (rounded up) for a desired margin of error (E)
    in a confidence interval for a proportion p, given alpha.

    Formula: n >= (z_{alpha/2} / E)^2 * p(1 - p)
    """
    z_value = norm.ppf(1 - alpha/2)
    n_float = (z_value / E) ** 2 * p * (1 - p)
    return math.ceil(n_float)

def main():
    st.title("Sample Size Calculator (Method 1)")

    st.markdown(
        r"""
        ### Determine Sample Size for a Desired CI Width

        We use the formula:

        $$
        n \ge \left(\frac{z_{\alpha/2}}{E}\right)^2 \, p(1 - p)
        $$

        - $ \alpha $: Significance level (e.g., 0.05 for 95% CI)  
        - $ E $: Margin of error (half-width of the CI)  
        - $ p $: Estimated (or worst-case) population proportion  

        ---
        """,
        unsafe_allow_html=True
    )

    st.sidebar.header("Parameters")
    alpha = st.sidebar.number_input("Alpha (α)", 0.0, 1.0, 0.05, 0.01)
    E     = st.sidebar.number_input("Margin of Error (E)", 0.0, 1.0, 0.1789, 0.01)
    p_val = st.sidebar.number_input("Proportion (p)", 0.0, 1.0, 0.5, 0.01)

    n_required = calculate_sample_size(alpha, E, p_val)
    z_value = norm.ppf(1 - alpha/2)

    st.subheader("Results:")
    st.latex(r"""
    n \;\ge\; \left(\frac{z_{\alpha/2}}{E}\right)^2 \; p\,(1 - p)
    """)

    st.markdown(f"- $z_{{\\alpha/2}} = {z_value:.4f}$")
    st.markdown(f"- $\\alpha = {alpha}$")
    st.markdown(f"- $E = {E}$")
    st.markdown(f"- $p = {p_val}$")

    st.markdown(f"### **Required Sample Size:** {n_required}")

if __name__ == "__main__":
    main()
