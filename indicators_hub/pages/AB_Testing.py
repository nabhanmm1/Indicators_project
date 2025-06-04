import streamlit as st
import numpy as np
from scipy.stats import norm, ttest_ind, t as t_dist
import math
import pandas as pd
from statsmodels.stats.proportion import proportions_ztest, confint_proportions_2indep
from scipy.stats import beta as beta_dist, gaussian_kde
import matplotlib.pyplot as plt
from itertools import product # For creating segment combinations

# --- Page Configuration ---
st.set_page_config(
    page_title="A/B Testing Guide & Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Placeholder for Future Content (Backburner List) ---
FUTURE_FEATURES = {
    "Advanced Factor Modeling (in 'Analyze Results')": "Implement logistic regression/ANOVA for analyzing simultaneous factor impacts and interactions.",
    "Estimated Test Duration (in 'Sample Size Calculator')": "Calculate how long a test might need to run based on traffic and desired power/MDE.",
    "Relative MDE Input (in 'Sample Size Calculator')": "Provide relative MDE (e.g., 10% lift) as an alternative input for determining sample size.",
    "Enhanced Segmentation Display (e.g., Cross-Tabulation)": "Automatically create and analyze segments for combinations of multiple selected factors.",
    "Arabic Language Support": "Add Arabic localization to the application.",
    "Expected Loss/Uplift (Advanced Bayesian Metrics)": "Include more detailed Bayesian decision theory metrics like Expected Loss if choosing a suboptimal variation.",
    "Support for Continuous Outcomes (Advanced Bayesian Models)": "Explore more complex Bayesian models (e.g., hierarchical models, MCMC via libraries like PyMC or Stan) for continuous data if simpler t-distribution approximations are insufficient or assumptions are violated.",
    "Support for Ratio Metrics": "Enable analysis for metrics that are ratios of two continuous variables (e.g., revenue per transaction, items per user). This often requires specialized statistical methods.",
    "Multiple Comparisons Adjustment (Frequentist)": "Implement Bonferroni correction, Benjamini-Hochberg, or other methods when multiple variations are compared against a single control to manage the family-wise error rate."
}

# --- Helper Functions ---
def calculate_binary_sample_size(baseline_cr, mde_abs, power, alpha, num_variations):
    """Calculates sample size per variation for binary outcome A/B tests."""
    if not (0 < baseline_cr < 1): return None, "Baseline Conversion Rate (BCR) must be between 0 and 1 (exclusive)."
    if mde_abs <= 0: return None, "Minimum Detectable Effect (MDE) must be positive."
    if not (0 < power < 1): return None, "Statistical Power must be between 0 and 1 (exclusive)."
    if not (0 < alpha < 1): return None, "Significance Level (Alpha) must be between 0 and 1 (exclusive)."
    if num_variations < 2: return None, "Number of variations (including control) must be at least 2."
    
    p1 = baseline_cr
    p2 = baseline_cr + mde_abs
    if not (0 < p2 < 1): # Check if target CR is within valid range
        return None, f"The sum of Baseline CR ({baseline_cr*100:.1f}%) and MDE ({mde_abs*100:.1f}%) results in a target CR of {p2*100:.1f}%, which is outside the valid range (0-100%). Please adjust BCR or MDE."

    try:
        z_alpha_half = norm.ppf(1 - alpha / 2)
        z_beta = norm.ppf(power)
    except Exception as e:
        return None, f"Error calculating Z-scores: {e}. Ensure Alpha and Power are valid probabilities."
    
    var_p1 = p1 * (1 - p1)
    var_p2 = p2 * (1 - p2)
    
    numerator = (z_alpha_half + z_beta)**2 * (var_p1 + var_p2)
    denominator = mde_abs**2
    
    if denominator == 0: return None, "MDE cannot be zero, as it leads to division by zero." 
    
    n_per_variation = math.ceil(numerator / denominator)
    return n_per_variation, None

def calculate_continuous_sample_size(baseline_mean, std_dev, mde_abs_mean, power, alpha, num_variations):
    """Calculates sample size per variation for continuous outcome A/B tests."""
    if std_dev <= 0: return None, "Standard Deviation must be positive and greater than zero."
    if mde_abs_mean == 0: return None, "Minimum Detectable Effect (MDE) for means cannot be zero." 
    if mde_abs_mean < 0: mde_abs_mean = abs(mde_abs_mean) 
    if not (0 < power < 1): return None, "Statistical Power must be between 0 and 1 (exclusive)."
    if not (0 < alpha < 1): return None, "Significance Level (Alpha) must be between 0 and 1 (exclusive)."
    if num_variations < 2: return None, "Number of variations (including control) must be at least 2."

    try:
        z_alpha_half = norm.ppf(1 - alpha / 2)
        z_beta = norm.ppf(power)
    except Exception as e:
        return None, f"Error calculating Z-scores: {e}. Ensure Alpha and Power are valid probabilities."
        
    n_per_variation = (2 * (std_dev**2) * (z_alpha_half + z_beta)**2) / (mde_abs_mean**2)
    
    return math.ceil(n_per_variation), None

def run_bayesian_binary_analysis(summary_stats, control_group_name, prior_alpha=1, prior_beta=1, n_samples=10000, ci_level=0.95):
    """Performs Bayesian analysis for binary outcomes using a Beta-Binomial model."""
    results = {}; posterior_params = {}
    if summary_stats is None or summary_stats.empty: 
        return None, "Summary statistics are empty or None for Bayesian binary analysis."
    if 'Variation' not in summary_stats.columns or 'Users' not in summary_stats.columns or 'Conversions' not in summary_stats.columns:
        return None, "Summary statistics DataFrame is missing required columns: 'Variation', 'Users', 'Conversions'."

    for index, row in summary_stats.iterrows():
        var_name = str(row['Variation']) 
        users = int(row['Users']) if pd.notna(row['Users']) else 0
        conversions = int(row['Conversions']) if pd.notna(row['Conversions']) else 0
        if users < 0 or conversions < 0 or conversions > users:
            st.warning(f"Invalid user/conversion counts for {var_name} (Users: {users}, Converts: {conversions}). Skipping Bayesian calculation for this variation.")
            results[var_name] = {k: np.nan for k in ['samples', 'mean_cr', 'median_cr', 'cr_ci_low', 'cr_ci_high', 'alpha_post', 'beta_post', 'diff_samples_vs_control', 'prob_better_than_control', 'uplift_ci_low', 'uplift_ci_high', 'expected_uplift_abs', 'prob_best']}
            results[var_name]['samples'] = np.full(n_samples, np.nan) 
            continue

        alpha_post = prior_alpha + conversions; beta_post = prior_beta + (users - conversions)
        posterior_params[var_name] = {'alpha': alpha_post, 'beta': beta_post}
        
        if users > 0 and alpha_post > 0 and beta_post > 0 : 
            try:
                samples = beta_dist.rvs(alpha_post, beta_post, size=n_samples)
                results[var_name] = {
                    'samples': samples, 'mean_cr': np.mean(samples), 'median_cr': np.median(samples), 
                    'cr_ci_low': beta_dist.ppf((1-ci_level)/2, alpha_post, beta_post), 
                    'cr_ci_high': beta_dist.ppf(1-(1-ci_level)/2, alpha_post, beta_post), 
                    'alpha_post': alpha_post, 'beta_post': beta_post, 'diff_samples_vs_control': None 
                }
            except Exception as e:
                st.warning(f"Error generating Beta samples for {var_name} (α={alpha_post}, β={beta_post}): {e}")
                results[var_name] = {k: np.nan for k in ['samples', 'mean_cr', 'median_cr', 'cr_ci_low', 'cr_ci_high', 'alpha_post', 'beta_post', 'diff_samples_vs_control', 'prob_better_than_control', 'uplift_ci_low', 'uplift_ci_high', 'expected_uplift_abs', 'prob_best']}
                results[var_name]['samples'] = np.full(n_samples, np.nan) 
        else: 
             results[var_name] = {
                'samples': np.full(n_samples, np.nan), 'mean_cr': np.nan, 'median_cr': np.nan,
                'cr_ci_low': np.nan, 'cr_ci_high': np.nan, 'alpha_post': alpha_post, 'beta_post': beta_post,
                'diff_samples_vs_control': np.full(n_samples, np.nan), 'prob_better_than_control': np.nan,
                'uplift_ci_low': np.nan, 'uplift_ci_high': np.nan, 'expected_uplift_abs': np.nan, 'prob_best': np.nan
            }
    
    if control_group_name not in results or (results[control_group_name] and np.all(np.isnan(results[control_group_name]['samples']))): 
        for var_name_key in results: 
            results[var_name_key].update({'prob_better_than_control': np.nan, 'uplift_ci_low': np.nan, 
                                      'uplift_ci_high': np.nan, 'expected_uplift_abs': np.nan, 'prob_best': np.nan})
        err_msg = (f"Control group '{control_group_name}' not found in results for Bayesian analysis." 
                   if control_group_name not in results else 
                   f"Control group '{control_group_name}' has insufficient data for Bayesian comparison.")
        return results, err_msg

    control_samples = results[control_group_name]['samples']
    
    for var_name, data in results.items():
        if var_name == control_group_name or np.all(np.isnan(data['samples'])): 
            data.update({'prob_better_than_control': None if var_name == control_group_name else np.nan, 
                         'uplift_ci_low': None if var_name == control_group_name else np.nan, 
                         'uplift_ci_high': None if var_name == control_group_name else np.nan, 
                         'expected_uplift_abs': None if var_name == control_group_name else np.nan})
            if np.all(np.isnan(data['samples'])) and var_name != control_group_name:
                 data['diff_samples_vs_control'] = np.full(n_samples, np.nan)
            continue

        var_samples = data['samples']; diff_samples = var_samples - control_samples
        data['diff_samples_vs_control'] = diff_samples
        valid_diff = diff_samples[~np.isnan(diff_samples)]
        if valid_diff.size > 0:
            data['prob_better_than_control'] = np.mean(valid_diff > 0)
            data['uplift_ci_low'] = np.percentile(valid_diff, (1-ci_level)/2 * 100)
            data['uplift_ci_high'] = np.percentile(valid_diff, (1-(1-ci_level)/2) * 100)
            data['expected_uplift_abs'] = np.mean(valid_diff)
        else:
            data.update({'prob_better_than_control': np.nan, 'uplift_ci_low': np.nan, 
                         'uplift_ci_high': np.nan, 'expected_uplift_abs': np.nan})
            
    all_var_names = summary_stats['Variation'].astype(str).tolist() 
    valid_ordered_vars = [name for name in all_var_names if name in results and not np.all(np.isnan(results[name]['samples']))]

    if not valid_ordered_vars:
        for var_name_key in all_var_names: 
            if var_name_key in results: results[var_name_key]['prob_best'] = np.nan
            else: results[var_name_key] = {'prob_best': np.nan}
        return results, "No variations with valid data for P(Best) calculation."
    
    all_samples_matrix = np.array([results[var]['samples'] for var in valid_ordered_vars])
    best_variation_counts = np.zeros(len(all_var_names)) 

    if all_samples_matrix.ndim == 2 and all_samples_matrix.shape[0] > 0 and all_samples_matrix.shape[1] == n_samples:
        valid_iterations = 0
        for i in range(n_samples):
            current_iter_samples = all_samples_matrix[:, i]
            if not np.all(np.isnan(current_iter_samples)):
                valid_iterations +=1
                best_idx_in_temp_matrix = np.nanargmax(current_iter_samples)
                best_var_name_this_iter = valid_ordered_vars[best_idx_in_temp_matrix]
                original_idx_for_counts = all_var_names.index(best_var_name_this_iter)
                best_variation_counts[original_idx_for_counts] += 1
        
        if valid_iterations > 0:
            prob_best = best_variation_counts / valid_iterations
        else:
            prob_best = np.full(len(all_var_names), np.nan)

        for i, var_name_key in enumerate(all_var_names):
            if var_name_key in results: 
                results[var_name_key]['prob_best'] = prob_best[i] if var_name_key in valid_ordered_vars else np.nan
            else: 
                 results[var_name_key] = {'prob_best': np.nan} 
    else: 
        for var_name_key in all_var_names:
            if var_name_key in results:
                is_single_valid = (len(valid_ordered_vars) == 1 and var_name_key == valid_ordered_vars[0])
                results[var_name_key]['prob_best'] = 1.0 if is_single_valid else (0.0 if len(valid_ordered_vars) > 1 else np.nan)
            else: 
                results[var_name_key] = {'prob_best': np.nan}
                
    return results, None

def run_bayesian_continuous_analysis(summary_stats, control_group_name, n_samples=10000, ci_level=0.95):
    """Performs Bayesian analysis for continuous outcomes using a t-distribution approximation."""
    results = {}
    if summary_stats is None or summary_stats.empty:
        return None, "Summary statistics are empty or None for Bayesian continuous analysis."
    
    if 'Variation' not in summary_stats.columns:
        return None, "Summary statistics missing 'Variation' column."
    if not {'Users', 'Mean_Value', 'Std_Dev'}.issubset(summary_stats.columns):
        return None, "Summary statistics missing one or more required columns: 'Users', 'Mean_Value', 'Std_Dev'."

    for index, row in summary_stats.iterrows():
        var_name = str(row['Variation']) 
        n = int(row['Users']) if pd.notna(row['Users']) else 0
        sample_mean = row['Mean_Value'] if pd.notna(row['Mean_Value']) else np.nan
        sample_std_dev = row['Std_Dev'] if pd.notna(row['Std_Dev']) else np.nan
        
        current_samples = np.array([])
        posterior_mean_val = np.nan
        ci_low_val = np.nan
        ci_high_val = np.nan

        if n <= 1 or pd.isna(sample_mean) or pd.isna(sample_std_dev) or (sample_std_dev == 0 and n <=1) : 
            current_samples = np.full(n_samples, sample_mean if n == 1 and pd.notna(sample_mean) else np.nan) 
            posterior_mean_val = sample_mean if n == 1 and pd.notna(sample_mean) else np.nan
        elif sample_std_dev == 0 and n > 1: 
            current_samples = np.full(n_samples, sample_mean)
            posterior_mean_val = sample_mean
            ci_low_val = sample_mean
            ci_high_val = sample_mean
        else: 
            df_t_dist = n - 1 
            std_err = sample_std_dev / np.sqrt(n) # Ensure sample_std_dev is not NaN here
            if pd.notna(std_err) and std_err > 0: # Check if std_err is valid
                try:
                    t_samples = t_dist.rvs(df_t_dist, size=n_samples)
                    current_samples = sample_mean + (std_err * t_samples)
                    posterior_mean_val = np.mean(current_samples)
                    ci_low_val = np.percentile(current_samples, (1 - ci_level) / 2 * 100)
                    ci_high_val = np.percentile(current_samples, (1 - (1 - ci_level) / 2) * 100)
                except Exception as e:
                    st.warning(f"Error generating t-distribution samples for {var_name}: {e}")
                    current_samples = np.full(n_samples, np.nan) 
            else: 
                 current_samples = np.full(n_samples, sample_mean if pd.notna(sample_mean) else np.nan)
                 posterior_mean_val = sample_mean if pd.notna(sample_mean) else np.nan
                 ci_low_val = sample_mean if pd.notna(sample_mean) else np.nan
                 ci_high_val = sample_mean if pd.notna(sample_mean) else np.nan


        results[var_name] = {
            'samples': current_samples,
            'posterior_mean': posterior_mean_val,
            'mean_ci_low': ci_low_val,
            'mean_ci_high': ci_high_val,
            'diff_samples_vs_control': None 
        }

    if control_group_name not in results or (results[control_group_name] and np.all(np.isnan(results[control_group_name]['samples']))):
        for var_name_key in results: 
            results[var_name_key].update({'prob_better_than_control': np.nan, 'diff_ci_low': np.nan, 
                                      'diff_ci_high': np.nan, 'expected_diff_abs': np.nan, 'prob_best': np.nan})
        err_msg = (f"Control group '{control_group_name}' not found in results for Bayesian analysis." 
                   if control_group_name not in results else 
                   f"Control group '{control_group_name}' has insufficient data for Bayesian comparison.")
        return results, err_msg

    control_samples = results[control_group_name]['samples']
    
    for var_name, data in results.items():
        if var_name == control_group_name or np.all(np.isnan(data['samples'])):
            data.update({'prob_better_than_control': None if var_name == control_group_name else np.nan, 
                         'diff_ci_low': None if var_name == control_group_name else np.nan, 
                         'diff_ci_high': None if var_name == control_group_name else np.nan, 
                         'expected_diff_abs': None if var_name == control_group_name else np.nan})
            if np.all(np.isnan(data['samples'])) and var_name != control_group_name:
                 data['diff_samples_vs_control'] = np.full(n_samples, np.nan)
            continue

        var_samples = data['samples']
        diff_samples = var_samples - control_samples
        data['diff_samples_vs_control'] = diff_samples
        
        valid_diff_samples = diff_samples[~np.isnan(diff_samples)]
        if valid_diff_samples.size > 0:
            data['prob_better_than_control'] = np.mean(valid_diff_samples > 0)
            data['diff_ci_low'] = np.percentile(valid_diff_samples, (1 - ci_level) / 2 * 100)
            data['diff_ci_high'] = np.percentile(valid_diff_samples, (1 - (1 - ci_level) / 2) * 100)
            data['expected_diff_abs'] = np.mean(valid_diff_samples)
        else: 
            data.update({'prob_better_than_control': np.nan, 'diff_ci_low': np.nan, 
                         'diff_ci_high': np.nan, 'expected_diff_abs': np.nan})

    all_var_names_cont = summary_stats['Variation'].astype(str).tolist() 
    valid_var_names_for_pbest = [name for name in all_var_names_cont if name in results and not np.all(np.isnan(results[name]['samples']))]

    if not valid_var_names_for_pbest:
        for var_name_key in all_var_names_cont: 
            if var_name_key in results: results[var_name_key]['prob_best'] = np.nan
            else: results[var_name_key] = {'prob_best': np.nan} 
        return results, "No variations with valid data found for P(Best) calculation in continuous Bayesian analysis."

    all_samples_matrix_cont = np.array([results[var]['samples'] for var in valid_var_names_for_pbest])
    best_variation_counts_cont = np.zeros(len(all_var_names_cont))

    if all_samples_matrix_cont.ndim == 2 and all_samples_matrix_cont.shape[0] > 0 and all_samples_matrix_cont.shape[1] == n_samples:
        valid_iterations_for_pbest = 0
        for i in range(n_samples): 
            current_iter_samples = all_samples_matrix_cont[:, i]
            if not np.all(np.isnan(current_iter_samples)): 
                valid_iterations_for_pbest += 1
                best_idx_in_temp_matrix = np.nanargmax(current_iter_samples)
                best_var_name_this_iter = valid_var_names_for_pbest[best_idx_in_temp_matrix]
                original_idx_for_counts = all_var_names_cont.index(best_var_name_this_iter)
                best_variation_counts_cont[original_idx_for_counts] += 1
        
        if valid_iterations_for_pbest > 0:
            prob_best_cont = best_variation_counts_cont / valid_iterations_for_pbest
        else: 
            prob_best_cont = np.full(len(all_var_names_cont), np.nan)

        for i, var_name_key in enumerate(all_var_names_cont):
            if var_name_key in results:
                results[var_name_key]['prob_best'] = prob_best_cont[i] if var_name_key in valid_var_names_for_pbest else np.nan
            else: 
                results[var_name_key] = {'prob_best': np.nan} 
                
    else: 
        for var_name_key in all_var_names_cont:
            if var_name_key in results:
                is_single_valid_var = (len(valid_var_names_for_pbest) == 1 and var_name_key == valid_var_names_for_pbest[0])
                results[var_name_key]['prob_best'] = 1.0 if is_single_valid_var else (0.0 if len(valid_var_names_for_pbest) > 1 else np.nan)
            else:
                results[var_name_key] = {'prob_best': np.nan}
                
    return results, None


# --- Page Functions ---
def show_introduction_page():
    # Displays the introduction to A/B testing.
    st.header("Introduction to A/B Testing 🧪")
    st.markdown("""
    Welcome to the A/B Testing Guide & Analyzer! This application is designed to assist you in understanding, designing,
    analyzing, and interpreting A/B tests, empowering you to make data-driven decisions.
    Whether you're new to A/B testing or looking for a handy tool, this guide aims to be a valuable resource.
    """) 
    st.markdown("---")
    st.subheader("What is A/B Testing?") 
    st.markdown("""
    A/B testing (also known as split testing or bucket testing) is a method of comparing two or more versions 
    of a webpage, app feature, email, advertisement, or other marketing asset to determine which one performs better 
    in achieving a specific goal (e.g., higher conversion rate, more clicks, increased engagement).

    The core idea is to make **data-driven decisions** rather than relying on gut feelings or opinions. 
    You show one version (the 'control' or 'A') to one group of users, and another version (the 'variation' or 'B') 
    to a different, similarly sized group of users, simultaneously. Then, you measure how each version performs 
    based on your key metric.
    """)
    st.markdown("*Analogy:* Imagine you're a chef with two different recipes for a cake (Recipe A and Recipe B). You want to know which one your customers like more. You bake both cakes and offer a slice of Recipe A to one group of customers and a slice of Recipe B to another. Then, you ask them which one they preferred or count how many slices of each were eaten. That's essentially what A/B testing does for digital experiences!")
    st.markdown("---")
    st.subheader("Why Use A/B Testing? (The Benefits)")
    st.markdown("""
    A/B testing is a powerful tool because it can help you:
    * ✅ **Improve Key Metrics:** Increase conversion rates, boost engagement, drive sales, or improve any other metric you care about.
    * 🛡️ **Reduce Risk:** Test changes on a smaller scale before rolling them out to your entire user base, minimizing the impact of potentially negative changes.
    * 💡 **Gain Insights:** Understand your users' behavior, preferences, and motivations better. Even a "failed" test (where the variation doesn't outperform the control) can provide valuable learnings.
    * ✨ **Optimize User Experience:** Make your website, app, or product more user-friendly and effective by identifying what resonates most with your audience.
    * 🔄 **Foster Iterative Improvement:** A/B testing supports a cycle of continuous learning and optimization, allowing for incremental gains over time.
    """)
    st.markdown("---")
    st.subheader("Basic A/B Testing Terminology")
    st.markdown("Here are a few key terms you'll encounter frequently. More detailed explanations are available and will appear in context throughout the app.")
    basic_terms = {
        "Control (Version A)": "The existing, unchanged version that you're comparing against. It acts as a baseline to measure performance.",
        "Variation (Version B, C, etc.)": "A modified version that you're testing to see if it performs differently than the control. You can test multiple variations against a control (A/B/n testing).",
        "Conversion / Goal / Metric": "The specific, measurable action or outcome you are tracking to determine success (e.g., a sign-up, a purchase, a click-through-rate, average order value).",
        "Conversion Rate (CR)": "For binary metrics, this is the percentage of users who complete the desired goal, out of the total number of users in that group (e.g., (Conversions / Total Users) * 100%)."
    }
    for term, definition in basic_terms.items(): st.markdown(f"**{term}:** {definition}")
    with st.expander("📖 Learn more about other common A/B testing terms..."):
        st.markdown("""
        * **Lift / Uplift:** The percentage increase (or decrease) in performance of a variation compared to the control. Calculated as `((Variation Metric - Control Metric) / Control Metric) * 100%`.
        * **Statistical Significance (p-value, Alpha):** In frequentist testing, this indicates whether an observed difference is likely due to a real effect or just random chance. A p-value less than the significance level (alpha, typically 0.05) suggests the result is statistically significant.
        * **Confidence Interval (CI):** A range of values, derived from sample data, that is likely to contain the true value of an unknown population parameter (e.g., the true difference in conversion rates) with a certain degree of confidence (e.g., 95%).
        * **Credible Interval (CrI) (Bayesian):** A range of values within which an unobserved parameter value lies with a particular probability, based on the data and the prior distribution. Often interpreted more directly than a frequentist CI.
        * **Statistical Power (1 - β or Beta):** The probability that your test will correctly detect a real difference (i.e., reject a false null hypothesis) if one actually exists at a certain magnitude (MDE). Typically aimed for 80% or higher.
        * **Minimum Detectable Effect (MDE):** The smallest change in your key metric that you want your test to be able to reliably detect. This should be a practically significant amount for your business.
        * *(More terms will be added and explained in context throughout the app's sections.)*
        """)
    st.markdown("---")
    st.subheader("The A/B Testing Process at a Glance")
    st.markdown("""
    A typical A/B testing process involves several key steps. This app is designed to help you with some of these:
    1.  🤔 **Define Your Goal & Formulate a Hypothesis:** What specific metric do you want to improve? What change do you believe will achieve this improvement, and why? (e.g., "Changing the button color from blue to green will increase click-through rate because green is more attention-grabbing.")
    2.  📐 **Design Your Test & Calculate Sample Size:** Determine your variations, how you'll split traffic, how long the test needs to run, and critically, calculate the sample size needed per variation for a statistically reliable test. (➡️ *The "Designing Your A/B Test" section will help here!*)
    3.  🚀 **Run Your Test & Collect Data:** Implement the test using your A/B testing platform or internal tools. Ensure data is collected accurately and consistently for all variations.
    4.  📊 **Analyze Your Results:** Once the test has run for the predetermined duration or reached the required sample size, process the collected data to compare the performance of your variations. (➡️ *The "Analyze Results" section is built for this!*)
    5.  🧐 **Interpret Results & Make a Decision:** Understand what the statistical outputs mean in the context of your business goals and make an informed decision. (➡️ *The "Interpreting Results & Detailed Decision Guidance" section will guide you.*)
    """)
    st.markdown("---")
    st.subheader("Where This App Fits In")
    st.markdown("This application aims to be your companion for the critical stages of A/B testing: * Helping you **design robust tests** by calculating the necessary sample size. * Enabling you to **analyze the data** you've collected using both Frequentist and Bayesian statistical approaches. * Guiding you in **interpreting those results** to make informed, data-driven decisions. * Providing **educational content** (like common pitfalls and FAQs) to improve your A/B testing knowledge.")

def show_design_test_page():
    # Page for designing A/B tests, including sample size calculators.
    st.header("Designing Your A/B Test 📐")
    st.markdown("A crucial step in designing an A/B test is determining the appropriate sample size. A test with too few users may not have enough statistical power to detect a real difference, while a test with too many users can be wasteful. This calculator will help you estimate the number of users needed per variation.")
    st.markdown("---")
    cycle_key_suffix_ds = "_c8_1" 
    metric_type_ss = st.radio("Select your primary metric type for sample size calculation:", ('Binary (e.g., Conversion Rate)', 'Continuous (e.g., Average Order Value)'), key=f"ss_metric_type_radio{cycle_key_suffix_ds}", help="Choose 'Binary' for yes/no outcomes like clicks or sign-ups. Choose 'Continuous' for numerical outcomes like revenue or time spent.") 
    st.markdown("---")
    if metric_type_ss == 'Binary (e.g., Conversion Rate)':
        st.subheader("Sample Size Calculator (for Binary Outcomes)")
        st.markdown("**Calculator Inputs:**")
        cols_bin = st.columns(2)
        with cols_bin[0]: baseline_cr_percent = st.number_input(label="Baseline Conversion Rate (BCR) (%)", min_value=0.1, max_value=99.9, value=5.0, step=0.1, format="%.1f", help="The current conversion rate of your control group, expressed as a percentage (e.g., 5 for 5%).", key=f"ss_bcr{cycle_key_suffix_ds}")
        with cols_bin[1]: mde_abs_percent = st.number_input(label="Minimum Detectable Effect (MDE) - Absolute (%)", min_value=0.1, max_value=50.0, value=1.0, step=0.1, format="%.1f", help="The smallest *absolute* increase in conversion rate you want to be able to detect (e.g., if BCR is 5% and MDE is 1%, you're targeting 6%).", key=f"ss_mde{cycle_key_suffix_ds}")
        cols2_bin = st.columns(2)
        with cols2_bin[0]: power_percent_bin = st.slider(label="Statistical Power (1 - \u03B2) (%)", min_value=50, max_value=99, value=80, step=1, format="%d%%", help="The probability of detecting an effect if one truly exists (typically 80-90%). Higher power requires more samples.", key=f"ss_power{cycle_key_suffix_ds}")
        with cols2_bin[1]: alpha_percent_bin = st.slider(label="Significance Level (\u03B1) (%) - Two-sided", min_value=1, max_value=20, value=5, step=1, format="%d%%", help="The probability of a Type I error (false positive) - concluding there's an effect when there isn't (typically 1-5%). Lower alpha requires more samples.", key=f"ss_alpha{cycle_key_suffix_ds}")
        num_variations_ss_bin = st.number_input(label="Number of Variations (including Control)", min_value=2, value=2, step=1, help="Total number of versions being tested (e.g., Control + 1 Variation = 2).", key=f"ss_num_var{cycle_key_suffix_ds}")
        st.markdown("<p style='font-size: smaller; font-style: italic;'>Note: For binary outcomes, sample size is based on a two-proportion z-test, comparing each variation to the control at the specified alpha level.</p>", unsafe_allow_html=True)
        if st.button("Calculate Sample Size (Binary)", key=f"ss_calc_button{cycle_key_suffix_ds}_bin", help="Click to calculate the required sample size based on your inputs."):
            baseline_cr, mde_abs, power, alpha = baseline_cr_percent/100.0, mde_abs_percent/100.0, power_percent_bin/100.0, alpha_percent_bin/100.0
            sample_size, error_msg = calculate_binary_sample_size(baseline_cr, mde_abs, power, alpha, num_variations_ss_bin)
            if error_msg: st.error(f"Input Error: {error_msg}")
            elif sample_size:
                st.success("Calculation Successful!")
                target_cr_pct = (baseline_cr + mde_abs) * 100
                res_cols = st.columns(2); res_cols[0].metric("Required Sample Size PER Variation", f"{sample_size:,}"); res_cols[1].metric("Total Required Sample Size", f"{(sample_size * num_variations_ss_bin):,}")
                st.markdown(f"**Summary of Inputs Used:**\n- Baseline Conversion Rate (BCR): `{baseline_cr_percent:.1f}%`\n- Absolute MDE: `{mde_abs_percent:.1f}%` (Targeting a CR of at least `{target_cr_pct:.2f}%` for variations)\n- Statistical Power: `{power_percent_bin}%` (1 - \u03B2)\n- Significance Level (\u03B1): `{alpha_percent_bin}%` (two-sided)\n- Number of Variations: `{num_variations_ss_bin}`.")
                with st.expander("Show Formula Used (Binary)"):
                    st.markdown("For comparing two proportions ($n$ per group):")
                    st.latex(r'''n = \frac{(Z_{\alpha/2} + Z_{\beta})^2 \cdot (p_1(1-p_1) + p_2(1-p_2))}{(p_2 - p_1)^2}''')
                    st.markdown(r"**Where:** $n$=Sample size per variation, $p_1$=BCR, $p_2=p_1+\text{MDE}$, $Z_{\alpha/2}$=Z-score for $\alpha$, $Z_{\beta}$=Z-score for power.")
    elif metric_type_ss == 'Continuous (e.g., Average Order Value)':
        st.subheader("Sample Size Calculator (for Continuous Outcomes)")
        st.markdown("**Calculator Inputs:**")
        cols_cont = st.columns(2)
        with cols_cont[0]: baseline_mean = st.number_input(label="Baseline Mean (Control Group)", value=100.0, step=1.0, format="%.2f", help="The current average value of your metric for the control group (e.g., average order value, average time on page).", key=f"ss_mean{cycle_key_suffix_ds}")
        with cols_cont[1]: std_dev = st.number_input(label="Standard Deviation (of the metric)", value=20.0, min_value=0.01, step=0.1, format="%.2f", help="Estimated standard deviation of your continuous metric. Obtain from historical data if possible. Must be greater than 0.", key=f"ss_stddev{cycle_key_suffix_ds}") 
        mde_abs_mean = st.number_input(label="Minimum Detectable Effect (MDE) - Absolute Mean Difference", value=5.0,min_value=0.01, step=0.1, format="%.2f", help="Smallest *absolute* difference in means you want to detect (e.g., a $2 increase in AOV). Must be greater than 0.", key=f"ss_mde_mean{cycle_key_suffix_ds}")
        cols2_cont = st.columns(2)
        with cols2_cont[0]: power_percent_cont = st.slider(label="Statistical Power (1 - \u03B2) (%)", min_value=50, max_value=99, value=80, step=1, format="%d%%", help="The probability of detecting an effect if one truly exists (typically 80-90%). Higher power requires more samples.", key=f"ss_power_cont{cycle_key_suffix_ds}")
        with cols2_cont[1]: alpha_percent_cont = st.slider(label="Significance Level (\u03B1) (%) - Two-sided", min_value=1, max_value=20, value=5, step=1, format="%d%%", help="The probability of a Type I error (false positive) - concluding there's an effect when there isn't (typically 1-5%). Lower alpha requires more samples.", key=f"ss_alpha_cont{cycle_key_suffix_ds}")
        num_variations_ss_cont = st.number_input(label="Number of Variations (including Control)", min_value=2, value=2, step=1, help="Total number of versions being tested.", key=f"ss_num_var_cont{cycle_key_suffix_ds}")
        st.markdown("<p style='font-size: smaller; font-style: italic;'>Note: For continuous outcomes, sample size is based on a two-sample t-test, comparing each variation to the control at the specified alpha, assuming similar standard deviations across groups.</p>", unsafe_allow_html=True)
        if st.button("Calculate Sample Size (Continuous)", key=f"ss_calc_button{cycle_key_suffix_ds}_cont", help="Click to calculate the required sample size based on your inputs."):
            power, alpha = power_percent_cont/100.0, alpha_percent_cont/100.0
            sample_size, error_msg = calculate_continuous_sample_size(baseline_mean, std_dev, mde_abs_mean, power, alpha, num_variations_ss_cont)
            if error_msg: st.error(f"Input Error: {error_msg}")
            elif sample_size:
                st.success("Calculation Successful!")
                target_mean_positive = baseline_mean + mde_abs_mean 
                target_mean_negative = baseline_mean - mde_abs_mean
                res_cols = st.columns(2); res_cols[0].metric("Required Sample Size PER Variation", f"{sample_size:,}"); res_cols[1].metric("Total Required Sample Size", f"{(sample_size * num_variations_ss_cont):,}")
                st.markdown(f"**Summary of Inputs Used:**\n- Baseline Mean: `{baseline_mean:.2f}`\n- Estimated Standard Deviation: `{std_dev:.2f}`\n- Absolute MDE (Mean Difference): `{mde_abs_mean:.2f}` (Targeting a mean of approx. `{target_mean_positive:.2f}` or `{target_mean_negative:.2f}` for variations)\n- Statistical Power: `{power_percent_cont}%` (1 - \u03B2)\n- Significance Level (\u03B1): `{alpha_percent_cont}%` (two-sided)\n- Number of Variations: `{num_variations_ss_cont}`.")
                with st.expander("Show Formula Used (Continuous)"):
                    st.markdown("For comparing two means ($n$ per group, assuming similar standard deviations $\sigma$):")
                    st.latex(r'''n = \frac{2 \cdot \sigma^2 \cdot (Z_{\alpha/2} + Z_{\beta})^2}{\text{MDE}^2}''')
                    st.markdown(r"**Where:** $n$=Sample size per variation, $\sigma$=Standard Deviation, MDE=Absolute difference in means, $Z_{\alpha/2}$=Z-score for $\alpha$, $Z_{\beta}$=Z-score for power.")
    st.markdown("---")
    with st.expander("💡 Understanding Input Impacts on Sample Size"):
        st.markdown(r"""Adjusting input parameters affects required sample size. Understanding these trade-offs is key:
        * **Baseline Conversion Rate (BCR) / Baseline Mean:**
        * *Binary:* Sample size is largest when BCR is near 50%. It's harder to detect changes for very low or very high BCRs.
        * *Continuous:* The baseline mean itself doesn't directly affect the sample size formula as much as Standard Deviation and MDE do, but understanding it is crucial for setting a meaningful MDE.
        * **Standard Deviation (Continuous Metrics Only):**
        * *Impact:* *Increasing* Standard Deviation **significantly increases** required sample size.
        * *Trade-off:* Higher variability in your metric naturally requires more data to detect a true signal from noise. Reducing underlying variability (if possible through better targeting or more consistent user experiences) can make tests more efficient.
        * **Minimum Detectable Effect (MDE):** (Applies to both absolute difference in CRs or Means)
        * *Impact:* *Decreasing* MDE (i.e., trying to detect smaller changes) **significantly increases** sample size.
        * *Trade-off:* Detecting smaller changes usually requires more samples and longer test durations. A larger MDE is cheaper/faster to test but you risk missing smaller, yet potentially valuable, effects. Balance what's meaningful to the business with what's feasible to test.
        * **Statistical Power (1 - $\beta$):**
        * *Impact:* *Increasing* power **increases** sample size.
        * *Trade-off:* Higher power reduces the risk of a Type II error (false negative - failing to detect a real effect). This increased confidence comes at the cost of more samples.
        * **Significance Level ($\alpha$):**
        * *Impact:* *Decreasing* $\alpha$ (e.g., from 5% to 1%, making the test more stringent) **increases** sample size.
        * *Trade-off:* A lower alpha reduces the risk of a Type I error (false positive - concluding an effect exists when it doesn't). This increased certainty also requires more samples.
        * **Number of Variations:** Total sample size increases proportionally with the number of variations if you want to maintain the same power for pairwise comparisons against the control.
        Balancing these factors is key for designing feasible and statistically sound A/B tests.
        """)
    st.markdown("---")
    st.subheader("Common Pitfalls in A/B Test Design & Execution")
    pitfalls = {
        "Too Short Test Duration / Insufficient Sample Size": {"what": "Ending a test before collecting enough data (as determined by sample size calculation) or running it for an arbitrarily short period (e.g., less than a full business cycle).", "problem": "Results may be statistically underpowered (high chance of missing a real effect) or heavily influenced by random daily fluctuations and novelty effects, leading to unreliable conclusions.", "howto": "Calculate the required sample size *before* starting the test. Run the test until that sample size is achieved per variation. It's also generally recommended to run tests for at least one full business cycle (e.g., one week, or two weeks if your traffic varies significantly by day of week) to account for natural variations in user behavior.", "analogy": "Judging a marathon by the first 100 meters, or a restaurant's popularity based on just one hour of service."},
        "Ignoring Statistical Significance / Power": {"what": "Making decisions based on observed differences without considering if those differences are statistically significant or if the test had enough power to detect a meaningful effect.", "problem": "You might implement changes that have no real impact (or even a negative one if the observed difference was due to chance), or you might discard truly better variations because the test was underpowered.", "howto": "Always check p-values and Confidence/Credible Intervals (Frequentist) or relevant probabilities and Credible Intervals (Bayesian). Ensure your test was designed with adequate power (typically 80%+) for an MDE you care about.", "analogy": "Flipping a coin 3 times and getting 2 heads doesn't mean the coin is biased. You need more data (a larger sample) to be confident."},
        "Testing Too Many Things at Once (in a Single Variation)": {"what": "Changing multiple elements (e.g., headline, image, button color, layout) in one variation compared to the control.", "problem": "If the variation performs differently, it's impossible to isolate which specific change caused the performance difference. You lose the ability to learn what worked.", "howto": "Ideally, test one significant change at a time to understand its specific impact. If testing multiple changes, they should be part of a cohesive new design concept, and you acknowledge you're testing the *concept* as a whole.", "analogy": "Changing multiple ingredients in a cake recipe at once. If the cake tastes better (or worse), you won't know which ingredient change was responsible."},
        "External Factors Affecting the Test": {"what": "Uncontrolled events outside your test (e.g., holidays, major news events, marketing campaigns, website outages, competitor actions) influencing user behavior during the test period.", "problem": "These can skew results, making variations appear better or worse due to these external factors rather than the changes you made.", "howto": "Be aware of the broader environment during your test. Document any significant external events that occur. If a major event heavily skews data, consider pausing or restarting the test, or at least segmenting out the affected period if possible.", "analogy": "Measuring plant growth with different fertilizers during a surprise, unseasonal heatwave that affects all plants. The heatwave is a confounding variable."},
        "Regression to the Mean": {"what": "The phenomenon where initial extreme results (either very high or very low) tend to become closer to the average over time as more data is collected.", "problem": "Stopping tests early based on exciting (but potentially random) initial extreme results can be misleading. A variation might look like a huge winner initially, only for its performance to normalize.", "howto": "Run tests for their planned duration and sample size. Don't get overly excited or discouraged by very early results.", "analogy": "A basketball player might have an unusually 'hot' shooting streak for a few minutes, but their overall game performance will likely regress towards their season average."},
        "Not Segmenting Results (When Appropriate)": {"what": "Only looking at overall average results and not analyzing how different user segments (e.g., new vs. returning users, mobile vs. desktop, different demographics) reacted to the variations.", "problem": "Overall 'flat' or inconclusive results might hide significant wins in one segment and losses in another, canceling each other out. You might miss opportunities to personalize experiences or identify a change that's great for a specific valuable segment.", "howto": "If you have hypotheses about how different segments might react, or if segments are strategically important, plan to analyze them. Ensure segments are large enough for meaningful analysis (calculate sample size per segment if it's a primary goal). This app will support basic segmentation display in later cycles.", "analogy": "A new song might have mixed reviews overall (average rating), but specific age groups or fans of a certain genre might love it, while others dislike it."},
        "Peeking at Results Too Often (and Stopping Early with Frequentist Methods)": {"what": "Constantly monitoring test results and stopping the test as soon as statistical significance (e.g., p < 0.05) is (randomly) hit, especially if the planned sample size hasn't been reached.", "problem": "This dramatically increases the Type I error rate (false positives). Each 'peek' is like an extra chance to find a significant result purely by chance. This is often called 'p-hacking'.", "howto": "Determine your sample size and test duration *in advance*. Avoid making decisions based on interim results unless you are using specific sequential testing methods designed for peeking (e.g., SPRT, Bayesian methods which are more robust to this).", "analogy": "If you flip a coin repeatedly and stop the 'experiment' the very first moment you get a streak of 3 heads, you might wrongly conclude the coin is biased. You need to commit to a set number of flips (sample size) beforehand."},
        "Simpson's Paradox": {"what": "A statistical phenomenon where a trend appears in several different groups of data but disappears or reverses when these groups are combined.", "problem": "Aggregated data can sometimes lead to incorrect conclusions if there's an important confounding variable that differs across variations (e.g., if one variation accidentally got more mobile users, and mobile users convert differently).", "howto": "Be aware of potential confounding variables. Randomization in A/A testing helps, but also check for imbalances in key segment distributions between variations. Analyzing important segments can help uncover this.", "analogy": "Hospital A has a higher overall patient survival rate than Hospital B. However, Hospital A might specialize in less severe cases, while Hospital B takes more critical patients. If you look at survival rates *within each severity category* (e.g., mild, moderate, severe cases), Hospital B might actually have better survival rates for *each* category."}
    }
    for pitfall, details in pitfalls.items():
        with st.expander(f"⚠️ {pitfall}"):
            st.markdown(f"**What it is:** {details['what']}")
            st.markdown(f"**Why it's a problem:** {details['problem']}")
            st.markdown(f"**How to avoid it / What to do:** {details['howto']}")
            st.markdown(f"**Analogy / Example:** {details['analogy']}")
    st.markdown("---")
    st.info("Sample Size Calculator now supports both Binary and Continuous outcomes!")

# --- Refactored Analysis Display Functions ---
def display_frequentist_analysis(df_for_analysis, metric_type, outcome_col, variation_col, control_name, alpha, summary_stats_key_suffix=""):
    """
    Helper function to calculate and display Frequentist analysis results.
    Can be used for overall data or a specific segment.
    Returns the summary statistics DataFrame for potential reuse.
    """
    if df_for_analysis is None or df_for_analysis.empty:
        st.warning(f"No data provided for Frequentist analysis {summary_stats_key_suffix}.")
        return None

    # Calculate summary statistics
    if metric_type == 'Binary':
        if '__outcome_processed__' not in df_for_analysis.columns:
            st.error(f"Internal error: '__outcome_processed__' column missing for binary frequentist analysis {summary_stats_key_suffix}.")
            return None
        summary_stats = df_for_analysis.groupby(variation_col).agg(
            Users=('__outcome_processed__', 'count'),
            Conversions=('__outcome_processed__', 'sum') 
        ).reset_index()
        summary_stats.rename(columns={variation_col: 'Variation'}, inplace=True)
        if summary_stats['Users'].sum() == 0: 
            st.warning(f"No users found in the provided data for binary Frequentist analysis {summary_stats_key_suffix}.")
            return None
        summary_stats['Metric Value (%)'] = (summary_stats['Conversions'] / summary_stats['Users'].replace(0, np.nan) * 100).round(2)
        metric_col_name_display = 'Metric Value (%)'
    else: # Continuous
        df_for_analysis_cont = df_for_analysis.copy() 
        df_for_analysis_cont[outcome_col] = pd.to_numeric(df_for_analysis_cont[outcome_col], errors='coerce')
        df_cleaned = df_for_analysis_cont.dropna(subset=[outcome_col])

        if df_cleaned.empty:
            st.warning(f"No valid numeric data in outcome column '{outcome_col}' for continuous Frequentist analysis {summary_stats_key_suffix}.")
            return None
        
        summary_stats = df_cleaned.groupby(variation_col).agg(
            Users=(outcome_col, 'count'),
            Mean_Value=(outcome_col, 'mean'),
            Std_Dev=(outcome_col, 'std'),
            Median_Value=(outcome_col, 'median'),
            Std_Err=(outcome_col, lambda x: x.std(ddof=1) / np.sqrt(x.count()) if x.count() > 0 and pd.notna(x.std(ddof=1)) else np.nan)
        ).reset_index()
        summary_stats.rename(columns={variation_col: 'Variation'}, inplace=True)
        if summary_stats['Users'].sum() == 0:
            st.warning(f"No users found in the provided data for continuous Frequentist analysis {summary_stats_key_suffix}.")
            return None
        for col_to_round in ['Mean_Value', 'Std_Dev', 'Median_Value', 'Std_Err']:
            if col_to_round in summary_stats.columns:
                summary_stats[col_to_round] = summary_stats[col_to_round].round(3)
        metric_col_name_display = 'Mean_Value'

    st.markdown("##### 📊 Descriptive Statistics")
    if metric_type == 'Binary':
        st.dataframe(summary_stats[['Variation', 'Users', 'Conversions', metric_col_name_display]].fillna('N/A (0 Users)'))
    else:
        st.dataframe(summary_stats[['Variation', 'Users', 'Mean_Value', 'Median_Value', 'Std_Dev', 'Std_Err']].fillna('N/A'))

    if metric_type == 'Continuous':
        for index, row in summary_stats.iterrows():
            if pd.notna(row['Std_Dev']) and row['Std_Dev'] == 0 and row['Users'] > 1:
                st.warning(f"⚠️ Variation '{row['Variation']}' has a Standard Deviation of 0. All outcome values are identical. This impacts t-test interpretation.")

    if metric_col_name_display in summary_stats.columns:
        chart_data = summary_stats.set_index('Variation')[metric_col_name_display].fillna(0)
        if not chart_data.empty:
            st.bar_chart(chart_data, y=metric_col_name_display)

    st.markdown(f"##### 📈 Comparison vs. Control ('{control_name}')")
    control_data_rows = summary_stats[summary_stats['Variation'] == control_name]
    if control_data_rows.empty:
        st.error(f"Control group '{control_name}' data missing in this segment/dataset.")
        return summary_stats 

    control_data = control_data_rows.iloc[0]
    comparison_results_freq = []

    if metric_type == 'Binary':
        control_users, control_conversions = control_data['Users'], control_data['Conversions']
        control_metric_val = control_conversions / control_users if control_users > 0 else 0
        for index, row in summary_stats.iterrows():
            var_name, var_users, var_conversions = row['Variation'], row['Users'], row['Conversions']
            if var_name == control_name: continue
            var_metric_val = var_conversions / var_users if var_users > 0 else 0
            p_val_disp, ci_disp, sig_disp, abs_disp, rel_disp = 'N/A', 'N/A', 'N/A', 'N/A', 'N/A'
            if control_users > 0 and var_users > 0:
                abs_uplift = var_metric_val - control_metric_val; abs_disp = f"{abs_uplift*100:.2f}"
                rel_disp = f"{(abs_uplift / control_metric_val) * 100:.2f}%" if control_metric_val != 0 else "N/A (Control CR is 0)"
                count, nobs = np.array([var_conversions, control_conversions]), np.array([var_users, control_users])
                if not (np.any(count < 0) or np.any(nobs <= 0) or np.any(count > nobs)):
                    try:
                        _, p_value = proportions_ztest(count, nobs, alternative='two-sided')
                        p_val_disp = f"{p_value:.4f}"
                        sig_bool = p_value < alpha; sig_disp = f"Yes (p={p_value:.4f})" if sig_bool else f"No (p={p_value:.4f})"
                        ci_low, ci_high = confint_proportions_2indep(var_conversions, var_users, control_conversions, control_users, method='wald', alpha=alpha)
                        ci_disp = f"[{ci_low*100:.2f}, {ci_high*100:.2f}]"
                    except Exception as e_prop_z: 
                        p_val_disp, ci_disp, sig_disp = 'Error', 'Error', 'Error'; st.caption(f"Z-test error for {var_name}: {e_prop_z}")
                else: sig_disp = 'N/A (Invalid counts/nobs for Z-test)'
            else: sig_disp = 'N/A (Zero users in control or variation)'
            comparison_results_freq.append({"Variation": var_name, "CR (%)": f"{var_metric_val*100:.2f}", "Abs. Uplift (%)": abs_disp, "Rel. Uplift (%)": rel_disp, "P-value": p_val_disp, f"CI {100*(1-alpha):.0f}% Diff (%)": ci_disp, "Significant?": sig_disp})
    
    elif metric_type == 'Continuous':
        control_mean = control_data['Mean_Value']
        control_group_data_raw = df_cleaned[df_cleaned[variation_col] == control_name][outcome_col].dropna() 
        control_users_raw = len(control_group_data_raw)

        for index, row in summary_stats.iterrows():
            var_name, var_mean = row['Variation'], row['Mean_Value']
            if var_name == control_name: continue
            
            var_group_data_raw = df_cleaned[df_cleaned[variation_col] == var_name][outcome_col].dropna() 
            var_users_raw = len(var_group_data_raw)
            p_val_disp, ci_disp, sig_disp, abs_disp, rel_disp = 'N/A', 'N/A', 'N/A', 'N/A', 'N/A'
            
            if control_users_raw > 1 and var_users_raw > 1:
                abs_diff_means = var_mean - control_mean if pd.notna(var_mean) and pd.notna(control_mean) else np.nan
                abs_disp = f"{abs_diff_means:.3f}" if pd.notna(abs_diff_means) else "N/A"
                rel_disp = f"{(abs_diff_means / control_mean) * 100:.2f}%" if pd.notna(abs_diff_means) and control_mean != 0 and pd.notna(control_mean) else "N/A"
                
                control_var_raw = control_group_data_raw.var(ddof=1)
                var_var_raw = var_group_data_raw.var(ddof=1)

                if pd.notna(control_var_raw) and pd.notna(var_var_raw) and control_var_raw == 0 and var_var_raw == 0:
                    if control_mean == var_mean: p_value = 1.0
                    else: p_value = 0.0 
                    p_val_disp = f"{p_value:.4f}"
                    sig_bool = p_value < alpha; sig_disp = f"Yes (p={p_value:.4f})" if sig_bool else f"No (p={p_value:.4f})"
                    ci_disp = f"[{abs_diff_means:.3f}, {abs_diff_means:.3f}] (Exact)" if pd.notna(abs_diff_means) else "N/A"
                else:
                    try:
                        t_stat, p_value = ttest_ind(var_group_data_raw, control_group_data_raw, equal_var=False, nan_policy='omit')
                        p_val_disp = f"{p_value:.4f}" if pd.notna(p_value) else "N/A (t-test error)"
                        if pd.notna(p_value):
                            sig_bool = p_value < alpha; sig_disp = f"Yes (p={p_value:.4f})" if sig_bool else f"No (p={p_value:.4f})"
                        
                        N1, N2 = var_users_raw, control_users_raw
                        s1_sq, s2_sq = var_var_raw, control_var_raw
                        
                        if N1 > 0 and N2 > 0 and pd.notna(s1_sq) and pd.notna(s2_sq) and pd.notna(abs_diff_means):
                            pooled_se_diff_sq = (s1_sq / N1) + (s2_sq / N2) if N1 > 0 and N2 > 0 else 0 
                            if pooled_se_diff_sq > 0 : 
                                pooled_se_diff = math.sqrt(pooled_se_diff_sq)
                                df_t_approx = min(N1 - 1, N2 - 1)
                                if df_t_approx > 0:
                                    t_crit = t_dist.ppf(1 - alpha / 2, df=df_t_approx)
                                    ci_low_mean_diff, ci_high_mean_diff = abs_diff_means - t_crit * pooled_se_diff, abs_diff_means + t_crit * pooled_se_diff
                                    ci_disp = f"[{ci_low_mean_diff:.3f}, {ci_high_mean_diff:.3f}]"
                                else: ci_disp = "N/A (df CI <=0)"
                            else: ci_disp = "N/A (SE diff is 0 or invalid)" 
                        else: ci_disp = "N/A (N or var issue for CI)"
                    except Exception as e_ttest: 
                        p_val_disp, ci_disp, sig_disp = 'Error', 'Error', 'Error'; st.caption(f"T-test error for {var_name}: {e_ttest}")
            else: sig_disp = 'N/A (N <= 1 in a group)'
            comparison_results_freq.append({"Variation": var_name, "Mean Value": f"{var_mean:.3f}" if pd.notna(var_mean) else "N/A", "Abs. Diff.": abs_disp, "Rel. Diff. (%)": rel_disp, "P-value": p_val_disp, f"CI {100*(1-alpha):.0f}% Diff.": ci_disp, "Significant?": sig_disp})

    if comparison_results_freq:
        comparison_df_freq = pd.DataFrame(comparison_results_freq)
        st.dataframe(comparison_df_freq)
        for _, row_data in comparison_df_freq.iterrows():
            if "Yes" in str(row_data["Significant?"]): st.caption(f"Frequentist: Diff between **{row_data['Variation']}** & control is significant at {alpha*100:.0f}% level (P-value: {row_data['P-value']}).")
            elif "No" in str(row_data["Significant?"]): st.caption(f"Frequentist: Diff between **{row_data['Variation']}** & control is not significant at {alpha*100:.0f}% level (P-value: {row_data['P-value']}).")
    
    if metric_type == 'Continuous': 
        st.markdown("##### Distribution of Outcomes by Variation (Box Plots)")
        try:
            if not df_cleaned.empty and variation_col in df_cleaned.columns: 
                unique_vars_plot = sorted(df_cleaned[variation_col].astype(str).unique()) 
                if control_name in unique_vars_plot: 
                    unique_vars_plot.insert(0, unique_vars_plot.pop(unique_vars_plot.index(control_name)))

                boxplot_data = []
                valid_labels = []
                for var_name_plot in unique_vars_plot:
                    data_series = df_cleaned[df_cleaned[variation_col].astype(str) == var_name_plot][outcome_col] 
                    if not data_series.empty and data_series.notna().any(): 
                        boxplot_data.append(data_series.dropna()) 
                        valid_labels.append(var_name_plot)
                
                if not boxplot_data or not valid_labels: 
                    st.caption(f"Not enough valid data for one or more variations to display box plots {summary_stats_key_suffix}.")
                else:
                    fig_box, ax_box = plt.subplots(); 
                    ax_box.boxplot(boxplot_data, labels=valid_labels, patch_artist=True) 
                    ax_box.set_title(f"Outcome Distributions: {outcome_col} by Variation"); 
                    ax_box.set_ylabel(outcome_col); 
                    ax_box.set_xlabel("Variation")
                    plt.xticks(rotation=45, ha="right") 
                    plt.tight_layout()
                    st.pyplot(fig_box); plt.close(fig_box)
            else: st.caption(f"Not enough data or columns missing to display box plots {summary_stats_key_suffix}.")
        except Exception as e_plot: st.warning(f"Could not generate box plots {summary_stats_key_suffix}: {e_plot}")
    return summary_stats

# --- Helper function for Bayesian Binary Display ---
def display_bayesian_binary_results(bayesian_results, summary_stats_for_ordering, control_name, alpha, section_title_prefix="Overall"):
    if not bayesian_results:
        st.info(f"{section_title_prefix} Bayesian analysis results for binary outcomes are not available or could not be computed.")
        return

    st.markdown(f"---"); st.subheader(f"{section_title_prefix} Bayesian Analysis Results (Binary Outcome)")
    st.markdown(f"Using a Beta(1,1) uninformative prior. Credible Intervals (CrI) at {100*(1-alpha):.0f}% level.")
    
    bayesian_data_disp_bin = []
    ordered_vars_for_display = summary_stats_for_ordering['Variation'].tolist() if summary_stats_for_ordering is not None and 'Variation' in summary_stats_for_ordering.columns else list(bayesian_results.keys())


    for var_name in ordered_vars_for_display:
        if var_name not in bayesian_results: continue 
        b_res = bayesian_results[var_name]
        prob_better_html = f"<span title=\"Probability that this variation's true conversion rate is higher than the control's. Also consider the Credible Interval for Uplift to understand magnitude and uncertainty.\">{b_res.get('prob_better_than_control',np.nan)*100:.2f}%</span>" if pd.notna(b_res.get('prob_better_than_control')) else ("N/A (Control)" if var_name == control_name else "N/A")
        cri_uplift_html = f"<span title=\"The range where the true uplift over control likely lies. If this interval includes 0, 'no difference' or a negative effect are plausible.\">[{b_res.get('uplift_ci_low', np.nan)*100:.2f}, {b_res.get('uplift_ci_high', np.nan)*100:.2f}]</span>" if pd.notna(b_res.get('uplift_ci_low')) else ("N/A (Control)" if var_name == control_name else "N/A")
        
        mean_cr_disp = f"{b_res.get('mean_cr',np.nan)*100:.2f}" if pd.notna(b_res.get('mean_cr')) else "N/A"
        cr_ci_low_disp = f"{b_res.get('cr_ci_low',np.nan)*100:.2f}" if pd.notna(b_res.get('cr_ci_low')) else "N/A"
        cr_ci_high_disp = f"{b_res.get('cr_ci_high',np.nan)*100:.2f}" if pd.notna(b_res.get('cr_ci_high')) else "N/A"
        exp_uplift_disp = f"{b_res.get('expected_uplift_abs', np.nan)*100:.2f}" if pd.notna(b_res.get('expected_uplift_abs')) else ("N/A (Control)" if var_name == control_name else "N/A")
        prob_best_disp = f"{b_res.get('prob_best',np.nan)*100:.2f}" if pd.notna(b_res.get('prob_best')) else "N/A"

        bayesian_data_disp_bin.append({
            "Variation": var_name, "Posterior Mean CR (%)": mean_cr_disp, 
            f"{100*(1-alpha):.0f}% CrI for CR (%)": f"[{cr_ci_low_disp}, {cr_ci_high_disp}]", 
            "P(Better > Control) (%)": prob_better_html, 
            "Expected Uplift (abs %)": exp_uplift_disp, 
            f"{100*(1-alpha):.0f}% CrI for Uplift (abs %)": cri_uplift_html, 
            "P(Being Best) (%)": prob_best_disp
        })
    if bayesian_data_disp_bin:
        bayesian_df_bin = pd.DataFrame(bayesian_data_disp_bin); st.markdown(bayesian_df_bin.to_html(escape=False, index=False), unsafe_allow_html=True)
    
    if summary_stats_for_ordering is not None and 'Metric Value (%)' in summary_stats_for_ordering.columns:
        st.markdown("##### Posterior Distributions for Conversion Rates (Binary)"); fig_cr_bin, ax_cr_bin = plt.subplots()
        observed_max_cr_for_plot = 0.0
        numeric_crs_bin = pd.to_numeric(summary_stats_for_ordering['Metric Value (%)'], errors='coerce')
        if not numeric_crs_bin.empty and numeric_crs_bin.notna().any(): observed_max_cr_for_plot = numeric_crs_bin.max() / 100.0
        else: observed_max_cr_for_plot = 0.1 
        posterior_max_cr_for_plot_bin = 0.0
        all_posterior_highs_bin = [res.get('cr_ci_high') for res in bayesian_results.values() if res.get('cr_ci_high') is not None and pd.notna(res.get('cr_ci_high'))]
        if all_posterior_highs_bin: posterior_max_cr_for_plot_bin = max(all_posterior_highs_bin)
        final_x_limit_candidate_bin = max(observed_max_cr_for_plot, posterior_max_cr_for_plot_bin)
        x_cr_plot_limit_bin = min(1.0, final_x_limit_candidate_bin + 0.05) 
        if x_cr_plot_limit_bin <= 0.01: x_cr_plot_limit_bin = 0.1 
        x_cr_range_bin = np.linspace(0, x_cr_plot_limit_bin, 300)
        max_density_bin = 0
        for var_name in ordered_vars_for_display: 
            if var_name not in bayesian_results: continue
            b_res = bayesian_results[var_name]
            alpha_p, beta_p = b_res.get('alpha_post', 1), b_res.get('beta_post', 1)
            if alpha_p > 0 and beta_p > 0 and not (np.isnan(alpha_p) or np.isnan(beta_p)): 
                posterior_pdf = beta_dist.pdf(x_cr_range_bin, alpha_p, beta_p)
                ax_cr_bin.plot(x_cr_range_bin, posterior_pdf, label=f"{var_name} (α={alpha_p:.1f},β={beta_p:.1f})")
                ax_cr_bin.fill_between(x_cr_range_bin, posterior_pdf, alpha=0.2)
                if posterior_pdf is not None and np.any(np.isfinite(posterior_pdf)): 
                    finite_pdf = posterior_pdf[np.isfinite(posterior_pdf)]
                    if finite_pdf.size > 0: max_density_bin = max(max_density_bin, np.nanmax(finite_pdf))
        if max_density_bin > 0: ax_cr_bin.set_ylim(0, max_density_bin * 1.1)
        else: ax_cr_bin.set_ylim(0,1)
        ax_cr_bin.set_title("Posterior Distributions of CRs"); ax_cr_bin.set_xlabel("Conversion Rate"); ax_cr_bin.set_ylabel("Density"); ax_cr_bin.legend(); st.pyplot(fig_cr_bin); plt.close(fig_cr_bin)

        st.markdown("##### Posterior Distribution of Uplift (Variation CR - Control CR)")
        num_vars_to_plot_bin = sum(1 for var_name_uplift in bayesian_results if var_name_uplift != control_name and bayesian_results[var_name_uplift].get('diff_samples_vs_control') is not None and not np.all(np.isnan(bayesian_results[var_name_uplift].get('diff_samples_vs_control'))))
        if num_vars_to_plot_bin > 0:
            cols_diff_plots_bin = st.columns(min(num_vars_to_plot_bin, 3)); col_idx_bin = 0
            for var_name in ordered_vars_for_display: 
                if var_name == control_name or var_name not in bayesian_results: continue
                b_res = bayesian_results[var_name]
                diff_samples = b_res.get('diff_samples_vs_control')
                if diff_samples is None or np.all(np.isnan(diff_samples)): continue
                with cols_diff_plots_bin[col_idx_bin % min(num_vars_to_plot_bin, 3)]:
                    fig_diff_bin, ax_diff_bin = plt.subplots(); 
                    ax_diff_bin.hist(diff_samples[~np.isnan(diff_samples)], bins=50, density=True, alpha=0.6, label=f"{var_name} - {control_name}")
                    ax_diff_bin.axvline(0, color='grey', linestyle='--'); 
                    ax_diff_bin.axvline(b_res.get('expected_uplift_abs',0), color='red', linestyle=':', label=f"Mean Diff: {b_res.get('expected_uplift_abs',0)*100:.2f}%")
                    ax_diff_bin.set_title(f"Uplift: {var_name} vs {control_name}"); ax_diff_bin.set_xlabel("Difference in CR"); ax_diff_bin.set_ylabel("Density"); 
                    ax_diff_bin.legend(); st.pyplot(fig_diff_bin); plt.close(fig_diff_bin)
                    col_idx_bin +=1
    st.markdown("""**Interpreting Bayesian Results (Binary - Briefly):** (Full guidance in 'Bayesian Analysis Guidelines' section)
    - **Posterior Mean CR:** Average CR after data. - **CrI for CR:** Range for true CR. - **P(Better > Control):** Probability variation's true CR is higher. - **Expected Uplift:** Average expected improvement. - **CrI for Uplift:** Range for true uplift. If includes 0, 'no difference' is plausible. - **P(Being Best):** Probability variation has highest true CR.""")

# --- Helper function for Bayesian Continuous Display ---
def display_bayesian_continuous_results(bayesian_results, summary_stats_for_ordering, control_name, alpha, outcome_col_name, section_title_prefix="Overall"):
    if not bayesian_results:
        st.info(f"{section_title_prefix} Bayesian analysis results for continuous outcomes are not available or could not be computed.")
        return

    st.markdown(f"---"); st.subheader(f"{section_title_prefix} Bayesian Analysis Results (Continuous Outcome)")
    st.markdown(f"Using a t-distribution approximation for posteriors. Credible Intervals (CrI) at {100*(1-alpha):.0f}% level.")
    
    bayesian_data_disp_cont = []
    ordered_vars_for_display = summary_stats_for_ordering['Variation'].tolist() if summary_stats_for_ordering is not None and 'Variation' in summary_stats_for_ordering.columns else list(bayesian_results.keys())


    for var_name in ordered_vars_for_display:
        if var_name not in bayesian_results: continue
        b_res_cont = bayesian_results[var_name]
        
        tt_posterior_mean = "The average value of the metric for this variation, based on its posterior distribution (typically very close to the sample mean)."
        tt_cri_mean = f"The {100*(1-alpha):.0f}% Credible Interval for the true mean of this variation. We are {100*(1-alpha):.0f}% confident the true mean lies here."
        tt_prob_better = "Probability that this variation's true mean is greater than the control's true mean. Consider with the CrI for Difference."
        tt_exp_diff = "The average expected absolute difference (Variation Mean - Control Mean)."
        tt_cri_diff = f"The {100*(1-alpha):.0f}% Credible Interval for the true absolute difference in means. If it includes 0, 'no difference' is plausible."
        tt_prob_best = "Probability that this variation has the highest true mean among all tested variations."

        post_mean_disp = f"<span title='{tt_posterior_mean}'>{b_res_cont.get('posterior_mean', np.nan):.3f}</span>" if pd.notna(b_res_cont.get('posterior_mean')) else "N/A"
        mean_ci_disp = f"<span title='{tt_cri_mean}'>[{b_res_cont.get('mean_ci_low', np.nan):.3f}, {b_res_cont.get('mean_ci_high', np.nan):.3f}]</span>" if pd.notna(b_res_cont.get('mean_ci_low')) else "N/A"
        prob_better_disp = f"<span title='{tt_prob_better}'>{b_res_cont.get('prob_better_than_control', np.nan)*100:.2f}%</span>" if pd.notna(b_res_cont.get('prob_better_than_control')) else ("N/A (Control)" if var_name == control_name else "N/A")
        exp_diff_disp = f"<span title='{tt_exp_diff}'>{b_res_cont.get('expected_diff_abs', np.nan):.3f}</span>" if pd.notna(b_res_cont.get('expected_diff_abs')) else ("N/A (Control)" if var_name == control_name else "N/A")
        diff_ci_disp = f"<span title='{tt_cri_diff}'>[{b_res_cont.get('diff_ci_low', np.nan):.3f}, {b_res_cont.get('diff_ci_high', np.nan):.3f}]</span>" if pd.notna(b_res_cont.get('diff_ci_low')) else ("N/A (Control)" if var_name == control_name else "N/A")
        prob_best_disp = f"<span title='{tt_prob_best}'>{b_res_cont.get('prob_best', np.nan)*100:.2f}%</span>" if pd.notna(b_res_cont.get('prob_best')) else "N/A"
        
        bayesian_data_disp_cont.append({
            "Variation": var_name, "Posterior Mean": post_mean_disp,
            f"{100*(1-alpha):.0f}% CrI for Mean": mean_ci_disp,
            "P(Better > Control) (%)": prob_better_disp,
            "Expected Diff. (abs)": exp_diff_disp,
            f"{100*(1-alpha):.0f}% CrI for Diff. (abs)": diff_ci_disp,
            "P(Being Best) (%)": prob_best_disp
        })
    if bayesian_data_disp_cont:
        bayesian_df_cont = pd.DataFrame(bayesian_data_disp_cont)
        st.markdown(bayesian_df_cont.to_html(escape=False, index=False), unsafe_allow_html=True) 

        st.markdown("##### Key Bayesian Insights (Continuous):")
        for item in bayesian_data_disp_cont:
            var_name_bayes = item["Variation"]
            if var_name_bayes == control_name: continue 
            prob_better_val_str = item["P(Better > Control) (%)"] 
            prob_better_numeric = np.nan
            if "N/A" not in prob_better_val_str:
                try: prob_better_numeric = float(prob_better_val_str.split('>')[1].split('%')[0]) / 100
                except: pass 
            cri_diff_str = item[f"{100*(1-alpha):.0f}% CrI for Diff. (abs)"] 
            cri_diff_contains_zero = "N/A" not in cri_diff_str and "[" in cri_diff_str and "]" in cri_diff_str and \
                                        float(cri_diff_str.split('[')[1].split(',')[0]) < 0 < float(cri_diff_str.split(',')[1].split(']')[0])
            expected_diff_val_str = item["Expected Diff. (abs)"]
            expected_diff_numeric = np.nan
            if "N/A" not in expected_diff_val_str:
                try: expected_diff_numeric = float(expected_diff_val_str.split('>')[1].split('<')[0])
                except: pass
            
            if pd.notna(prob_better_numeric):
                insight = f"For **{var_name_bayes}** vs Control:"
                if prob_better_numeric > (1 - alpha): 
                    insight += f" Strong evidence it's better (P(Better) = {prob_better_numeric*100:.1f}%)."
                elif prob_better_numeric > 0.5:
                    insight += f" More likely better than not (P(Better) = {prob_better_numeric*100:.1f}%)."
                else:
                    insight += f" Less likely to be better (P(Better) = {prob_better_numeric*100:.1f}%)."
                
                if pd.notna(expected_diff_numeric):
                    insight += f" Expected difference is {expected_diff_numeric:.3f}."
                if "N/A" not in cri_diff_str:
                    insight += f" The {100*(1-alpha):.0f}% CrI for difference is {cri_diff_str.replace('<span>','').replace('</span>','')}" 
                    if cri_diff_contains_zero:
                        insight += ", which includes zero (suggesting the difference may not be practically significant or could be due to chance)."
                    else:
                        insight += "."
                st.caption(insight)
    
    if summary_stats_for_ordering is not None and 'Mean_Value' in summary_stats_for_ordering.columns: 
        st.markdown("##### Posterior Distributions for Means (Continuous)")
        fig_mean_cont, ax_mean_cont = plt.subplots()
        max_density_mean_cont = 0
        all_mean_samples_for_plot = []
        for var_name_plot in ordered_vars_for_display:
            if var_name_plot in bayesian_results:
                samples = bayesian_results[var_name_plot].get('samples')
                if samples is not None and samples.size > 0 and not np.all(np.isnan(samples)):
                    all_mean_samples_for_plot.extend(samples[~np.isnan(samples)])
        
        if all_mean_samples_for_plot: 
            x_min_mean_cont, x_max_mean_cont = np.min(all_mean_samples_for_plot), np.max(all_mean_samples_for_plot)
            padding_mean_cont = (x_max_mean_cont - x_min_mean_cont) * 0.1 if (x_max_mean_cont - x_min_mean_cont) > 0 else 1.0
            x_range_mean_cont = np.linspace(x_min_mean_cont - padding_mean_cont, x_max_mean_cont + padding_mean_cont, 300)

            for var_name in ordered_vars_for_display:
                if var_name not in bayesian_results: continue
                b_res_cont = bayesian_results[var_name]
                samples = b_res_cont.get('samples')
                if samples is not None and samples.size > 0 and not np.all(np.isnan(samples)):
                    valid_samples = samples[~np.isnan(samples)]
                    if valid_samples.size > 1: 
                        kde = gaussian_kde(valid_samples)
                        pdf_values = kde(x_range_mean_cont)
                        ax_mean_cont.plot(x_range_mean_cont, pdf_values, label=f"{var_name}")
                        ax_mean_cont.fill_between(x_range_mean_cont, pdf_values, alpha=0.2)
                        max_density_mean_cont = max(max_density_mean_cont, np.nanmax(pdf_values))
                    elif valid_samples.size == 1: 
                        ax_mean_cont.axvline(valid_samples[0], label=f"{var_name} (single data point)", linestyle="--", alpha=0.7)
            if max_density_mean_cont > 0: ax_mean_cont.set_ylim(0, max_density_mean_cont * 1.1)
            elif not all_mean_samples_for_plot: pass 
            else: ax_mean_cont.set_ylim(0,1) 
            ax_mean_cont.set_title("Posterior Distributions of Means"); ax_mean_cont.set_xlabel(f"Mean of {outcome_col_name}"); 
            ax_mean_cont.set_ylabel("Density"); ax_mean_cont.legend(); st.pyplot(fig_mean_cont); plt.close(fig_mean_cont)
        else:
            st.caption(f"Not enough valid sample data to plot posterior distributions of means for {section_title_prefix.lower()} results.")

        st.markdown("##### Posterior Distribution of Difference (Variation Mean - Control Mean)")
        num_vars_to_plot_cont_diff = sum(1 for var_name_diff in bayesian_results if var_name_diff != control_name and bayesian_results[var_name_diff].get('diff_samples_vs_control') is not None and not np.all(np.isnan(bayesian_results[var_name_diff].get('diff_samples_vs_control'))))
        if num_vars_to_plot_cont_diff > 0:
            cols_diff_plots_cont = st.columns(min(num_vars_to_plot_cont_diff, 2)) 
            col_idx_cont_diff = 0
            for var_name in ordered_vars_for_display:
                if var_name == control_name or var_name not in bayesian_results: continue
                b_res_cont = bayesian_results[var_name]
                diff_samples = b_res_cont.get('diff_samples_vs_control')
                if diff_samples is None or diff_samples.size == 0 or np.all(np.isnan(diff_samples)): continue
                
                with cols_diff_plots_cont[col_idx_cont_diff % min(num_vars_to_plot_cont_diff, 2)]:
                    fig_diff_cont, ax_diff_cont = plt.subplots(); 
                    valid_diff_samples_plot = diff_samples[~np.isnan(diff_samples)]
                    if valid_diff_samples_plot.size > 0:
                        ax_diff_cont.hist(valid_diff_samples_plot, bins=50, density=True, alpha=0.6, label=f"{var_name} - {control_name}")
                        ax_diff_cont.axvline(0, color='grey', linestyle='--'); 
                        expected_diff_val = b_res_cont.get('expected_diff_abs', np.nan)
                        if pd.notna(expected_diff_val):
                            ax_diff_cont.axvline(expected_diff_val, color='red', linestyle=':', label=f"Mean Diff: {expected_diff_val:.3f}")
                        ax_diff_cont.set_title(f"Diff: {var_name} vs {control_name}"); 
                        ax_diff_cont.set_xlabel(f"Difference in Mean of {outcome_col_name}"); ax_diff_cont.set_ylabel("Density"); 
                        ax_diff_cont.legend(); st.pyplot(fig_diff_cont); plt.close(fig_diff_cont)
                    else:
                        st.caption(f"Not enough valid difference data to plot for {var_name} vs {control_name} for {section_title_prefix.lower()} results.")
                    col_idx_cont_diff +=1
    st.markdown("""**Interpreting Bayesian Results (Continuous - Briefly):** (Full guidance in 'Bayesian Analysis Guidelines' section)
    - **Posterior Mean:** Average value of the metric after data. - **CrI for Mean:** Range for true mean. - **P(Better > Control):** Probability variation's true mean is higher. - **Expected Difference:** Average expected difference from control. - **CrI for Difference:** Range for true difference. If includes 0, 'no difference' is plausible. - **P(Being Best):** Probability variation has highest true mean.""")


def show_analyze_results_page():
    # Page for analyzing A/B test results.
    st.header("Analyze Your A/B Test Results 📊")
    st.markdown("Upload your A/B test data (as a CSV file) to perform an analysis.")
    st.markdown("---")

    cycle_suffix = "_c8_1" 
    # Initialize session state variables
    if f'analysis_done{cycle_suffix}' not in st.session_state: st.session_state[f'analysis_done{cycle_suffix}'] = False
    if f'df_analysis{cycle_suffix}' not in st.session_state: st.session_state[f'df_analysis{cycle_suffix}'] = None
    if f'metric_type_analysis{cycle_suffix}' not in st.session_state: st.session_state[f'metric_type_analysis{cycle_suffix}'] = 'Binary'
    if f'variation_col_analysis{cycle_suffix}' not in st.session_state: st.session_state[f'variation_col_analysis{cycle_suffix}'] = None
    if f'outcome_col_analysis{cycle_suffix}' not in st.session_state: st.session_state[f'outcome_col_analysis{cycle_suffix}'] = None
    if f'success_value_analysis{cycle_suffix}' not in st.session_state: st.session_state[f'success_value_analysis{cycle_suffix}'] = None
    if f'control_group_name_analysis{cycle_suffix}' not in st.session_state: st.session_state[f'control_group_name_analysis{cycle_suffix}'] = None
    if f'alpha_for_analysis{cycle_suffix}' not in st.session_state: st.session_state[f'alpha_for_analysis{cycle_suffix}'] = 0.05
    if f'overall_freq_summary_stats{cycle_suffix}' not in st.session_state: st.session_state[f'overall_freq_summary_stats{cycle_suffix}'] = None
    if f'overall_bayesian_results_binary{cycle_suffix}' not in st.session_state: st.session_state[f'overall_bayesian_results_binary{cycle_suffix}'] = None
    if f'overall_bayesian_results_continuous{cycle_suffix}' not in st.session_state: st.session_state[f'overall_bayesian_results_continuous{cycle_suffix}'] = None 
    if f'metric_col_name{cycle_suffix}' not in st.session_state: st.session_state[f'metric_col_name{cycle_suffix}'] = None
    if f'segmentation_cols{cycle_suffix}' not in st.session_state: st.session_state[f'segmentation_cols{cycle_suffix}'] = [] 
    if f'segmented_freq_results{cycle_suffix}' not in st.session_state: st.session_state[f'segmented_freq_results{cycle_suffix}'] = {}
    if f'segmented_bayesian_results{cycle_suffix}' not in st.session_state: st.session_state[f'segmented_bayesian_results{cycle_suffix}'] = {} 

    uploaded_file = st.file_uploader("Upload your CSV data file", type=["csv"], key=f"file_uploader{cycle_suffix}", help="Upload a CSV file with your A/B test data. Ensure columns are clearly named.")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            df.columns = df.columns.str.strip()
            st.session_state[f'df_analysis{cycle_suffix}'] = df 
            st.success("File Uploaded Successfully!")
            with st.expander("Show Data Preview (first 5 rows)", expanded=False):
                 st.dataframe(df.head())
            st.markdown("---")

            st.subheader("1. Map Data & Select Metric Type")
            st.markdown("Identify the columns in your data that correspond to variations, outcomes, and (optionally) user segments.")
            columns = df.columns.tolist() 
            
            map_col1, map_col2, map_col3 = st.columns(3)
            with map_col1: 
                var_col_index = 0 
                if columns and 'variation_name' in columns: var_col_index = columns.index('variation_name')
                elif columns and 'variation' in columns: var_col_index = columns.index('variation')
                st.session_state[f'variation_col_analysis{cycle_suffix}'] = st.selectbox("Select 'Variation ID' column:", options=columns, index=var_col_index if columns else -1, key=f"var_col{cycle_suffix}", help="Column indicating which variation each user was exposed to (e.g., 'Control', 'TreatmentA').")
            
            with map_col2: 
                out_col_index = len(columns)-1 if len(columns)>1 else (0 if columns else -1) 
                if columns and 'converted' in columns: out_col_index = columns.index('converted')
                elif columns and 'outcome' in columns: out_col_index = columns.index('outcome')
                elif columns and 'metric' in columns: out_col_index = columns.index('metric')
                st.session_state[f'outcome_col_analysis{cycle_suffix}'] = st.selectbox("Select 'Outcome' column:", options=columns, index=out_col_index, key=f"out_col{cycle_suffix}", help="Column containing the result you measured (e.g., 0/1 for conversion, or a numeric value like revenue).")
            
            with map_col3: 
                st.session_state[f'metric_type_analysis{cycle_suffix}'] = st.radio(
                    "Select Metric Type for Outcome Column:", ('Binary', 'Continuous'), 
                    key=f"metric_type_analysis_radio{cycle_suffix}", horizontal=True,
                    help="Is your outcome a yes/no (Binary) or a numerical value (Continuous)?"
                )

            if st.session_state[f'metric_type_analysis{cycle_suffix}'] == 'Binary':
                outcome_col = st.session_state[f'outcome_col_analysis{cycle_suffix}']
                if outcome_col and outcome_col in df.columns:
                    unique_outcomes = df[outcome_col].unique()
                    if len(unique_outcomes) == 1: st.warning(f"Outcome column '{outcome_col}' has only one value: `{unique_outcomes[0]}`. A/B testing requires at least two different outcome values to compare performance.")
                    elif len(unique_outcomes) > 2 and len(unique_outcomes) <=10 : st.warning(f"Outcome column '{outcome_col}' has >2 unique values: `{unique_outcomes}`. For binary analysis, please select the value representing 'Conversion' or 'Success'. Other values will be treated as non-conversion.")
                    elif len(unique_outcomes) > 10: st.warning(f"Outcome column '{outcome_col}' has many unique values ({len(unique_outcomes)}). Ensure this is truly a binary outcome and select the correct success value. If it's a continuous metric, please change the 'Metric Type' selection above.")

                    if len(unique_outcomes) > 0:
                        str_options = sorted([str(val) for val in unique_outcomes]) 
                        default_success_idx = 0
                        common_success_indicators = ['1', 'True', 'true', 'yes', 'Yes', 'Success', 'Converted']
                        for indicator in common_success_indicators:
                            if indicator in str_options: 
                                default_success_idx = str_options.index(indicator)
                                break
                        
                        success_value_str = st.selectbox(f"Which value in '{outcome_col}' represents 'Conversion' (Success)?", options=str_options, index=default_success_idx, key=f"succ_val{cycle_suffix}", help="Select the value that indicates a successful outcome.")
                        
                        original_dtype = df[outcome_col].dtype
                        if success_value_str.lower() == 'nan' and any(pd.isna(val) for val in unique_outcomes): st.session_state[f'success_value_analysis{cycle_suffix}'] = np.nan
                        elif pd.api.types.is_numeric_dtype(original_dtype) and not pd.api.types.is_bool_dtype(original_dtype):
                            try: st.session_state[f'success_value_analysis{cycle_suffix}'] = original_dtype.type(success_value_str)
                            except ValueError: st.session_state[f'success_value_analysis{cycle_suffix}'] = success_value_str 
                        elif pd.api.types.is_bool_dtype(original_dtype): st.session_state[f'success_value_analysis{cycle_suffix}'] = (success_value_str.lower() == 'true') 
                        else: st.session_state[f'success_value_analysis{cycle_suffix}'] = success_value_str
                    else: st.warning(f"Could not determine distinct values in outcome column '{outcome_col}'. Please check the column content.")
                elif outcome_col: st.warning(f"Selected outcome column '{outcome_col}' not found in the uploaded CSV. Please check column names.")
                else: st.warning("Please select a valid outcome column for binary analysis.")
            else: 
                st.session_state[f'success_value_analysis{cycle_suffix}'] = None 
                outcome_col = st.session_state[f'outcome_col_analysis{cycle_suffix}']
                if outcome_col and outcome_col in df.columns:
                    if not pd.api.types.is_numeric_dtype(df[outcome_col]):
                        try:
                            df[outcome_col] = pd.to_numeric(df[outcome_col])
                            st.session_state[f'df_analysis{cycle_suffix}'] = df 
                            st.info(f"Attempted to convert outcome column '{outcome_col}' to numeric. Please verify data preview if conversion was as expected.")
                        except ValueError:
                             st.error(f"For 'Continuous' metric type, the outcome column '{outcome_col}' must be numeric. Current type: {df[outcome_col].dtype}. Could not automatically convert to numeric. Please select a numeric column or clean your data.")
                elif outcome_col: st.warning(f"Selected outcome column '{outcome_col}' not found in the uploaded CSV.")
            
            st.markdown("---"); st.subheader("2. Select Your Control Group & Analysis Alpha")
            var_col_sel = st.session_state[f'variation_col_analysis{cycle_suffix}']
            if var_col_sel and var_col_sel in df.columns and st.session_state[f'df_analysis{cycle_suffix}'] is not None:
                variation_names = st.session_state[f'df_analysis{cycle_suffix}'][var_col_sel].astype(str).unique().tolist() 
                if variation_names: 
                    default_control_idx = 0
                    for i, v_name in enumerate(variation_names):
                        if v_name.lower() in ['control', 'baseline', 'a']:
                            default_control_idx = i
                            break
                    st.session_state[f'control_group_name_analysis{cycle_suffix}'] = st.selectbox("Select 'Control Group':", options=variation_names, index=default_control_idx, key=f"ctrl_grp{cycle_suffix}", help="Choose the variation that represents your baseline or existing version.")
                else: st.warning(f"No unique variations found in column '{var_col_sel}'. Please check your data.")
            elif var_col_sel: st.warning(f"Selected variation column '{var_col_sel}' not found in the uploaded CSV.")
            else: st.warning("Please select a valid variation column.")
            st.session_state[f'alpha_for_analysis{cycle_suffix}'] = st.slider("Significance Level (\u03B1) for Analysis (%)", 1, 20, 5, 1, key=f"alpha_analysis{cycle_suffix}_slider", help="Typically 5%. This is your tolerance for a Type I error (false positive) in Frequentist tests.") / 100.0
            
            st.markdown("---"); st.subheader("3. Optional: Segmentation Analysis")
            st.markdown("Select one or more columns to segment your results by. Analysis will be performed for the overall data and then for each segment.")
            
            potential_segment_cols = [col for col in columns if col not in [st.session_state[f'variation_col_analysis{cycle_suffix}'], st.session_state[f'outcome_col_analysis{cycle_suffix}']] ]
            good_segment_cols = []
            if df is not None:
                for col in potential_segment_cols:
                    if df[col].nunique(dropna=True) <= 20: 
                        good_segment_cols.append(col)
                    else:
                        st.caption(f"Column '{col}' has >20 unique values and is excluded from segmentation options for performance reasons.")

            if not good_segment_cols:
                st.info("No suitable columns found for segmentation (e.g., columns might have too many unique values, or only variation/outcome columns remain).")
                st.session_state[f'segmentation_cols{cycle_suffix}'] = [] 
            else:
                st.session_state[f'segmentation_cols{cycle_suffix}'] = st.multiselect(
                    "Select segmentation column(s):", options=good_segment_cols,
                    default=st.session_state.get(f'segmentation_cols{cycle_suffix}', []), 
                    help="Results will be broken down by unique combinations of values in these columns. Choose columns with a manageable number of unique values."
                )
            
            st.markdown("---") 
            analysis_button_label = f"🚀 Run Analysis ({st.session_state[f'metric_type_analysis{cycle_suffix}']} Outcome)"
            if st.button(analysis_button_label, key=f"run_analysis_button{cycle_suffix}", help="Click to perform the statistical analysis based on your selections."):
                st.session_state[f'analysis_done{cycle_suffix}'] = False
                st.session_state[f'overall_freq_summary_stats{cycle_suffix}'] = None
                st.session_state[f'overall_bayesian_results_binary{cycle_suffix}'] = None
                st.session_state[f'overall_bayesian_results_continuous{cycle_suffix}'] = None
                st.session_state[f'segmented_freq_results{cycle_suffix}'] = {} 
                st.session_state[f'segmented_bayesian_results{cycle_suffix}'] = {} 

                valid_setup = True
                var_col_val = st.session_state[f'variation_col_analysis{cycle_suffix}']
                out_col_val = st.session_state[f'outcome_col_analysis{cycle_suffix}']
                control_val = st.session_state[f'control_group_name_analysis{cycle_suffix}']
                df_val = st.session_state[f'df_analysis{cycle_suffix}']

                if not (var_col_val and var_col_val in df_val.columns and \
                        out_col_val and out_col_val in df_val.columns and \
                        control_val is not None):
                    st.error("Setup Error: Please complete all column mapping (Variation ID, Outcome) and control group selections with valid columns from the CSV."); valid_setup = False
                
                if st.session_state[f'metric_type_analysis{cycle_suffix}'] == 'Binary' and st.session_state[f'success_value_analysis{cycle_suffix}'] is None:
                    st.error("Setup Error: For Binary outcome, please specify the 'Conversion (Success)' value."); valid_setup = False
                
                if st.session_state[f'metric_type_analysis{cycle_suffix}'] == 'Continuous':
                    if not (out_col_val and out_col_val in df_val.columns): 
                        st.error("Setup Error: Please select a valid outcome column for continuous analysis."); valid_setup = False
                    elif not pd.api.types.is_numeric_dtype(df_val[out_col_val]):
                        st.error(f"Data Type Error: For 'Continuous' metric type, outcome column '{out_col_val}' must be numeric. Please check your data or column selection."); valid_setup = False
                
                if valid_setup:
                    try:
                        overall_df_for_analysis = df_val.copy() 
                        metric_type = st.session_state[f'metric_type_analysis{cycle_suffix}']
                        alpha = st.session_state[f'alpha_for_analysis{cycle_suffix}']
                        
                        # --- Overall Analysis Calculation ---
                        overall_summary_stats_calc = None 
                        if metric_type == 'Binary':
                            success_val_bin = st.session_state[f'success_value_analysis{cycle_suffix}']
                            if pd.isna(success_val_bin): 
                                overall_df_for_analysis['__outcome_processed__'] = overall_df_for_analysis[out_col_val].isna().astype(int)
                            else: 
                                overall_df_for_analysis['__outcome_processed__'] = (overall_df_for_analysis[out_col_val] == success_val_bin).astype(int)
                            st.session_state[f'metric_col_name{cycle_suffix}'] = 'Metric Value (%)'
                            
                            overall_summary_stats_calc = overall_df_for_analysis.groupby(var_col_val).agg(Users=('__outcome_processed__', 'count'),Conversions=('__outcome_processed__', 'sum')).reset_index()
                            overall_summary_stats_calc.rename(columns={var_col_val: 'Variation'}, inplace=True)
                            if not overall_summary_stats_calc.empty and overall_summary_stats_calc['Users'].sum() > 0:
                                overall_summary_stats_calc['Metric Value (%)'] = (overall_summary_stats_calc['Conversions'] / overall_summary_stats_calc['Users'].replace(0, np.nan) * 100).round(2)
                            else: overall_summary_stats_calc = None 
                        
                        elif metric_type == 'Continuous':
                            overall_df_for_analysis[out_col_val] = pd.to_numeric(overall_df_for_analysis[out_col_val], errors='coerce')
                            st.session_state[f'metric_col_name{cycle_suffix}'] = 'Mean_Value'
                            cleaned_overall_df_calc = overall_df_for_analysis.dropna(subset=[out_col_val])
                            if not cleaned_overall_df_calc.empty:
                                overall_summary_stats_calc = cleaned_overall_df_calc.groupby(var_col_val).agg(Users=(out_col_val, 'count'), Mean_Value=(out_col_val, 'mean'), Std_Dev=(out_col_val, 'std'), Median_Value=(out_col_val,'median'), Std_Err=(out_col_val, lambda x: x.std(ddof=1) / np.sqrt(x.count()) if x.count() > 0 and pd.notna(x.std(ddof=1)) else np.nan)).reset_index()
                                overall_summary_stats_calc.rename(columns={var_col_val: 'Variation'}, inplace=True)
                                if overall_summary_stats_calc['Users'].sum() == 0: overall_summary_stats_calc = None 
                                else:
                                    for col_to_round in ['Mean_Value', 'Std_Dev', 'Median_Value', 'Std_Err']:
                                         if col_to_round in overall_summary_stats_calc.columns: overall_summary_stats_calc[col_to_round] = overall_summary_stats_calc[col_to_round].round(3)
                            else: overall_summary_stats_calc = None 

                        st.session_state[f'overall_freq_summary_stats{cycle_suffix}'] = overall_summary_stats_calc

                        if overall_summary_stats_calc is not None:
                            if metric_type == 'Binary':
                                bayes_bin_res, bayes_bin_err = run_bayesian_binary_analysis(overall_summary_stats_calc, control_val, ci_level=(1-alpha))
                                if bayes_bin_err: st.error(f"Overall Bayesian Binary Analysis Error: {bayes_bin_err}")
                                st.session_state[f'overall_bayesian_results_binary{cycle_suffix}'] = bayes_bin_res
                            elif metric_type == 'Continuous':
                                bayes_cont_res, bayes_cont_err = run_bayesian_continuous_analysis(overall_summary_stats_calc, control_val, ci_level=(1-alpha))
                                if bayes_cont_err: st.error(f"Overall Bayesian Continuous Analysis Error: {bayes_cont_err}")
                                st.session_state[f'overall_bayesian_results_continuous{cycle_suffix}'] = bayes_cont_res
                        
                        st.session_state[f'analysis_done{cycle_suffix}'] = True 

                        # --- Segmentation Logic (Calculations) ---
                        selected_segment_cols = st.session_state.get(f'segmentation_cols{cycle_suffix}', [])
                        if selected_segment_cols:
                            segment_group_col = '__segment_group__'
                            missing_seg_cols = [sc for sc in selected_segment_cols if sc not in overall_df_for_analysis.columns]
                            if missing_seg_cols:
                                st.error(f"Selected segmentation column(s) not found: {', '.join(missing_seg_cols)}")
                            else:
                                overall_df_for_analysis[segment_group_col] = overall_df_for_analysis[selected_segment_cols].astype(str).agg(' | '.join, axis=1)
                                unique_segments = overall_df_for_analysis[segment_group_col].unique()
                                
                                if not unique_segments.size:
                                     st.info("No unique segments found based on selected columns for segmentation.")
                                else:
                                    st.info(f"Found {len(unique_segments)} unique segment(s). Performing analysis for each.")

                                for segment_value in unique_segments:
                                    segment_df = overall_df_for_analysis[overall_df_for_analysis[segment_group_col] == segment_value].copy()
                                    
                                    segment_summary_stats_calc = None 
                                    if metric_type == 'Binary':
                                        if '__outcome_processed__' in segment_df.columns: 
                                            segment_summary_stats_calc = segment_df.groupby(var_col_val).agg(Users=('__outcome_processed__', 'count'),Conversions=('__outcome_processed__', 'sum')).reset_index()
                                            segment_summary_stats_calc.rename(columns={var_col_val: 'Variation'}, inplace=True)
                                            if not segment_summary_stats_calc.empty and segment_summary_stats_calc['Users'].sum() > 0:
                                                segment_summary_stats_calc['Metric Value (%)'] = (segment_summary_stats_calc['Conversions'] / segment_summary_stats_calc['Users'].replace(0, np.nan) * 100).round(2)
                                            else: segment_summary_stats_calc = None
                                    elif metric_type == 'Continuous':
                                        segment_df[out_col_val] = pd.to_numeric(segment_df[out_col_val], errors='coerce') 
                                        cleaned_segment_df_calc = segment_df.dropna(subset=[out_col_val]) 
                                        if not cleaned_segment_df_calc.empty:
                                            segment_summary_stats_calc = cleaned_segment_df_calc.groupby(var_col_val).agg(Users=(out_col_val, 'count'), Mean_Value=(out_col_val, 'mean'), Std_Dev=(out_col_val, 'std'), Median_Value=(out_col_val,'median'), Std_Err=(out_col_val, lambda x: x.std(ddof=1) / np.sqrt(x.count()) if x.count() > 0 and pd.notna(x.std(ddof=1)) else np.nan)).reset_index()
                                            segment_summary_stats_calc.rename(columns={var_col_val: 'Variation'}, inplace=True)
                                            if segment_summary_stats_calc['Users'].sum() == 0: segment_summary_stats_calc = None
                                            else:
                                                for col_to_round_seg in ['Mean_Value', 'Std_Dev', 'Median_Value', 'Std_Err']:
                                                    if col_to_round_seg in segment_summary_stats_calc.columns: segment_summary_stats_calc[col_to_round_seg] = segment_summary_stats_calc[col_to_round_seg].round(3)
                                        else: segment_summary_stats_calc = None
                                    
                                    st.session_state[f'segmented_freq_results{cycle_suffix}'][segment_value] = {'data': segment_df, 'summary_stats': segment_summary_stats_calc}
                                    
                                    if segment_summary_stats_calc is not None and not segment_summary_stats_calc.empty:
                                        if metric_type == 'Binary':
                                            seg_bayes_bin_res, seg_bayes_bin_err = run_bayesian_binary_analysis(segment_summary_stats_calc, control_val, ci_level=(1-alpha))
                                            if seg_bayes_bin_err: st.error(f"Segment '{segment_value}' Bayesian Binary Analysis Error: {seg_bayes_bin_err}")
                                            st.session_state[f'segmented_bayesian_results{cycle_suffix}'][segment_value] = seg_bayes_bin_res
                                        elif metric_type == 'Continuous':
                                            seg_bayes_cont_res, seg_bayes_cont_err = run_bayesian_continuous_analysis(segment_summary_stats_calc, control_val, ci_level=(1-alpha))
                                            if seg_bayes_cont_err: st.error(f"Segment '{segment_value}' Bayesian Continuous Analysis Error: {seg_bayes_cont_err}")
                                            st.session_state[f'segmented_bayesian_results{cycle_suffix}'][segment_value] = seg_bayes_cont_res
                                    else:
                                        st.session_state[f'segmented_bayesian_results{cycle_suffix}'][segment_value] = None 
                        else:
                             st.session_state[f'segmented_freq_results{cycle_suffix}'] = {} 
                             st.session_state[f'segmented_bayesian_results{cycle_suffix}'] = {}
                    except Exception as e: st.error(f"An error occurred during analysis setup or execution: {e}"); st.exception(e)
        except Exception as e: st.error(f"Error reading/processing CSV: {e}"); st.exception(e)
    else: st.info("Upload a CSV file to begin analysis.")

    # --- Display Results ---
    if st.session_state[f'analysis_done{cycle_suffix}']:
        alpha_display = st.session_state[f'alpha_for_analysis{cycle_suffix}']
        metric_type_display = st.session_state[f'metric_type_analysis{cycle_suffix}']
        control_name_display = st.session_state[f'control_group_name_analysis{cycle_suffix}']
        outcome_col_display = st.session_state[f'outcome_col_analysis{cycle_suffix}']
        variation_col_display = st.session_state[f'variation_col_analysis{cycle_suffix}']
        df_overall_display = st.session_state[f'df_analysis{cycle_suffix}'] 
        
        # --- Display Overall Frequentist Analysis ---
        st.markdown("---"); st.subheader(f"Overall Frequentist Analysis Results ({metric_type_display} Outcome)")
        overall_df_for_freq_display = df_overall_display.copy() 
        if metric_type_display == 'Binary': 
            if '__outcome_processed__' not in overall_df_for_freq_display.columns: 
                success_val = st.session_state[f'success_value_analysis{cycle_suffix}']
                if pd.isna(success_val): overall_df_for_freq_display['__outcome_processed__'] = overall_df_for_freq_display[outcome_col_display].isna().astype(int)
                else: overall_df_for_freq_display['__outcome_processed__'] = (overall_df_for_freq_display[outcome_col_display] == success_val).astype(int)
        
        display_frequentist_analysis(
            overall_df_for_freq_display, 
            metric_type_display, 
            outcome_col_display, 
            variation_col_display, 
            control_name_display, 
            alpha_display, 
            summary_stats_key_suffix="_overall_display" 
        )

        # --- Display Overall Bayesian Results ---
        overall_summary_stats_for_bayes_display = st.session_state.get(f'overall_freq_summary_stats{cycle_suffix}') 
        
        if metric_type_display == 'Binary':
            bayesian_results_to_display = st.session_state.get(f'overall_bayesian_results_binary{cycle_suffix}')
            display_bayesian_binary_results(bayesian_results_to_display, overall_summary_stats_for_bayes_display, control_name_display, alpha_display, section_title_prefix="Overall")
        elif metric_type_display == 'Continuous':
            bayesian_results_to_display_cont = st.session_state.get(f'overall_bayesian_results_continuous{cycle_suffix}')
            display_bayesian_continuous_results(bayesian_results_to_display_cont, overall_summary_stats_for_bayes_display, control_name_display, alpha_display, outcome_col_display, section_title_prefix="Overall")

        # --- Display Segmented Analysis ---
        segmented_freq_data = st.session_state.get(f'segmented_freq_results{cycle_suffix}', {})
        segmented_bayes_data = st.session_state.get(f'segmented_bayesian_results{cycle_suffix}', {})
        selected_segment_cols_disp = st.session_state.get(f'segmentation_cols{cycle_suffix}', [])


        if selected_segment_cols_disp: 
            st.markdown("---"); st.subheader("Segmented Analysis Results")
            if not segmented_freq_data and selected_segment_cols_disp: 
                 st.info("No segments to display analysis for (e.g. data might be empty after filtering for segments).")

            for segment_name, segment_info in segmented_freq_data.items():
                with st.expander(f"Segment: {segment_name}", expanded=False):
                    st.markdown(f"#### Frequentist Analysis for Segment: {segment_name}")
                    segment_df_for_display = segment_info['data'] 
                    
                    if metric_type_display == 'Binary' and '__outcome_processed__' not in segment_df_for_display.columns:
                        success_val_seg = st.session_state[f'success_value_analysis{cycle_suffix}']
                        if pd.isna(success_val_seg): segment_df_for_display['__outcome_processed__'] = segment_df_for_display[outcome_col_display].isna().astype(int)
                        else: segment_df_for_display['__outcome_processed__'] = (segment_df_for_display[outcome_col_display] == success_val_seg).astype(int)
                    elif metric_type_display == 'Continuous': 
                        segment_df_for_display[outcome_col_display] = pd.to_numeric(segment_df_for_display[outcome_col_display], errors='coerce')

                    segment_summary_stats_for_segment_display = display_frequentist_analysis(
                        segment_df_for_display, 
                        metric_type_display, 
                        outcome_col_display, 
                        variation_col_display, 
                        control_name_display, 
                        alpha_display,
                        summary_stats_key_suffix=f"_segment_display_{segment_name.replace(' | ','_')}"
                    )
                    
                    segment_bayes_result_data = segmented_bayes_data.get(segment_name)
                    segment_summary_for_bayes_ordering_from_run = segment_info.get('summary_stats') 

                    if segment_bayes_result_data is not None and segment_summary_for_bayes_ordering_from_run is not None:
                        if metric_type_display == 'Binary':
                            display_bayesian_binary_results(segment_bayes_result_data, segment_summary_for_bayes_ordering_from_run, control_name_display, alpha_display, section_title_prefix=f"Segment '{segment_name}'")
                        elif metric_type_display == 'Continuous':
                            display_bayesian_continuous_results(segment_bayes_result_data, segment_summary_for_bayes_ordering_from_run, control_name_display, alpha_display, outcome_col_display, section_title_prefix=f"Segment '{segment_name}'")
                    elif segment_summary_stats_for_segment_display is None or segment_summary_stats_for_segment_display.empty:
                         st.warning(f"Bayesian analysis for segment '{segment_name}' skipped due to lack of valid summary statistics from Frequentist step.")
                    else:
                        st.info(f"Bayesian analysis results for segment '{segment_name}' are not available or could not be computed.")
        
    st.markdown("---")
    st.info("This app provides guidance and calculations for A/B testing. Always consult with a statistician for critical decisions.")


def show_interpret_results_page():
    # Page for interpreting results and decision guidance.
    st.header("Interpreting Results & Detailed Decision Guidance 🧐")
    st.markdown("""
    Once your A/B test is complete and you've analyzed the data, the next crucial step is interpretation. This involves looking beyond just a single number (like a p-value or a lift percentage) and considering the broader context to make an informed decision.
    This page aims to guide you through that process.
    """)
    st.markdown("---")

    st.subheader("1. Statistical Significance vs. Practical Significance")
    st.markdown("""
    * **Statistical Significance (p-value & alpha):**
        * A statistically significant result (typically p-value < alpha, e.g., < 0.05) suggests that the observed difference between variations is unlikely to have occurred by random chance alone, assuming there's no real difference (the null hypothesis).
        * It **does not** tell you the *size* or *importance* of the difference.
        * With very large sample sizes, even tiny, practically meaningless differences can become statistically significant.
    * **Practical Significance (Business Impact & MDE):**
        * This refers to whether the observed difference is large enough to be meaningful or valuable from a business or user experience perspective.
        * Does the change lead to an improvement that justifies the cost of implementation, potential risks, or engineering effort?
        * Your **Minimum Detectable Effect (MDE)**, defined during test design, is a key benchmark for practical significance. If the observed effect (and its confidence/credible interval) is smaller than your MDE, it might not be practically significant even if it's statistically significant.
    * **The Four Scenarios:**
        1.  **Statistically Significant & Practically Significant:** Ideal outcome! You have evidence the effect is real and large enough to matter. This is usually a strong candidate for launch.
        2.  **Statistically Significant & Not Practically Significant:** The effect is likely real, but too small to be worth implementing based on your MDE. Consider if the cost of implementation is negligible or if there are other non-quantifiable benefits.
        3.  **Not Statistically Significant & Practically Significant (Potentially):** The observed difference is large enough that it *would* be valuable if real, but your test lacked the power to confirm it statistically. This might warrant further testing with a larger sample if the potential gain is high and the cost of further testing is acceptable.
        4.  **Not Statistically Significant & Not Practically Significant:** No strong evidence of a real effect, and even if there were one, it's likely too small to matter. Usually, this means sticking with the control.
    * **Example:** A 0.01% increase in conversion rate might be statistically significant with millions of users, but it's unlikely to be practically significant for most businesses. Conversely, a 5% increase might be practically significant, but if your sample size was too small, it might not achieve statistical significance.
    """)
    st.markdown("---")

    st.subheader("2. The Role of Effect Size")
    st.markdown("""
    * **What is it?** Effect size is a quantitative measure of the magnitude of an experimental effect. Unlike significance tests (p-values), it's not directly affected by sample size in the same way.
    * **Why is it important?**
        * It helps you understand the *strength* of the relationship or the *magnitude* of the difference between groups.
        * It provides a more complete picture when combined with p-values. A small p-value tells you an effect is likely real; the effect size tells you how big that effect is.
        * It allows for comparison of results across different studies or tests (meta-analysis).
        * It's crucial for determining practical significance – is the effect large enough to care about?
    * **Common Measures (Examples - this app provides some of these directly in the results tables):**
        * **For Binary Outcomes (Proportions):**
            * *Risk Difference (Absolute Uplift/Difference):* `CR_variation - CR_control`. Directly interpretable (e.g., a 2 percentage point increase).
            * *Relative Risk (Relative Uplift/Difference):* `CR_variation / CR_control`. Often expressed as a percentage change.
            * *Odds Ratio:* `(Conversions_V / NonConversions_V) / (Conversions_C / NonConversions_C)`. Less intuitive for direct business interpretation but common in some statistical fields.
        * **For Continuous Outcomes (Means):**
            * *Difference in Means (Absolute Difference):* `Mean_variation - Mean_control`. Directly interpretable in the units of your metric (e.g., \$2.50 increase in AOV).
            * *Cohen's d:* A standardized mean difference, `(Mean_variation - Mean_control) / Pooled_StdDev`. Provides a scale-independent measure (e.g., small ≈ 0.2, medium ≈ 0.5, large ≈ 0.8). This app does not calculate Cohen's d directly, but the absolute difference is provided.
    * **Interpretation:** Always consider the effect size in the context of your specific domain and goals. What constitutes a "small" or "large" effect can vary greatly. An "Expected Uplift" or "Expected Difference" from the Bayesian analysis is also a form of effect size (the mean of the posterior distribution of the difference).
    """)
    st.markdown("---")

    st.subheader("3. Leveraging Confidence and Credible Intervals")
    st.markdown("""
    Both Frequentist Confidence Intervals (CIs) and Bayesian Credible Intervals (CrIs) provide a range of plausible values for the true effect (e.g., the true difference in conversion rates or means between a variation and the control).
    * **Range of Plausible Values:** They quantify the uncertainty around your point estimate (e.g., the observed lift or difference).
        * A **Frequentist 95% CI** means that if you were to repeat the experiment many times, 95% of the CIs constructed in that manner would contain the true population parameter. It's a statement about the interval construction process.
        * A **Bayesian 95% CrI** means there is a 95% probability that the true population parameter lies within this interval, given your data and the chosen prior. This is often more intuitive.
    * **Checking for "No Effect" (Zero):**
        * If a CI/CrI for the *difference* between a variation and control **includes zero**, it means that "no difference" is a plausible value.
            * For Frequentist CIs, this often aligns with a non-significant p-value (p > alpha).
            * For Bayesian CrIs, it means there's a non-negligible probability that the true difference is zero or even in the opposite direction of your point estimate.
    * **Assessing Practical Significance:**
        * Compare the entire interval (especially the lower bound for an improvement, or upper bound for a cost/decrease) against your pre-defined **Minimum Detectable Effect (MDE)** or any other threshold of practical importance.
        * If the *entire* CI/CrI is above your MDE for a positive metric (e.g., the lower bound of the CI/CrI for uplift is greater than MDE), you have stronger evidence of a practically significant result.
        * If the interval overlaps with your MDE, the result is more ambiguous regarding practical significance. The true effect could be smaller or larger than your MDE.
        * If the interval is mostly below your MDE (even if it's all positive), the effect might be real but not practically significant.
    * **Width of the Interval:**
        * A **narrow** interval suggests a more precise estimate of the true effect. This typically comes from larger sample sizes or lower data variability.
        * A **wide** interval indicates more uncertainty. Even if the point estimate (e.g., mean uplift) looks good, a wide interval should temper your confidence and suggests the true effect could be quite different.
    * **Example (using CI for difference in CR % points):**
        * Variation B shows a 2% lift. 95% CI: \[0.5%, 3.5%\]. MDE is 1%.
            * *Interpretation:* Statistically significant (doesn't include 0). Practically significant (entire CI is above the 1% MDE, or at least the lower bound is close enough to be considered). High confidence in a meaningful positive effect.
        * Variation C shows a 1% lift. 95% CI: \[-1.0%, 3.0%\]. MDE is 1%.
            * *Interpretation:* Not statistically significant (CI includes 0). Practical significance is uncertain; the true effect could range from a decrease to a notable increase.
        * Variation D shows a 0.2% lift. 95% CI: \[0.1%, 0.3%\]. MDE is 1%.
            * *Interpretation:* Statistically significant (CI doesn't include 0). However, not practically significant as the entire CI is well below the 1% MDE. The effect is likely real but too small to matter.
    """)
    st.markdown("---")

    st.subheader("4. Integrating Bayesian Probabilities")
    st.markdown("""
    Bayesian analysis provides direct probabilities that can be very intuitive for decision-making:
    * **P(Variation > Control) (Probability of Being Better):**
        * This tells you the likelihood that the variation is truly outperforming the control, given the data.
        * A high value (e.g., >90% or >95%, depending on your risk tolerance) gives you confidence in the direction of the effect.
        * **Combine with Magnitude:** Always look at this alongside the **Expected Uplift/Difference** and its **Credible Interval**. A 99% P(Better) is less exciting if the CrI for uplift is \[0.01%, 0.02%\] than if it's \[2%, 5%\].
    * **P(Being Best) (Probability of Being the Best Variation):**
        * In tests with multiple variations (A/B/n), this helps identify the overall winner.
        * The variation with the highest P(Being Best) is the most likely candidate to have the highest true performance.
        * Again, consider this alongside the **magnitude of differences** between the top contenders. If Var C has P(Best) = 60% and Var D has P(Best) = 35%, but their expected performances and CrIs are very similar, the choice might not be clear-cut based on this probability alone.
    * **Using these probabilities:**
        * Set decision thresholds: e.g., "We will consider launching if P(Better > Control) is > 95% AND the lower bound of the CrI for uplift is above our MDE."
        * Risk assessment: If P(Better > Control) is 80%, it means there's a 20% chance the variation is actually the same or worse. Is this an acceptable risk?
    """)
    st.markdown("---")

    st.subheader("5. Decision-Making Frameworks & Business Context")
    st.markdown("""
    Statistical results are just one piece of the puzzle. Always integrate them with your business context:
    * **Cost vs. Benefit:**
        * What is the cost of implementing the change (development time, resources, potential disruption)?
        * What is the potential benefit if the variation is successful (increased revenue, engagement, efficiency)?
        * Is the expected uplift (considering its uncertainty via the CrI) large enough to justify the costs?
    * **Risk Tolerance:**
        * How much risk is the business willing to take?
        * If a variation has a 70% chance of being better but a 30% chance of being slightly worse, is that acceptable? This depends on the potential downside.
        * Consider the "Expected Loss" (an advanced Bayesian concept not yet in this app) if you were to make the wrong decision.
    * **Strategic Alignment:**
        * Does the change align with overall business strategy and goals?
        * Even a winning test might not be implemented if it conflicts with long-term vision or brand identity.
    * **User Experience:**
        * Beyond the primary metric, does the change negatively impact other aspects of user experience? (e.g., increased clicks but also increased frustration).
    * **Segmentation Insights:**
        * Did the change perform differently for various user segments? A variation might be a loser overall but a big winner for a key demographic, potentially leading to a targeted rollout.
    """)
    st.markdown("---")

    st.subheader("6. Common Post-Test Actions")
    st.markdown("""
    Based on your interpretation, common actions include:
    * ✅ **Launch the Winning Variation:**
        * Strong evidence (statistical and practical significance, favorable Bayesian probabilities, positive CrI).
        * Benefits outweigh costs and risks.
        * Monitor post-launch performance closely.
    * 🔄 **Iterate on the Variation:**
        * The test showed promise but wasn't a clear winner (e.g., statistically significant but small effect, or P(Better) is moderate but CrI includes zero).
        * Learnings from the test suggest how the variation could be improved.
        * Formulate a new hypothesis and design a new test.
    * 🗑️ **Discard the Variation (Stick with Control):**
        * No evidence of improvement, or evidence of negative impact.
        * Statistically significant but not practically significant, and implementation costs are non-trivial.
    * 📚 **Learn and Re-evaluate Hypothesis:**
        * Even "failed" tests provide learnings. Why didn't the hypothesis hold true?
        * Analyze segments to understand if the effect varied.
        * Use the insights to form new, more informed hypotheses for future tests.
    * 🧪 **Conduct Further Testing:**
        * Results are inconclusive but suggest a potentially valuable effect (e.g., not statistically significant but observed effect is large and MDE was ambitious).
        * Consider if a larger sample size is feasible to gain more precision.
    """)
    st.markdown("---")
    
    st.subheader("7. A Practical Decision-Making Checklist")
    st.markdown("""
    Use this checklist to guide your decision-making process after analyzing your A/B test results. It's not exhaustive but covers key considerations:

    **A. Statistical Evidence Review:**
    * **Frequentist Metrics (if applicable):**
        * Is the p-value for the difference between the variation and control below your chosen alpha (e.g., 0.05)? (This indicates statistical significance).
        * What is the Confidence Interval (CI) for the difference?
            * Does it include zero? (If yes, suggests no statistically significant difference at your alpha level).
            * How wide is the CI? (Wider implies more uncertainty about the true effect size).
    * **Bayesian Metrics (if applicable):**
        * What is the Probability of Being Better (P(Variation > Control))? (A higher value, e.g., >95%, suggests stronger evidence the variation is superior).
        * What is the Credible Interval (CrI) for the difference/uplift?
            * Does it include zero? (If yes, "no difference" is a plausible true state).
            * How wide is the CrI?
        * What is the Expected Uplift/Difference? (This is your best guess for the magnitude of the effect).
        * If multiple variations were tested, what is the Probability of Being Best for each?

    **B. Practical & Business Significance Assessment:**
    * **Effect Size:** Is the observed lift/difference (e.g., absolute uplift from frequentist results, or expected difference from Bayesian results) large enough to be meaningful for your business goals?
    * **MDE Comparison:** How does the observed effect (and particularly the lower bound of its CI/CrI) compare to your pre-defined Minimum Detectable Effect (MDE)?
        * Is the lower bound of the CI/CrI for a positive effect comfortably above your MDE? (Strongest case for practical significance).
        * Does the CI/CrI overlap significantly with values below your MDE?
    * **Cost of Implementation:** What are the engineering, design, marketing, or other operational costs associated with launching the variation?
    * **Potential ROI:** If the observed lift is real (consider the range in CI/CrI), what is the estimated return on investment when factoring in implementation costs? Is it compelling?
    * **Risk Assessment:**
        * What is the risk of a false positive (Type I error - launching a variation that isn't truly better)? (Consider your alpha level).
        * What is the risk of a false negative (Type II error - missing a truly better variation)? (Consider your test's statistical power).
        * From a Bayesian perspective, what's the probability the variation is actually worse or no different (i.e., 1 - P(Better > Control))?
        * What is the potential negative impact (e.g., financial, user experience) if the variation performs worse than expected or worse than the control post-launch?

    **C. Broader Context & Strategic Considerations:**
    * **Strategic Alignment:** Does the proposed change align with your overall product roadmap, business strategy, and brand identity?
    * **User Experience (Holistic):** Beyond the primary metric, are there any potential positive or negative impacts on other aspects of user experience (e.g., usability, accessibility, other secondary metrics)?
    * **Segmentation Insights:** Did the variation perform significantly differently for key user segments? Could this lead to a targeted launch or highlight areas for further investigation?
    * **Learnings:** Regardless of whether you launch the variation, what did you learn from this test? How can these insights inform future hypotheses and experiments?
    * **Confidence in Decision:** Based on all the above, how confident are you in making a decision? Is more information or testing needed?

    **Decision Point:**
    * Based on the comprehensive review, select one of the "Common Post-Test Actions" (Launch, Iterate, Discard, Learn More/Further Testing).
    * Document your decision and the rationale behind it.
    """)
    st.markdown("---")

    st.subheader("8. Scenario-Based Examples")
    st.markdown("""
    Let's walk through a few hypothetical scenarios to see how these principles might be applied.
    *(Note: These are simplified examples. Real-world interpretation often involves more nuance and domain expertise.)*

    **Scenario 1: "The Clear Winner"**
    * **Metric:** Purchase Conversion Rate (Binary)
    * **Control CR:** 2.0%
    * **Variation B CR:** 2.5% (Observed Absolute Uplift: +0.5 percentage points, Relative Uplift: +25%)
    * **Frequentist:** P-value = 0.001 (Statistically significant at α=0.05); 95% CI for Difference: \[+0.2%, +0.8%\]
    * **Bayesian:** P(Variation B > Control) = 99.8%; Expected Uplift = +0.51 pp; 95% CrI for Uplift: \[+0.22%, +0.82%\]
    * **Business Context:** MDE was set at +0.3 percentage points. Implementation cost is low.
    * **Interpretation & Decision:**
        * Strong statistical evidence from both approaches (p < 0.05, P(B>A) is very high).
        * The entire CI/CrI for the uplift is above zero, indicating high confidence the effect is positive.
        * The lower bound of both intervals (+0.2% and +0.22%) is below the MDE of +0.3pp, but the point estimates (0.5pp, 0.51pp) and upper bounds are well above it.
        * The expected uplift is practically significant.
        * **Likely Decision:** Launch Variation B. The evidence strongly suggests a real and meaningful positive impact that likely exceeds the MDE.

    **Scenario 2: "Statistically Significant, Practically Meh?"**
    * **Metric:** Average Session Duration (Continuous)
    * **Control Mean:** 120 seconds
    * **Variation B Mean:** 122 seconds (Observed Absolute Difference: +2 seconds)
    * **Frequentist:** P-value = 0.04 (Statistically significant at α=0.05); 95% CI for Difference: \[+0.3s, +3.7s\]
    * **Bayesian:** P(Variation B > Control) = 96%; Expected Difference = +1.9 seconds; 95% CrI for Difference: \[+0.2s, +3.6s\]
    * **Business Context:** MDE was set at +10 seconds increase in session duration. The change was a minor UI tweak.
    * **Interpretation & Decision:**
        * Statistically significant, and Bayesian results also favor Variation B.
        * However, the observed effect (+2s) and the entire CI/CrI (roughly \[+0.2s, +3.7s\]) are well below the MDE of +10 seconds.
        * The improvement, while likely real, is not practically significant according to the pre-defined business goal.
        * **Likely Decision:** Discard Variation B for now, or deprioritize. The cost of rolling out even a minor change might not be justified for a ~2-second gain that doesn't meet the desired impact level. Consider if the MDE was too ambitious for such a minor change, or if session duration is truly the right success metric for this UI tweak.

    **Scenario 3: "Promising but Uncertain"**
    * **Metric:** Sign-up Rate (Binary)
    * **Control CR:** 5.0%
    * **Variation B CR:** 5.8% (Observed Absolute Uplift: +0.8 percentage points, Relative Uplift: +16%)
    * **Frequentist:** P-value = 0.15 (Not statistically significant at α=0.05); 95% CI for Difference: \[-0.2%, +1.8%\]
    * **Bayesian:** P(Variation B > Control) = 88%; Expected Uplift = +0.75 pp; 95% CrI for Uplift: \[-0.15%, +1.75%\]
    * **Business Context:** MDE was +0.5 percentage points. Sign-ups are a critical KPI.
    * **Interpretation & Decision:**
        * Not statistically significant by frequentist standards (p > 0.05, CI includes 0).
        * Bayesian P(Better > Control) is high (88%), suggesting it's quite likely better, but not with overwhelming certainty (e.g., not >95%).
        * The Expected Uplift (+0.75pp) from the Bayesian analysis *is* above the MDE (+0.5pp).
        * However, both the CI and CrI are wide and include zero, indicating considerable uncertainty. The true effect could range from a small loss to a substantial gain.
        * **Likely Decision:** This is where business judgment is key.
            * **Option 1 (Risk-Averse/Iterate):** Given the uncertainty and lack of statistical significance, iterate on the design or stick with the control. The 12% chance it's not better (100-88) might be too high for a critical KPI.
            * **Option 2 (Consider Further Testing):** If the potential +1.75pp gain is very valuable, and the cost of running a larger test is acceptable, consider re-testing with more power to narrow the CI/CrI and get a clearer signal.
            * **Option 3 (Calculated Risk Launch - Less Common):** If the cost of implementation is very low and the potential downside of a small negative effect is also very low, a business *might* consider launching based on the 88% P(Better) and positive expected uplift, but this is riskier. It depends heavily on the "cost of being wrong."
    """)
    st.markdown("---")
    st.info("This page will continue to be expanded with more detailed frameworks, interactive decision trees (future cycle), and examples for various situations.")


def show_faq_page():
    # Displays FAQ on common misinterpretations.
    st.header("FAQ on Common Misinterpretations ❓")
    st.markdown("This section addresses some common questions and misinterpretations that arise when looking at A/B test results.")
    faqs = {
        "Q: My p-value is 0.06. Is my test almost significant? Should I run it longer?": {
            "answer": "Not exactly. A p-value of 0.06 (if your alpha is 0.05) means that if the null hypothesis (no difference) were true, you'd see a result as extreme as, or more extreme than, what you observed 6% of the time due to random chance. It doesn't mean it's 'almost' significant in a graded sense. \n\n**Regarding running it longer:** If you determined your sample size correctly *before* starting the test to achieve adequate power, extending the test just to 'get' a p-value below 0.05 can inflate your Type I error rate (false positives). This is a form of p-hacking. Decisions to extend should be based on pre-defined rules or sequential testing methodologies, not on peeking at p-values.",
            "example": "Think of it like a high jump. If the bar is at 2.00m (your alpha=0.05 significance level) and you jump 1.98m (p=0.06), you haven't cleared the bar. Simply trying again and again hoping to randomly clear it changes the nature of the competition."
        },
        "Q: If a test isn't statistically significant, does it mean there's no difference between the variations?": {
            "answer": "No, not necessarily. A non-significant result means you **failed to find sufficient evidence to reject the null hypothesis** (the hypothesis of no difference). It doesn't prove the null hypothesis is true. There could be a real difference, but your test might have been underpowered (too small sample size to detect it), or the true difference might be smaller than your MDE.",
            "example": "Imagine looking for a very small, specific type of fish in a large lake with a small net. If you don't catch it, it doesn't mean the fish isn't in the lake; your net might have been too small or you might not have fished long enough in the right spot."
        },
        "Q: My A/B test showed Variation B was significantly better, but when I launched it, performance didn't improve or even got worse. Why?": {
            "answer": "This can be frustrating and can happen for several reasons:\n1.  **Regression to the Mean:** Your test might have caught Variation B on a random 'hot streak'. Over a longer period with more users, its performance might naturally regress towards its true (less impressive) mean.\n2.  **Novelty Effect:** Users might have initially reacted positively to something new and different, but this effect wears off over time.\n3.  **Segmentation Issues:** The overall 'win' might have been driven by a specific segment of users in your test. If the live traffic composition is different, the effect might dilute or reverse.\n4.  **External Factors:** Were there any marketing campaigns, holidays, or site issues during the test that might not be present post-launch (or vice-versa)?\n5.  **Type I Error (False Positive):** Even with an alpha of 5%, there's a 1 in 20 chance that your significant result was purely due to random chance.\n6.  **Implementation Issues:** Was Variation B implemented *exactly* the same way in the live environment as it was in the test? Small differences can have big impacts.",
            "example": "A new song might shoot up the charts due to initial hype (novelty) but then fade as long-term appeal isn't as strong."
        },
        "Q: Can I combine results from two separate A/B tests that tested the same feature but at different times?": {
            "answer": "Generally, this is not recommended and should be approached with extreme caution. User behavior can change over time due to seasonality, different traffic sources, product updates, or other external factors. Combining data from periods with potentially different underlying baselines or user populations can lead to misleading conclusions (Simpson's Paradox is a risk here).",
            "example": "Trying to combine lemonade sales data from a hot summer week with data from a cool autumn week to determine the effectiveness of a new sign. The weather (an external factor) is a major confounder."
        },
        "Q: Is a 200% lift with a small sample size (e.g., 1 conversion vs. 3 conversions) reliable?": {
            "answer": "Not necessarily, and often not. While the percentage lift might look huge, the absolute numbers are tiny. With such small numbers, results are highly susceptible to random chance. A single extra conversion can create a massive percentage lift. Statistical significance tests (and Bayesian credible intervals) will likely show very high uncertainty (e.g., a very wide confidence/credible interval for the difference). Always consider the absolute numbers, sample size, and the uncertainty metrics, not just the headline lift percentage.",
            "example": "If one person buys a \\$100 item in Control (1 user, 1 conversion) and three people buy in Variation (3 users, 3 conversions), that's a 200% lift in users who converted if you only count those specific users. But it's based on tiny numbers. What if the next user in Control also buys? The lift changes dramatically."
        },
        "Q: My Bayesian test shows P(B>A) = 92%. Does this mean there's a 92% chance my decision to launch B is correct?": {
            "answer": "No, not directly. P(B>A) = 92% (or 'Probability B is Better than A') means there's a 92% probability that the *true underlying parameter* (e.g., true conversion rate) of Variation B is greater than that of Variation A, given your data and your prior beliefs. \n\nWhile this is strong evidence in favor of B, the 'correctness' of a launch decision also involves considering:\n* **Magnitude of the difference:** Is the expected uplift practically significant? (See the Credible Interval for Uplift).\n* **Costs and risks:** What are the costs of implementation? What are the risks if B is actually slightly worse (there's still an 8% chance A is better or they are the same)?\n* **Business goals:** How does this uplift align with overall objectives?",
            "example": "If a weather forecast says there's a 92% chance of rain, it's a high probability. But your decision to cancel an outdoor picnic (the 'launch' decision) might also depend on how important the picnic is, whether you have a backup indoor venue, etc."
        },
        "Q: What if my control group's conversion rate in the A/B test is very different from its historical average?": {
            "answer": "This is a good flag to investigate before drawing firm conclusions from the test. It could indicate:\n1.  **Seasonality/Trends:** User behavior naturally changes over time.\n2.  **Different Traffic Mix:** The users in your test might be from different sources or demographics than usual.\n3.  **Instrumentation Error:** Are your tracking and data collection for the test set up correctly?\n4.  **Actual Change in Baseline:** Something fundamental might have shifted in overall user behavior recently.\nIt's important to understand why this discrepancy exists, as it affects the context and generalizability of your test results.",
            "example": "If your ice cream shop's historical average daily sales are 100 cones, but during a one-week test of a new flavor, the control (regular vanilla) only sells 50 cones per day, you'd want to understand why (e.g., was it unusually cold that week?) before concluding much about the new flavor's relative performance."
        },
        "Q: The A/B/n test shows Variation C is best overall based on the highest metric. Can I just ignore A and B and compare C only to the historical baseline to report the lift?": {
            "answer": "Not always safely. The direct comparison in your A/B/n test is between C and the *concurrent control (A)* run during the same period, with the same users, under the same conditions. This is the most reliable comparison. \n\nIf you compare C to a historical baseline: \n* You reintroduce all the problems of time-based differences (seasonality, traffic mix, etc.) that A/B testing is designed to control for.\n* The 'lift' might be artificially inflated or deflated due to these temporal factors, not just the impact of C. \nAlways prioritize the lift calculated against the concurrent control from the test itself.",
            "example": "In a race, even if a runner (C) finishes first, their performance improvement is best measured against their own previous times in similar conditions or against other runners (A, B) in that *same race*, not against a historical world record set years ago under different circumstances."
        }
    }
    for question, details in faqs.items():
        with st.expander(question):
            st.markdown(f"**A:** {details['answer']}")
            if "example" in details: st.markdown(f"**Analogy / Example:** {details['example']}")
    st.markdown("---")
    st.info("Content for this section will be reviewed and expanded as needed.")

def show_bayesian_guidelines_page():
    # Page for Bayesian analysis guidelines.
    st.header("Bayesian Analysis Guidelines 🧠")
    st.markdown("This section provides a guide to understanding and interpreting Bayesian A/B test results, complementing the direct outputs from the 'Analyze Results' page.")
    
    st.subheader("Core Bayesian Concepts in this App")
    st.markdown("""
    * **Prior Distribution:** Represents your beliefs about a parameter *before* seeing the test data.
        * For **binary outcomes (Conversion Rates)**, this app uses a `Beta(1,1)` prior by default. This is an "uninformative" or "flat" prior, meaning it assumes all conversion rates between 0% and 100% are equally likely before the test. This is a common starting point when strong prior information is unavailable.
        * For **continuous outcomes (Means)**, the app uses a method that approximates the posterior distribution of the mean using a t-distribution derived from your sample data (sample mean, sample standard deviation, sample size). This approach implicitly uses non-informative (or "vague") priors for the underlying true mean and variance, which is a common simplification. More advanced Bayesian models for continuous data might involve explicitly setting more informative priors (e.g., Normal prior for the mean, Inverse-Gamma prior for the variance).
    * **Likelihood:** This is how your collected A/B test data informs the model about the parameters.
        * For binary data (conversions/non-conversions), the likelihood is typically Binomial.
        * For continuous data (assuming the data points are approximately normally distributed within each group), the likelihood is Normal.
    * **Posterior Distribution:** Your updated belief about the parameter *after* combining the prior and the data (via the likelihood). This is the key output of Bayesian analysis.
        * For binary outcomes: Beta Prior + Binomial Likelihood = Beta Posterior. The Beta distribution is convenient as it's the "conjugate prior" for the Binomial likelihood.
        * For continuous outcomes: When using uninformative priors for the mean and variance, and a Normal likelihood, the marginal posterior distribution for the mean (after integrating out the unknown variance) follows a t-distribution. This is analogous to how the t-distribution arises in frequentist inference for means with unknown variance.
    """)

    st.subheader("Interpreting Key Bayesian Outputs")
    st.markdown("**For Binary Outcomes (e.g., Conversion Rates):**")
    st.markdown("""
    * **Posterior Distribution Plot (CRs):** Visualizes the range of plausible values for the true conversion rate of each variation after seeing the data. Wider distributions mean more uncertainty.
    * **Posterior Mean CR & Credible Interval (CrI) for CR:**
        * *Posterior Mean CR:* The average conversion rate for a variation, based on the posterior distribution. It's your best guess for the true CR.
        * *CrI for CR:* We are X% confident (e.g., 95%) that the true conversion rate for this variation lies within this interval. Unlike frequentist Confidence Intervals, you *can* make direct probability statements about the parameter being in the interval.
    * **P(Variation > Control) (Probability of Being Better):** The probability that the variation's true underlying conversion rate is strictly greater than the control's true conversion rate. A high value (e.g., >95%) gives strong confidence the variation is an improvement.
        * *Important Note:* Even if this probability is high, also check the **Credible Interval for Uplift**. If that interval is very wide or very close to zero (e.g., [-0.1%, 0.3%]), the magnitude of the improvement might be small or uncertain, even if you're confident it's positive. A P(Better) of 96% with an uplift CrI of [2.0%, 5.0%] is more compelling.
    * **Expected Uplift (Absolute %):** The average absolute improvement (or decline) in conversion rate you might expect from choosing a variation over the control, based on the posterior distributions of the difference.
    * **Credible Interval (CrI) for Uplift (Absolute %):** We are X% confident that the true absolute uplift (Variation CR - Control CR) lies within this interval.
        * If this interval includes 0, then 'no difference' or even a negative impact are plausible outcomes. The wider the interval, the more uncertain the uplift.
        * If the entire interval is above 0, you have strong evidence of a positive uplift.
        * If the entire interval is below 0, you have strong evidence of a negative impact.
    * **P(Being Best):** In an A/B/n test, this is the probability that a specific variation has the highest true conversion rate among all tested variations. Useful for selecting a winner when multiple variations are present, but also consider the magnitudes of differences and their CrIs.
    """)

    st.markdown("**For Continuous Outcomes (e.g., Average Order Value, Time on Page):**")
    st.markdown("""
    * **Posterior Distribution Plot (Means):** Visualizes the plausible values for the true mean of the metric for each variation, based on the t-distribution approximation of the posterior. Similar to binary, wider distributions indicate more uncertainty about the true mean.
    * **Posterior Mean & Credible Interval (CrI) for Mean:**
        * *Posterior Mean:* The average value of the metric for a variation, based on its posterior distribution (will be very close to the sample mean given the uninformative prior approach).
        * *CrI for Mean:* We are X% confident that the true mean for this variation lies within this interval. For example, a 95% CrI of [\$10.50, \$12.30] for Average Order Value (AOV) means there's a 95% probability the true AOV for that variation is between \$10.50 and \$12.30.
    * **P(Variation > Control) (Probability of Being Better):** The probability that the variation's true underlying mean is strictly greater than the control's true mean. A high value suggests the variation is likely better.
        * *Example:* If P(Var_B_Mean > Control_Mean) = 97%, there's a 97% chance that Variation B truly has a higher average outcome than Control. This is a direct probabilistic statement about the hypothesis.
    * **Expected Difference (Absolute):** The average absolute difference (Variation Mean - Control Mean) you might expect, based on the posterior distribution of the difference. This is the mean of the difference samples.
        * *Example:* If the Expected Difference for Variation B vs Control is +\$1.50 for AOV, you expect, on average, Variation B to result in an AOV that is \$1.50 higher than Control.
    * **Credible Interval (CrI) for Difference (Absolute):** We are X% confident that the true absolute difference in means lies within this interval. This is crucial for understanding the magnitude and uncertainty of the difference.
        * *Example 1 (Uncertain but likely positive):* A 95% CrI for Difference of [-\$0.50, +\$3.50] for AOV, with an Expected Difference of +\$1.50. While the expected difference is positive and P(Better) might be high, the interval includes \$0. This means it's plausible there's no real difference, or even a slight negative effect. The decision to launch might depend on the cost/risk.
        * *Example 2 (Strongly positive):* A 95% CrI for Difference of [+\$0.50, +\$2.50]. Here, the entire interval is above zero, providing strong evidence that the variation has a positive impact, with the true effect likely between \$0.50 and \$2.50.
    * **P(Being Best):** The probability that a specific variation has the highest true mean among all tested variations. Useful for ranking, but always consider the CrIs for differences to ensure the "best" is meaningfully better than others, especially the control.
    """)
    
    st.subheader("Advantages of Bayesian A/B Testing")
    st.markdown("""
    * **Intuitive Results:** Probabilities like "P(Variation B is better than A) is 92%" are often more aligned with how business stakeholders think about risk and decisions.
    * **Good for Smaller Samples (with informative priors):** While this app uses uninformative priors by default, Bayesian methods can formally incorporate prior knowledge, which can be beneficial with limited data (though this feature is not yet in this app).
    * **Probability Statements about Hypotheses:** Bayesian analysis allows you to make direct probability statements about your hypotheses (e.g., "There is an X% chance that variation B's CR is above Y%").
    * **More Robust to "Peeking":** While not a license for continuous stopping, the interpretation of Bayesian posteriors is less affected by optional stopping compared to frequentist p-values. The posterior is simply your current state of belief given the data seen so far.
    * **Decision Making Frameworks:** Bayesian results (like expected loss/uplift) can directly feed into decision theory frameworks (a more advanced topic).
    """)
    st.info("This section will be further expanded with more detailed examples, discussion on the choice of priors for advanced users, and interpretation nuances for different scenarios.")


def show_roadmap_page():
    # Displays the roadmap and future features.
    st.header("Roadmap / Possible Future Features 🚀")
    st.markdown("This application has several potential features planned for future development:")
    if FUTURE_FEATURES:
        for feature, description in FUTURE_FEATURES.items(): st.markdown(f"- **{feature}:** {description}")
    else: st.write("No future features currently listed.")
    st.markdown("---")
    st.markdown("Feedback on feature prioritization is welcome.")

# --- Main App Navigation ---
st.sidebar.title("Navigation")
PAGES = {
    "Introduction to A/B Testing": show_introduction_page,
    "Designing Your A/B Test": show_design_test_page,
    "Analyze Results": show_analyze_results_page,
    "Interpreting Results & Detailed Decision Guidance": show_interpret_results_page, 
    "Bayesian Analysis Guidelines": show_bayesian_guidelines_page, 
    "FAQ on Misinterpretations": show_faq_page,
    "Roadmap / Possible Future Features": show_roadmap_page
}
selection = st.sidebar.radio("Go to", list(PAGES.keys()))

if callable(PAGES.get(selection)):
    page_function = PAGES[selection]
    page_function()
else:
    st.error("Selected page could not be loaded.")


st.sidebar.markdown("---")
st.sidebar.info("A/B Testing Guide & Analyzer | V1.0.0 (Final Polish & Review)")
