import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
import re # For RTL text in matplotlib
from matplotlib.font_manager import FontProperties # For custom fonts in matplotlib

# Attempt to set a font that supports Arabic for Matplotlib
try:
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Tahoma', 'DejaVu Sans', 'Arial', 'Amiri', 'KacstOne', 'sans-serif']
except Exception as e:
    st.warning(f"Could not set a specific Arabic font for Matplotlib, will use default. Error: {e}")


# Page Configuration
st.set_page_config(layout="wide", page_title="Adaptive Bayesian Estimation Proposal")

# --- Helper Functions for Interactive Illustrations ---
def plot_beta_distribution(alpha, beta, label, ax, lang='en'):
    """Plots a Beta distribution."""
    x = np.linspace(0, 1, 500)
    y = stats.beta.pdf(x, alpha, beta)

    label_display = label
    if lang == 'ar':
        if label == "Prior":
            label_display = "التوزيع الأولي"
        elif label == "Posterior":
            label_display = "التوزيع اللاحق"
        elif "Old Posterior" in label:
            label_display = "التوزيع اللاحق القديم (بيانات من T-1)"
        elif "Fixed Initial Prior" in label:
            label_display = "التوزيع الأولي الثابت"
        elif "New Prior" in label:
            match = re.search(r'δ=([\d.]+)', label)
            delta_val = match.group(1) if match else ""
            label_display = f"التوزيع الأولي الجديد (δ={delta_val})"

    ax.plot(x, y, label=f'{label_display} (α={alpha:.2f}, β={beta:.2f})')
    ax.fill_between(x, y, alpha=0.2)

def update_beta_parameters(prior_alpha, prior_beta, successes, failures):
    """Updates Beta parameters given new data."""
    posterior_alpha = prior_alpha + successes
    posterior_beta = prior_beta + failures
    return posterior_alpha, posterior_beta

def get_credible_interval(alpha, beta, conf_level=0.95):
    """Calculates the credible interval for a Beta distribution."""
    if alpha <= 0 or beta <= 0:
        return (0,0)
    return stats.beta.interval(conf_level, alpha, beta)

# --- Translation Dictionary ---
translations = {
    "en": {
        "page_title": "Proposal: Adaptive Bayesian Estimation for Pilgrim Satisfaction Surveys",
        "sidebar_title": "Proposal Sections",
        "sidebar_info": "This app presents a proposal for using Bayesian adaptive estimation for Hajj pilgrim satisfaction surveys. Developed by Dr. Mohammad Nabhan.",
        "go_to": "Go to",
        "sections": {
            "1. Introduction & Objectives": "1. Introduction & Objectives",
            "2. Challenges Addressed": "2. Challenges Addressed",
            "3. Bayesian Adaptive Methodology": "3. Bayesian Adaptive Methodology",
            "4. Implementation Roadmap": "4. Implementation Roadmap",
            "5. Note to Practitioners": "5. Note to Practitioners",
            "6. Interactive Illustration": "6. Interactive Illustration",
            "7. Conclusion": "7. Conclusion"
        },
        "introduction_objectives_header": "1. Introduction & Objectives",
        "introduction_objectives_markdown_1": """
This proposal outlines an **Adaptive Bayesian Estimation framework** designed to enhance the process of gathering and analyzing survey data for Hajj pilgrim satisfaction and the assessment of services provided by various companies.

The current practice of developing satisfaction metrics month over month faces complexities, such as delays in pilgrim arrivals or non-uniformity across different months, making it difficult to consistently achieve high-confidence and low-error confidence intervals for key indicators. This proposal aims to introduce a more dynamic, efficient, and robust methodology.
""",
        "introduction_objectives_subheader_1.1": "1.1. Primary Objectives",
        "introduction_objectives_markdown_1.1_content": """
The core objectives of implementing this adaptive Bayesian framework are:

* **Achieve Desired Precision Efficiently:** To obtain satisfaction metrics and service provider assessments with pre-defined levels of precision (i.e., narrow credible intervals at a specific confidence level) using optimized sample sizes.
* **Dynamic Sampling Adjustments:** To iteratively adjust sampling efforts based on accumulating evidence. This means collecting more data only when and where it's needed to meet precision targets, avoiding over-sampling or under-sampling.
* **Timely and Reliable Estimates:** To provide decision-makers with more timely and statistically robust estimates, allowing for quicker responses to emerging issues or trends in pilgrim satisfaction.
* **Incorporate Prior Knowledge:** To formally integrate knowledge from previous survey waves, historical data, or expert opinions into the estimation process, leading to more informed starting points and potentially faster convergence to precise estimates.
* **Adapt to Changing Conditions:** To develop a system that can adapt to changes in satisfaction levels or service provider performance over time, for instance, by adjusting the influence of older data.
* **Enhanced Subgroup Analysis:** To facilitate more reliable analysis of specific pilgrim subgroups or service aspects by adaptively ensuring sufficient data is collected for these segments.
""",
        "challenges_addressed_header": "2. Challenges Addressed by this Methodology",
        "challenges_addressed_markdown": """
The proposed Bayesian adaptive estimation framework directly addresses several key challenges currently faced in the Hajj survey process:

* **Difficulty in Obtaining Stable Confidence Intervals:**
    * **Challenge:** Operational complexities like staggered pilgrim arrivals, varying visa availability periods, and diverse pilgrim schedules lead to non-uniform data collection across time. This makes it hard to achieve consistent and narrow confidence intervals for satisfaction indicators using fixed sampling plans.
    * **Bayesian Solution:** The adaptive nature allows sampling to continue until a desired precision (credible interval width) is met, regardless of initial data flow irregularities. Estimates stabilize as more data is incorporated.

* **Inefficiency of Fixed Sample Size Approaches:**
    * **Challenge:** Predetermined sample sizes often lead to either over-sampling (wasting resources when satisfaction is homogenous or already precisely estimated) or under-sampling (resulting in inconclusive results or wide confidence intervals).
    * **Bayesian Solution:** Sampling effort is guided by the current level of uncertainty. If an estimate is already precise, sampling can be reduced or stopped for that segment. If it's imprecise, targeted additional sampling is guided by the model.

* **Incorporation of Prior Knowledge and Historical Data:**
    * **Challenge:** Valuable insights from past surveys or existing knowledge about certain pilgrim groups or services are often not formally used to inform current survey efforts or baseline estimates.
    * **Bayesian Solution:** Priors provide a natural mechanism to incorporate such information. This can lead to more accurate estimates, especially when current data is sparse, and can make the learning process more efficient.

* **Assessing Service Provider Performance with Evolving Data:**
    * **Challenge:** Evaluating service providers is difficult when their performance might change over time, or when initial data for a new provider is limited. Deciding when enough data has been collected to make a fair assessment is crucial.
    * **Bayesian Solution:** The framework can be designed to track performance iteratively. For new providers, it starts with less informative priors and builds evidence. For existing ones, it can incorporate past performance, potentially with mechanisms to down-weight older data if performance is expected to evolve (see Section 3.5).

* **Balancing Fresh Data with Historical Insights:**
    * **Challenge:** Determining how much weight to give to historical data versus new, incoming data is critical, especially if there's a possibility of changes in pilgrim sentiment or service quality.
    * **Bayesian Solution:** Techniques like using power priors or dynamic models allow for a tunable "forgetting factor" or learning rate, systematically managing the influence of past data on current estimates.

* **Resource Allocation for Data Collection:**
    * **Challenge:** Allocating limited survey resources (personnel, time, budget) effectively across numerous metrics, pilgrim segments, and service providers.
    * **Bayesian Solution:** The adaptive approach helps prioritize data collection where uncertainty is highest and the need for precision is greatest, leading to more optimal resource allocation.
""",
        "bayesian_adaptive_methodology_header": "3. Core Methodology: Bayesian Adaptive Estimation",
        "bayesian_adaptive_methodology_markdown_intro": """
The Bayesian adaptive estimation framework is an iterative process that leverages Bayes' theorem to update our beliefs about pilgrim satisfaction or service performance as new survey data is collected. This allows for dynamic adjustments to the sampling strategy.
""",
        "bayesian_adaptive_methodology_subheader_3.1": "3.1. Fundamental Concepts",
        "bayesian_adaptive_methodology_markdown_3.1_content": r"""
At its heart, Bayesian inference combines prior knowledge with observed data to arrive at an updated understanding, known as the posterior distribution.

* **Prior Distribution ($P(\theta)$):** This represents our initial belief about a parameter $\theta$ (e.g., the proportion of satisfied pilgrims) *before* observing new data. It can be based on historical data, expert opinion, or be deliberately "uninformative" if we want the data to speak for itself.
* **Likelihood ($P(D|\theta)$):** This quantifies how probable the observed data ($D$) is, given a particular value of the parameter $\theta$. It is the function that connects the data to the parameter.
* **Posterior Distribution ($P(\theta|D)$):** This is our updated belief about $\theta$ *after* observing the data. It is calculated using Bayes' Theorem:
    $$ P(\theta|D) = \frac{P(D|\theta) \times P(\theta)}{P(D)} $$
    Where $P(D)$ is the marginal likelihood of the data, acting as a normalizing constant. In practice, we often focus on the proportionality:
    $$ P(\theta|D) \propto P(D|\theta) \times P(\theta) $$
* **Credible Interval:** In Bayesian statistics, a credible interval is a range of values that contains the parameter $\theta$ with a certain probability (e.g., 95%). This is a direct probabilistic statement about the parameter, unlike the frequentist confidence interval.
""",
        "bayesian_adaptive_methodology_subheader_3.2": "3.2. The Iterative Process",
        "bayesian_adaptive_methodology_markdown_3.2_content": """
The adaptive methodology follows these steps:
1.  **Initialization:**
    * Define the parameter(s) of interest (e.g., satisfaction with lodging, food, logistics for a specific company).
    * Specify an initial **prior distribution** for each parameter. For satisfaction proportions, a Beta distribution is commonly used.
    * Set a target precision (e.g., a maximum width for the 95% credible interval).

2.  **Initial Data Collection:**
    * Collect an initial batch of survey responses relevant to the parameter(s). The size of this initial batch can be based on practical considerations or a small fixed number.

3.  **Posterior Update:**
    * Use the collected data (likelihood) and the current prior distribution to calculate the **posterior distribution** for each parameter.

4.  **Precision Assessment:**
    * Calculate the credible interval from the posterior distribution.
    * Compare the width of this interval to the target precision.

5.  **Adaptive Decision & Iteration:**
    * **If Target Precision Met:** For the given parameter, the current level of precision is sufficient. Sampling for this specific indicator/segment can be paused or stopped. The current posterior distribution provides the estimate and its uncertainty.
    * **If Target Precision Not Met:** More data is needed.
        * Determine an appropriate additional sample size. This can be guided by projecting how the credible interval width might decrease with more data (based on the current posterior).
        * Collect the additional batch of survey responses.
        * Return to Step 3 (Posterior Update), using the current posterior as the new prior for the next update.

This cycle continues until the desired precision is achieved for all key indicators or available resources for the current wave are exhausted.
""",
        "bayesian_adaptive_methodology_image_caption": "Conceptual Flow of Bayesian Updating (Source: Medium - adapted for context)",
        "bayesian_adaptive_methodology_subheader_3.3": "3.3. Modeling Satisfaction (e.g., using Beta-Binomial Model)",
        "bayesian_adaptive_methodology_markdown_3.3_content": r"""
For satisfaction metrics that are proportions (e.g., percentage of pilgrims rating a service as "satisfied" or "highly satisfied"), the Beta-Binomial model is highly suitable and commonly used.

* **Parameter of Interest ($\theta$):** The true underlying proportion of satisfied pilgrims.
* **Prior Distribution (Beta):** We assume the prior belief about $\theta$ follows a Beta distribution, denoted as $Beta(\alpha_0, \beta_0)$.
    * $\alpha_0 > 0$ and $\beta_0 > 0$ are the parameters of the prior.
    * An uninformative prior could be $Beta(1, 1)$, which is equivalent to a Uniform(0,1) distribution.
    * Prior knowledge can be incorporated by setting $\alpha_0$ and $\beta_0$ based on historical data (e.g., $\alpha_0$ = past successes, $\beta_0$ = past failures).
* **Likelihood (Binomial/Bernoulli):** If we collect $n$ new responses, and $k$ of them are "satisfied" (successes), the likelihood of observing $k$ successes in $n$ trials is given by the Binomial distribution:
    $$ P(k, n | \theta) = \binom{n}{k} \theta^k (1-\theta)^{n-k} $$
* **Posterior Distribution (Beta):** Due to the conjugacy between the Beta prior and Binomial likelihood, the posterior distribution of $\theta$ is also a Beta distribution:
    $$ \theta | k, n \sim Beta(\alpha_0 + k, \beta_0 + n - k) $$
    So, the updated parameters are $\alpha_{post} = \alpha_0 + k$ and $\beta_{post} = \beta_0 + n - k$.
    The mean of this posterior distribution, often used as the point estimate for satisfaction, is $\frac{\alpha_{post}}{\alpha_{post} + \beta_{post}}$.

This conjugacy simplifies calculations significantly.
""",
        "bayesian_adaptive_methodology_subheader_3.4": "3.4. Adaptive Sampling Logic & Determining Additional Sample Size",
        "bayesian_adaptive_methodology_markdown_3.4_content": r"""
The decision to continue sampling is based on whether the current credible interval for $\theta$ meets the desired precision.

* **Stopping Rule:** Stop sampling for a specific metric when (for a $(1-\gamma)\%$ credible interval $[L, U]$):
    $$ U - L \leq \text{Target Width} $$
    And/or when the credible interval lies entirely above/below a certain threshold of practical importance.

* **Estimating Required Additional Sample Size (Conceptual):**
    While exact formulas for sample size to guarantee a future credible interval width are complex because the width itself is a random variable, several approaches can guide this:
    1.  **Simulation:** Based on the current posterior $Beta(\alpha_{post}, \beta_{post})$, simulate drawing additional samples of various sizes. For each simulated sample size, calculate the resulting posterior and its credible interval width. This can give a distribution of expected widths for different additional $n$.
    2.  **Approximation Formulas:** Some researchers have developed approximations. For instance, one common approach for proportions aims for a certain margin of error (half-width) $E_{target}$ in the credible interval. If the current variance of the posterior is $Var(\theta | D_{current})$, and we approximate the variance of the posterior after $n_{add}$ additional samples as roughly $\frac{Var(\theta | D_{current}) \times N_0}{N_0 + n_{add}}$ (where $N_0 = \alpha_{post} + \beta_{post}$ is the "effective prior sample size"), one can solve for $n_{add}$ that makes the future standard deviation (and thus interval width) small enough.
    3.  **Bayesian Sequential Analysis:** More formal methods from Bayesian sequential analysis (e.g., Bayesian sequential probability ratio tests - BSPRTs) can be adapted, though they might be more complex to implement initially.
    4.  **Pragmatic Batching:** Collect data in smaller, manageable batches (e.g., 30-50 responses). After each batch, reassess precision. This is often a practical starting point.

The tool should aim to provide guidance on a reasonable next batch size based on the current uncertainty and the distance to the target precision.
""",
        "bayesian_adaptive_methodology_subheader_3.5": "3.5. Handling Data Heterogeneity Over Time",
        "bayesian_adaptive_methodology_markdown_3.5_content": """
A key challenge is that service provider performance or general pilgrim satisfaction might change over time. Using historical data uncritically as a prior might be misleading if changes have occurred.

* **The "Learning Hyperparameter" (Discount Factor / Power Prior):**
    One way to address this is to down-weight older data. If we have a series of data batches $D_1, D_2, \dots, D_t$ (from oldest to newest), when forming a prior for the current period $t+1$ based on data up to $t$, we can use a "power prior" approach or a simpler discount factor.
    For example, if using the posterior from period $t$ (with parameters $\alpha_t, \beta_t$) as a prior for period $t+1$, we might introduce a discount factor $\delta \in [0, 1]$:
    $$ \alpha_{prior, t+1} = \delta \times \alpha_t + (1-\delta) \times \alpha_{initial} $$
    $$ \beta_{prior, t+1} = \delta \times \beta_t + (1-\delta) \times \beta_{initial} $$
    Where $(\alpha_{initial}, \beta_{initial})$ could be parameters of a generic, uninformative prior.
    * If $\delta = 1$, all past information is carried forward fully.
    * If $\delta = 0$, all past information is discarded, and we restart with the initial prior (re-estimation).
    * Values between 0 and 1 provide a trade-off. The "learning hyperparameter" $\delta$ can be fixed, tuned, or even learned from the data if a more complex model is used. A simpler approach is to use a fixed $\delta$, e.g., $\delta=0.8$ or $\delta=0.9$, reflecting a belief that recent data is more relevant.

* **Change-Point Detection:**
    Periodically, statistical tests can be run to detect if there has been a significant change in the underlying satisfaction or performance metric. If a change point is detected (e.g., using CUSUM charts on posterior means, or more formal Bayesian change-point models), the prior for subsequent estimations might be reset to be less informative, or data before the change point heavily discounted or discarded.

* **Hierarchical Bayesian Models (Advanced):**
    These models can explicitly model variation over time or across different service providers simultaneously, allowing "borrowing strength" across units while also estimating individual trajectories. This is a more sophisticated approach suitable for later phases.

The choice of method depends on the complexity deemed appropriate and the available data. Starting with a discount factor is often a pragmatic first step.
""",
        "implementation_roadmap_header": "4. Implementation Roadmap (Conceptual)",
        "implementation_roadmap_markdown": """
Implementing the Bayesian adaptive estimation framework involves several key stages:
""",
        "roadmap_df_phase_col": "Phase",
        "roadmap_df_step_col": "Step",
        "roadmap_df_desc_col": "Description",
        "roadmap_data": {
            "Phase": ["Phase 1: Foundation & Pilot", "Phase 1: Foundation & Pilot", "Phase 2: Iterative Development & Testing", "Phase 2: Iterative Development & Testing", "Phase 3: Full-Scale Deployment & Refinement", "Phase 3: Full-Scale Deployment & Refinement"],
            "Step": [
                "1. Define Key Metrics & Precision Targets",
                "2. System Setup & Prior Elicitation",
                "3. Model Development & Initial Batching Logic",
                "4. Dashboard Development & Pilot Testing",
                "5. Scaled Rollout & Heterogeneity Modeling",
                "6. Continuous Monitoring & Improvement"
            ],
            "Description": [
                "Identify critical satisfaction indicators and service aspects. For each, define the desired level of precision (e.g., 95% credible interval width of ±3%).",
                "Establish data collection pathways. For each metric, determine initial priors (e.g., $Beta(1,1)$ for uninformative, or derive from historical averages if stable and relevant).",
                "Develop the Bayesian models (e.g., Beta-Binomial) for core metrics. Implement the logic for posterior updates and initial rules for determining subsequent sample batch sizes.",
                "Create a dashboard to visualize posterior distributions, credible intervals, precision achieved vs. target, and sampling progress. Conduct a pilot study on a limited scale to test the workflow, model performance, and adaptive logic.",
                "Gradually roll out the adaptive system across more survey areas/service providers. Implement or refine mechanisms for handling data heterogeneity over time (e.g., discount factors, change-point monitoring).",
                "Continuously monitor the system's performance, resource efficiency, and the quality of estimates. Refine models, priors, and adaptive rules based on ongoing learning and feedback."
            ]
        },
        "note_to_practitioners_header": "5. Note to Practitioners",
        "note_to_practitioners_subheader_5.1": "5.1. Benefits of the Bayesian Adaptive Approach",
        "note_to_practitioners_markdown_5.1_content": """
* **Efficiency:** Targets sampling effort where it's most needed, potentially reducing overall sample sizes compared to fixed methods while achieving desired precision.
* **Adaptability:** Responds to incoming data, making it suitable for dynamic environments where satisfaction might fluctuate or where initial knowledge is low.
* **Formal Use of Prior Knowledge:** Allows systematic incorporation of historical data or expert insights, which can be particularly useful with sparse initial data for new services or specific subgroups.
* **Intuitive Uncertainty Quantification:** Credible intervals offer a direct probabilistic interpretation of the parameter's range, which can be easier for stakeholders to understand than frequentist confidence intervals.
* **Rich Output:** Provides a full posterior distribution for each parameter, offering more insight than just a point estimate and an interval.
""",
        "note_to_practitioners_subheader_5.2": "5.2. Limitations and Considerations",
        "note_to_practitioners_markdown_5.2_content": """
* **Complexity:** Bayesian methods can be conceptually more demanding than traditional frequentist approaches. Implementation requires specialized knowledge.
* **Prior Selection:** The choice of prior distribution can influence posterior results, especially with small sample sizes. This requires careful justification and transparency. While "uninformative" priors aim to minimize this influence, truly uninformative priors are not always straightforward.
* **Computational Cost:** While Beta-Binomial models are computationally simple, more complex Bayesian models (e.g., hierarchical models, models requiring MCMC simulation) can be computationally intensive.
* **Interpretation Differences:** Practitioners familiar with frequentist statistics need to understand the different interpretations of Bayesian outputs (e.g., credible intervals vs. confidence intervals).
* **"Black Box" Perception:** If not explained clearly, the adaptive nature and Bayesian calculations might be perceived as a "black box" by those unfamiliar with the methods. Clear communication is key.
""",
        "note_to_practitioners_subheader_5.3": "5.3. Key Assumptions",
        "note_to_practitioners_markdown_5.3_content": """
* **Representativeness of Samples:** Each batch of collected data is assumed to be representative of the (sub)population of interest *at that point in time*. Sampling biases will affect the validity of estimates.
* **Model Appropriateness:** The chosen likelihood and prior distributions should reasonably reflect the data-generating process and existing knowledge. For satisfaction proportions, the Beta-Binomial model is often robust.
* **Stability (or Modeled Change):** The underlying parameter being measured (e.g., satisfaction rate) is assumed to be relatively stable between iterative updates within a survey wave, OR changes are explicitly modeled (e.g., via discount factors or dynamic models). Rapid, unmodeled fluctuations can be challenging.
* **Accurate Data:** Assumes responses are truthful and accurately recorded.
""",
        "note_to_practitioners_subheader_5.4": "5.4. Practical Recommendations",
        "note_to_practitioners_markdown_5.4_content": """
* **Start Simple:** Begin with core satisfaction metrics and simple models (like Beta-Binomial). Complexity can be added iteratively as experience is gained.
* **Invest in Training:** Ensure that the team involved in implementing and interpreting the results has adequate training in Bayesian statistics.
* **Transparency is Key:** Document choices for priors, models, and adaptive rules. Perform sensitivity analyses to understand the impact of different prior choices, especially in early stages or with limited data.
* **Regular Review and Validation:** Periodically review the performance of the models. Compare Bayesian estimates with those from traditional methods if possible, especially during a transition period. Validate assumptions.
* **Stakeholder Communication:** Develop clear ways to communicate the methodology, its benefits, and the interpretation of results to stakeholders who may not be statisticians.
* **Pilot Thoroughly:** Before full-scale implementation, conduct thorough pilot studies to refine the process, test the technology, and identify unforeseen challenges.
""",
        "interactive_illustration_header": "6. Interactive Illustration: Beta-Binomial Model",
        "interactive_illustration_markdown_intro": """
This section provides a simple interactive illustration of how a Beta prior is updated to a Beta posterior with new data (Binomial likelihood). This is the core of estimating a proportion (e.g., satisfaction rate) in a Bayesian way.
""",
        "interactive_illustration_prior_beliefs_header": "Prior Beliefs",
        "interactive_illustration_prior_beliefs_markdown": "The Beta distribution $Beta(\\alpha, \\beta)$ is a common prior for proportions. $\\alpha$ can be thought of as prior 'successes' and $\\beta$ as prior 'failures'. $Beta(1,1)$ is a uniform (uninformative) prior.",
        "interactive_illustration_prior_alpha_label": "Prior Alpha (α₀)",
        "interactive_illustration_prior_beta_label": "Prior Beta (β₀)",
        "interactive_illustration_prior_mean_label": "Prior Mean",
        "interactive_illustration_prior_ci_label": "95% Credible Interval (Prior)",
        "interactive_illustration_width_label": "Width",
        "interactive_illustration_new_data_header": "New Survey Data (Likelihood)",
        "interactive_illustration_new_data_markdown": "Enter the results from a new batch of surveys.",
        "interactive_illustration_num_surveys_label": "Number of New Surveys (n)",
        "interactive_illustration_num_satisfied_label": "Number Satisfied in New Surveys (k)",
        "interactive_illustration_observed_satisfaction_label": "Observed Satisfaction in New Data",
        "interactive_illustration_posterior_beliefs_header": "Posterior Beliefs (After Update)",
        "interactive_illustration_posterior_markdown": "The posterior distribution is $Beta(\\alpha_0 + k, \\beta_0 + n - k) = Beta({posterior_alpha:.1f}, {posterior_beta:.1f})$",
        "interactive_illustration_posterior_mean_label": "Posterior Mean",
        "interactive_illustration_posterior_ci_label": "95% Credible Interval (Posterior)",
        "interactive_illustration_target_width_label": "Target Credible Interval Width for Stopping",
        "interactive_illustration_success_message": "Target precision met! Current width ({current_width:.3f}) ≤ Target width ({target_width:.3f}).",
        "interactive_illustration_warning_message": "Target precision not yet met. Current width ({current_width:.3f}) > Target width ({target_width:.3f}). Consider more samples.",
        "interactive_illustration_plot1_title": "Prior and Posterior Distributions of Satisfaction Rate",
        "interactive_illustration_plot_xlabel": "Satisfaction Rate (θ)",
        "interactive_illustration_plot_ylabel": "Density",
        "interactive_illustration_discounting_header": "Conceptual Illustration: Impact of Discounting Older Data",
        "interactive_illustration_discounting_markdown": """
This illustrates how a discount factor might change the influence of 'old' posterior data when it's used to form a prior for a 'new' period.
Assume the 'Posterior' calculated above is now 'Old Data' from a previous period.
We want to form a new prior for the upcoming period.
An 'Initial Prior' (e.g., $Beta(1,1)$) represents a baseline, less informative belief.
""",
        "interactive_illustration_discount_factor_label": "Discount Factor (δ) for Old Data",
        "interactive_illustration_discount_factor_help": "Controls weight of old data. 1.0 = full weight, 0.0 = discard old data, rely only on initial prior.",
        "interactive_illustration_initial_prior_alpha_discount_label": "Initial Prior Alpha (for new period if discounting heavily)",
        "interactive_illustration_initial_prior_beta_discount_label": "Initial Prior Beta (for new period if discounting heavily)",
        "interactive_illustration_new_prior_discount_label": "New Prior for Next Period: $Beta({new_prior_alpha:.2f}, {new_prior_beta:.2f})$",
        "interactive_illustration_new_prior_mean_discount_label": "Mean of New Prior",
        "interactive_illustration_plot2_title": "Forming a New Prior with Discounting",
        "conclusion_header": "7. Conclusion",
        "conclusion_markdown": """
The proposed Bayesian adaptive estimation framework offers a sophisticated, flexible, and efficient approach to analyzing pilgrim satisfaction surveys. By iteratively updating beliefs and dynamically adjusting sampling efforts, this methodology promises more precise and timely insights, enabling better-informed decision-making for enhancing the Hajj experience.

While it introduces new concepts and requires careful implementation, the long-term benefits—including optimized resource use and a deeper understanding of satisfaction dynamics—are substantial. This proposal advocates for a phased implementation, starting with core metrics and gradually building complexity and scope.

We recommend proceeding with a pilot project to demonstrate the practical benefits and refine the operational aspects of this advanced analytical approach.
"""
    },
    "ar": {
        "page_title": "مقترح: تقدير Bayesian التكيفي لاستطلاعات رضا الحجاج",
        "sidebar_title": "أقسام المقترح",
        "sidebar_info": "يقدم هذا التطبيق مقترحًا لاستخدام تقدير Bayesian التكيفي لاستطلاعات رضا الحجاج. تم التطوير بواسطة د. محمد نبهان.",
        "go_to": "اذهب إلى",
        "sections": {
            "1. Introduction & Objectives": "١. المقدمة والأهداف",
            "2. Challenges Addressed": "٢. التحديات التي تعالجها هذه المنهجية",
            "3. Bayesian Adaptive Methodology": "٣. منهجية Bayesian التكيفية",
            "4. Implementation Roadmap": "٤. خارطة طريق التنفيذ",
            "5. Note to Practitioners": "٥. ملاحظات للممارسين",
            "6. Interactive Illustration": "٦. توضيح تفاعلي",
            "7. Conclusion": "٧. الخاتمة"
        },
        "introduction_objectives_header": "١. المقدمة والأهداف",
        "introduction_objectives_markdown_1": """
يطرح هذا المقترح إطار **تقدير Bayesian التكيفي** المصمم لتعزيز عملية جمع وتحليل بيانات الاستطلاع المتعلقة برضا حجاج بيت الله الحرام وتقييم الخدمات المقدمة من قبل مختلف الشركات.

تواجه الممارسة الحالية لتطوير مقاييس الرضا شهراً بعد شهر تعقيدات، مثل التأخير في وصول الحجاج أو عدم التوحيد عبر الأشهر المختلفة، مما يجعل من الصعب تحقيق فترات ثقة (confidence intervals) عالية ومعدلات خطأ منخفضة للمؤشرات الرئيسية بشكل مستمر. يهدف هذا المقترح إلى تقديم منهجية أكثر ديناميكية وكفاءة وقوة.
""",
        "introduction_objectives_subheader_1.1": "١.١. الأهداف الأساسية",
        "introduction_objectives_markdown_1.1_content": """
تتمثل الأهداف الأساسية لتطبيق إطار Bayesian التكيفي هذا في:

* **تحقيق الدقة المطلوبة بكفاءة:** الحصول على مقاييس الرضا وتقييمات مقدمي الخدمات بمستويات دقة محددة مسبقًا (أي، فترات موثوقية (credible intervals) ضيقة عند مستوى ثقة معين) باستخدام أحجام عينات مُحسَّنة.
* **تعديلات أخذ العينات الديناميكية:** تعديل جهود أخذ العينات بشكل متكرر بناءً على الأدلة المتراكمة. هذا يعني جمع المزيد من البيانات فقط عند الحاجة وحيثما تكون هناك حاجة لتحقيق أهداف الدقة، وتجنب الإفراط في أخذ العينات أو نقصها.
* **تقديرات موثوقة وفي الوقت المناسب:** تزويد صانعي القرار بتقديرات أكثر حداثة وقوة إحصائياً، مما يسمح باستجابات أسرع للمشكلات أو الاتجاهات الناشئة في رضا الحجاج.
* **دمج المعرفة المسبقة:** دمج المعرفة من موجات الاستطلاع السابقة أو البيانات التاريخية أو آراء الخبراء بشكل رسمي في عملية التقدير، مما يؤدي إلى نقاط انطلاق أكثر استنارة واحتمالية أسرع للوصول إلى تقديرات دقيقة.
* **التكيف مع الظروف المتغيرة:** تطوير نظام يمكنه التكيف مع التغيرات في مستويات الرضا أو أداء مقدمي الخدمات بمرور الوقت، على سبيل المثال، عن طريق تعديل تأثير البيانات الأقدم.
* **تحليل محسن للمجموعات الفرعية:** تسهيل تحليل أكثر موثوقية للمجموعات الفرعية المحددة من الحجاج أو جوانب الخدمة من خلال ضمان جمع بيانات كافية لهذه الشرائح بشكل تكيفي.
""",
        "challenges_addressed_header": "٢. التحديات التي تعالجها هذه المنهجية",
        "challenges_addressed_markdown": """
يعالج إطار تقدير Bayesian التكيفي المقترح بشكل مباشر العديد من التحديات الرئيسية التي تواجه حاليًا عملية استطلاع الحج:

* **صعوبة الحصول على فترات ثقة مستقرة:**
    * **التحدي:** التعقيدات التشغيلية مثل وصول الحجاج المتدرج، وفترات توفر التأشيرات المتفاوتة، وجداول الحجاج المتنوعة تؤدي إلى جمع بيانات غير منتظم عبر الزمن. هذا يجعل من الصعب تحقيق فترات ثقة متسقة وضيقة لمؤشرات الرضا باستخدام خطط أخذ عينات ثابتة.
    * **حل Bayesian:** تسمح الطبيعة التكيفية باستمرار أخذ العينات حتى يتم تحقيق الدقة المطلوبة (عرض فترة الموثوقية)، بغض النظر عن عدم انتظام تدفق البيانات الأولي. تستقر التقديرات مع دمج المزيد من البيانات.

* **عدم كفاءة مناهج حجم العينة الثابت:**
    * **التحدي:** غالبًا ما تؤدي أحجام العينات المحددة مسبقًا إما إلى الإفراط في أخذ العينات (إهدار الموارد عندما يكون الرضا متجانسًا أو مقدراً بدقة بالفعل) أو نقص أخذ العينات (مما يؤدي إلى نتائج غير حاسمة أو فترات ثقة واسعة).
    * **حل Bayesian:** يتم توجيه جهد أخذ العينات حسب المستوى الحالي من عدم اليقين. إذا كان التقدير دقيقًا بالفعل، يمكن تقليل أخذ العينات أو إيقافه لتلك الشريحة. إذا كان غير دقيق، يتم توجيه أخذ عينات إضافية مستهدفة بواسطة النموذج.

* **دمج المعرفة المسبقة والبيانات التاريخية:**
    * **التحدي:** غالبًا لا تُستخدم الرؤى القيمة من الاستطلاعات السابقة أو المعرفة الحالية حول مجموعات معينة من الحجاج أو الخدمات بشكل رسمي لإبلاغ جهود الاستطلاع الحالية أو التقديرات الأساسية.
    * **حل Bayesian:** توفر التوزيعات الأولية (Priors) آلية طبيعية لدمج هذه المعلومات. يمكن أن يؤدي ذلك إلى تقديرات أكثر دقة، خاصة عندما تكون البيانات الحالية متفرقة، ويمكن أن يجعل عملية التعلم أكثر كفاءة.

* **تقييم أداء مقدمي الخدمات ببيانات متطورة:**
    * **التحدي:** يعد تقييم مقدمي الخدمات أمرًا صعبًا عندما قد يتغير أداؤهم بمرور الوقت، أو عندما تكون البيانات الأولية لمقدم خدمة جديد محدودة. يعد تحديد متى تم جمع بيانات كافية لإجراء تقييم عادل أمرًا بالغ الأهمية.
    * **حل Bayesian:** يمكن تصميم الإطار لتتبع الأداء بشكل متكرر. بالنسبة لمقدمي الخدمات الجدد، يبدأ بتوزيعات أولية أقل إفادة ويبني الأدلة. بالنسبة للمقدمين الحاليين، يمكن أن يدمج الأداء السابق، مع آليات محتملة لتقليل وزن البيانات الأقدم إذا كان من المتوقع أن يتطور الأداء (انظر القسم ٣.٥).

* **الموازنة بين البيانات الحديثة والرؤى التاريخية:**
    * **التحدي:** يعد تحديد مقدار الوزن الذي يجب إعطاؤه للبيانات التاريخية مقابل البيانات الجديدة الواردة أمرًا بالغ الأهمية، خاصةً إذا كان هناك احتمال حدوث تغييرات في شعور الحجاج أو جودة الخدمة.
    * **حل Bayesian:** تسمح تقنيات مثل استخدام "power priors" أو النماذج الديناميكية بـ "عامل نسيان" قابل للضبط أو معدل تعلم، مما يدير بشكل منهجي تأثير البيانات السابقة على التقديرات الحالية.

* **تخصيص الموارد لجمع البيانات:**
    * **التحدي:** تخصيص موارد المسح المحدودة (الموظفون، الوقت، الميزانية) بفعالية عبر العديد من المقاييس وشرائح الحجاج ومقدمي الخدمات.
    * **حل Bayesian:** يساعد النهج التكيفي في تحديد أولويات جمع البيانات حيث يكون عدم اليقين هو الأعلى والحاجة إلى الدقة هي الأكبر، مما يؤدي إلى تخصيص موارد أكثر مثالية.
""",
        "bayesian_adaptive_methodology_header": "٣. المنهجية الأساسية: تقدير Bayesian التكيفي",
        "bayesian_adaptive_methodology_markdown_intro": """
إطار تقدير Bayesian التكيفي هو عملية تكرارية تستفيد من نظرية Bayes لتحديث معتقداتنا حول رضا الحجاج أو أداء الخدمة عند جمع بيانات استطلاع جديدة. وهذا يسمح بإجراء تعديلات ديناميكية على استراتيجية أخذ العينات.
""",
        "bayesian_adaptive_methodology_subheader_3.1": "٣.١. المفاهيم الأساسية",
        "bayesian_adaptive_methodology_markdown_3.1_content": r"""
في جوهره، يجمع استدلال Bayesian بين المعرفة المسبقة والبيانات المرصودة للوصول إلى فهم محدث، يُعرف بالتوزيع اللاحق (posterior distribution).

* **التوزيع الأولي ($P(\theta)$):** يمثل هذا اعتقادنا الأولي حول مُعلمة $\theta$ (على سبيل المثال، نسبة الحجاج الراضين) *قبل* ملاحظة بيانات جديدة. يمكن أن يستند إلى البيانات التاريخية، أو رأي الخبراء، أو أن يكون "غير مُعلِم" (uninformative) بشكل متعمد إذا أردنا أن تتحدث البيانات عن نفسها.
* **دالة الإمكان ($P(D|\theta)$):** تحدد هذه الدالة مدى احتمالية البيانات المرصودة ($D$)، بالنظر إلى قيمة معينة للمُعلمة $\theta$. إنها الدالة التي تربط البيانات بالمُعلمة.
* **التوزيع اللاحق ($P(\theta|D)$):** هذا هو اعتقادنا المحدث حول $\theta$ *بعد* ملاحظة البيانات. يتم حسابه باستخدام نظرية Bayes:
    $$ P(\theta|D) = \frac{P(D|\theta) \times P(\theta)}{P(D)} $$
    حيث $P(D)$ هي الإمكان الهامشي للبيانات، وتعمل كثابت تسوية. في الممارسة العملية، غالبًا ما نركز على التناسب:
    $$ P(\theta|D) \propto P(D|\theta) \times P(\theta) $$
* **فترة الموثوقية (Credible Interval):** في إحصاءات Bayesian، فترة الموثوقية هي نطاق من القيم يحتوي على المُعلمة $\theta$ باحتمال معين (على سبيل المثال، 95%). هذا بيان احتمالي مباشر حول المُعلمة، على عكس فترة الثقة (confidence interval) في الإحصاء التكراري.
""",
        "bayesian_adaptive_methodology_subheader_3.2": "٣.٢. العملية التكرارية",
        "bayesian_adaptive_methodology_markdown_3.2_content": """
تتبع المنهجية التكيفية الخطوات التالية:
١.  **التهيئة:**
    * تحديد المُعلمة (المُعلمات) محل الاهتمام (على سبيل المثال، الرضا عن السكن، الطعام، الخدمات اللوجستية لشركة معينة).
    * تحديد **توزيع أولي** مبدئي لكل مُعلمة. بالنسبة لنسب الرضا، يشيع استخدام توزيع Beta.
    * تحديد الدقة المستهدفة (على سبيل المثال، أقصى عرض لفترة الموثوقية بنسبة 95%).

٢.  **جمع البيانات الأولي:**
    * جمع دفعة أولية من استجابات الاستطلاع ذات الصلة بالمُعلمة (المُعلمات). يمكن أن يعتمد حجم هذه الدفعة الأولية على اعتبارات عملية أو عدد ثابت صغير.

٣.  **تحديث التوزيع اللاحق:**
    * استخدام البيانات المجمعة (دالة الإمكان) والتوزيع الأولي الحالي لحساب **التوزيع اللاحق** لكل مُعلمة.

٤.  **تقييم الدقة:**
    * حساب فترة الموثوقية من التوزيع اللاحق.
    * مقارنة عرض هذه الفترة بالدقة المستهدفة.

٥.  **القرار التكيفي والتكرار:**
    * **إذا تم تحقيق الدقة المستهدفة:** بالنسبة للمُعلمة المحددة، يكون مستوى الدقة الحالي كافيًا. يمكن إيقاف أخذ العينات مؤقتًا أو نهائيًا لهذا المؤشر/الشريحة المحددة. يوفر التوزيع اللاحق الحالي التقدير وعدم اليقين المصاحب له.
    * **إذا لم يتم تحقيق الدقة المستهدفة:** هناك حاجة إلى مزيد من البيانات.
        * تحديد حجم عينة إضافي مناسب. يمكن توجيه ذلك من خلال توقع كيف قد ينخفض عرض فترة الموثوقية مع المزيد من البيانات (بناءً على التوزيع اللاحق الحالي).
        * جمع الدفعة الإضافية من استجابات الاستطلاع.
        * العودة إلى الخطوة ٣ (تحديث التوزيع اللاحق)، باستخدام التوزيع اللاحق الحالي كتوزيع أولي جديد للتحديث التالي.

تستمر هذه الدورة حتى يتم تحقيق الدقة المطلوبة لجميع المؤشرات الرئيسية أو حتى استنفاد الموارد المتاحة للموجة الحالية.
""",
        "bayesian_adaptive_methodology_image_caption": "التدفق المفاهيمي لتحديث Bayesian (المصدر: Medium - تم تكييفه للسياق)",
        "bayesian_adaptive_methodology_subheader_3.3": "٣.٣. نمذجة الرضا (مثلاً، باستخدام نموذج Beta-Binomial)",
        "bayesian_adaptive_methodology_markdown_3.3_content": r"""
بالنسبة لمقاييس الرضا التي هي عبارة عن نسب (على سبيل المثال، النسبة المئوية للحجاج الذين يقيمون خدمة بأنها "مرضية" أو "مرضية للغاية")، فإن نموذج Beta-Binomial مناسب للغاية ويشيع استخدامه.

* **المُعلمة محل الاهتمام ($\theta$):** النسبة الحقيقية الكامنة للحجاج الراضين.
* **التوزيع الأولي (Beta):** نفترض أن الاعتقاد الأولي حول $\theta$ يتبع توزيع Beta، ويُرمز له بـ $Beta(\alpha_0, \beta_0)$.
    * $\alpha_0 > 0$ و $\beta_0 > 0$ هما معلمتا التوزيع الأولي.
    * يمكن أن يكون التوزيع الأولي غير المُعلِم هو $Beta(1, 1)$، وهو ما يعادل توزيع Uniform(0,1).
    * يمكن دمج المعرفة المسبقة عن طريق تعيين $\alpha_0$ و $\beta_0$ بناءً على البيانات التاريخية (على سبيل المثال، $\alpha_0$ = النجاحات السابقة، $\beta_0$ = الإخفاقات السابقة).
* **دالة الإمكان (Binomial/Bernoulli):** إذا جمعنا $n$ استجابة جديدة، وكان $k$ منها "راضياً" (نجاحات)، فإن إمكانية ملاحظة $k$ نجاحات في $n$ محاولات تُعطى بواسطة توزيع Binomial:
    $$ P(k, n | \theta) = \binom{n}{k} \theta^k (1-\theta)^{n-k} $$
* **التوزيع اللاحق (Beta):** نظرًا للترافق بين توزيع Beta الأولي ودالة الإمكان Binomial، فإن التوزيع اللاحق لـ $\theta$ هو أيضًا توزيع Beta:
    $$ \theta | k, n \sim Beta(\alpha_0 + k, \beta_0 + n - k) $$
    لذا، فإن المعلمات المحدثة هي $\alpha_{post} = \alpha_0 + k$ و $\beta_{post} = \beta_0 + n - k$.
    متوسط هذا التوزيع اللاحق، والذي يستخدم غالبًا كتقدير نقطي للرضا، هو $\frac{\alpha_{post}}{\alpha_{post} + \beta_{post}}$.

هذا الترافق يبسط الحسابات بشكل كبير.
""",
        "bayesian_adaptive_methodology_subheader_3.4": "٣.٤. منطق أخذ العينات التكيفي وتحديد حجم العينة الإضافي",
        "bayesian_adaptive_methodology_markdown_3.4_content": r"""
يعتمد قرار مواصلة أخذ العينات على ما إذا كانت فترة الموثوقية الحالية لـ $\theta$ تلبي الدقة المطلوبة.

* **قاعدة الإيقاف:** توقف عن أخذ العينات لمقياس معين عندما (لفترة موثوقية $(1-\gamma)\%$ $[L, U]$):
    $$ U - L \leq \text{Target Width} $$
    و/أو عندما تقع فترة الموثوقية بالكامل فوق/تحت عتبة معينة ذات أهمية عملية.

* **تقدير حجم العينة الإضافي المطلوب (مفاهيمي):**
    في حين أن الصيغ الدقيقة لحجم العينة لضمان عرض فترة موثوقية مستقبلية معقدة لأن العرض نفسه متغير عشوائي، يمكن للعديد من الأساليب توجيه ذلك:
    1.  **المحاكاة:** بناءً على التوزيع اللاحق الحالي $Beta(\alpha_{post}, \beta_{post})$، قم بمحاكاة سحب عينات إضافية بأحجام مختلفة. لكل حجم عينة محاكاة، احسب التوزيع اللاحق الناتج وعرض فترة الموثوقية الخاصة به. يمكن أن يعطي هذا توزيعًا للعروض المتوقعة لمختلف قيم $n$ الإضافية.
    2.  **صيغ التقريب:** طور بعض الباحثين صيغ تقريب. على سبيل المثال، يهدف نهج شائع للنسب إلى هامش خطأ معين (نصف العرض) $E_{target}$ في فترة الموثوقية. إذا كان التباين الحالي للتوزيع اللاحق هو $Var(\theta | D_{current})$، وقمنا بتقريب تباين التوزيع اللاحق بعد $n_{add}$ عينة إضافية على أنه تقريبًا $\frac{Var(\theta | D_{current}) \times N_0}{N_0 + n_{add}}$ (حيث $N_0 = \alpha_{post} + \beta_{post}$ هو "حجم العينة الأولي الفعال")، يمكن للمرء حل المعادلة لإيجاد $n_{add}$ التي تجعل الانحراف المعياري المستقبلي (وبالتالي عرض الفترة) صغيرًا بما فيه الكفاية.
    3.  **تحليل Bayesian التسلسلي:** يمكن تكييف طرق أكثر رسمية من تحليل Bayesian التسلسلي (مثل اختبارات نسبة الاحتمال التسلسلية لـ Bayesian - BSPRTs)، على الرغم من أنها قد تكون أكثر تعقيدًا في التنفيذ الأولي.
    4.  **التجميع العملي:** جمع البيانات في دفعات أصغر يمكن التحكم فيها (على سبيل المثال، 30-50 استجابة). بعد كل دفعة، أعد تقييم الدقة. غالبًا ما تكون هذه نقطة انطلاق عملية.

يجب أن تهدف الأداة إلى تقديم إرشادات بشأن حجم الدفعة التالية المعقول بناءً على عدم اليقين الحالي والمسافة إلى الدقة المستهدفة.
""",
        "bayesian_adaptive_methodology_subheader_3.5": "٣.٥. التعامل مع عدم تجانس البيانات بمرور الوقت",
        "bayesian_adaptive_methodology_markdown_3.5_content": """
أحد التحديات الرئيسية هو أن أداء مقدمي الخدمات أو رضا الحجاج بشكل عام قد يتغير بمرور الوقت. قد يكون استخدام البيانات التاريخية بشكل غير نقدي كتوزيع أولي مضللاً إذا حدثت تغييرات.

* **"مُعلمة التعلم الفائقة" (عامل الخصم / Power Prior):**
    إحدى طرق معالجة ذلك هي تقليل وزن البيانات الأقدم. إذا كان لدينا سلسلة من دفعات البيانات $D_1, D_2, \dots, D_t$ (من الأقدم إلى الأحدث)، عند تكوين توزيع أولي للفترة الحالية $t+1$ بناءً على البيانات حتى $t$، يمكننا استخدام نهج "power prior" أو عامل خصم أبسط.
    على سبيل المثال، إذا استخدمنا التوزيع اللاحق من الفترة $t$ (بالمعلمات $\alpha_t, \beta_t$) كتوزيع أولي للفترة $t+1$، فقد نقدم عامل خصم $\delta \in [0, 1]$:
    $$ \alpha_{prior, t+1} = \delta \times \alpha_t + (1-\delta) \times \alpha_{initial} $$
    $$ \beta_{prior, t+1} = \delta \times \beta_t + (1-\delta) \times \beta_{initial} $$
    حيث $(\alpha_{initial}, \beta_{initial})$ يمكن أن تكون معلمات لتوزيع أولي عام غير مُعلِم.
    * إذا كانت $\delta = 1$، يتم نقل جميع المعلومات السابقة بالكامل.
    * إذا كانت $\delta = 0$، يتم تجاهل جميع المعلومات السابقة، ونبدأ من جديد بالتوزيع الأولي المبدئي (إعادة التقدير).
    * توفر القيم بين 0 و 1 مقايضة. يمكن تثبيت "مُعلمة التعلم الفائقة" $\delta$ أو ضبطها أو حتى تعلمها من البيانات إذا تم استخدام نموذج أكثر تعقيدًا. النهج الأبسط هو استخدام $\delta$ ثابتة، على سبيل المثال، $\delta=0.8$ أو $\delta=0.9$، مما يعكس الاعتقاد بأن البيانات الحديثة أكثر صلة.

* **كشف نقطة التغيير:**
    بشكل دوري، يمكن إجراء اختبارات إحصائية للكشف عما إذا كان هناك تغيير كبير في مقياس الرضا أو الأداء الأساسي. إذا تم الكشف عن نقطة تغيير (على سبيل المثال، باستخدام مخططات CUSUM على متوسطات التوزيع اللاحق، أو نماذج نقطة التغيير الأكثر رسمية في Bayesian)، فقد يتم إعادة تعيين التوزيع الأولي للتقديرات اللاحقة ليكون أقل إفادة، أو يتم خصم البيانات قبل نقطة التغيير بشكل كبير أو تجاهلها.

* **نماذج Bayesian الهرمية (متقدم):**
    يمكن لهذه النماذج نمذجة التباين بمرور الوقت أو عبر مقدمي الخدمات المختلفين في وقت واحد، مما يسمح "باستعارة القوة" عبر الوحدات مع تقدير المسارات الفردية أيضًا. هذا نهج أكثر تعقيدًا ومناسب للمراحل اللاحقة.

يعتمد اختيار الطريقة على مدى التعقيد الذي يعتبر مناسبًا والبيانات المتاحة. غالبًا ما يكون البدء بعامل الخصم خطوة أولى عملية.
""",
        "implementation_roadmap_header": "٤. خارطة طريق التنفيذ (مفاهيمية)",
        "implementation_roadmap_markdown": """
يتضمن تنفيذ إطار تقدير Bayesian التكيفي عدة مراحل رئيسية:
""",
        "roadmap_df_phase_col": "المرحلة",
        "roadmap_df_step_col": "الخطوة",
        "roadmap_df_desc_col": "الوصف",
        "roadmap_data": {
            "Phase": ["المرحلة الأولى: التأسيس والتجربة الأولية", "المرحلة الأولى: التأسيس والتجربة الأولية", "المرحلة الثانية: التطوير التكراري والاختبار", "المرحلة الثانية: التطوير التكراري والاختبار", "المرحلة الثالثة: النشر على نطاق كامل والتحسين", "المرحلة الثالثة: النشر على نطاق كامل والتحسين"],
            "Step": [
                "١. تحديد المقاييس الرئيسية وأهداف الدقة",
                "٢. إعداد النظام واستنباط التوزيعات الأولية",
                "٣. تطوير النموذج ومنطق التجميع الأولي",
                "٤. تطوير لوحة المعلومات والاختبار التجريبي",
                "٥. التوسع التدريجي ونمذجة عدم التجانس",
                "٦. المراقبة المستمرة والتحسين"
            ],
            "Description": [
                "تحديد مؤشرات الرضا الحاسمة وجوانب الخدمة. لكل منها، حدد مستوى الدقة المطلوب (على سبيل المثال، عرض فترة موثوقية بنسبة 95% يبلغ ±3%).",
                "إنشاء مسارات جمع البيانات. لكل مقياس، حدد التوزيعات الأولية المبدئية (على سبيل المثال، $Beta(1,1)$ لغير المُعلِمة، أو اشتقها من المتوسطات التاريخية إذا كانت مستقرة وذات صلة).",
                "تطوير نماذج Bayesian (مثل Beta-Binomial) للمقاييس الأساسية. تنفيذ منطق تحديثات التوزيع اللاحق والقواعد الأولية لتحديد أحجام دفعات العينات اللاحقة.",
                "إنشاء لوحة معلومات لتصور التوزيعات اللاحقة، وفترات الموثوقية، والدقة المحققة مقابل الهدف، وتقدم أخذ العينات. إجراء دراسة تجريبية على نطاق محدود لاختبار سير العمل وأداء النموذج والمنطق التكيفي.",
                "التوسع التدريجي للنظام التكيفي عبر المزيد من مناطق المسح/مقدمي الخدمات. تنفيذ أو تحسين آليات التعامل مع عدم تجانس البيانات بمرور الوقت (مثل عوامل الخصم، ومراقبة نقاط التغيير).",
                "مراقبة أداء النظام باستمرار، وكفاءة الموارد، وجودة التقديرات. تحسين النماذج، والتوزيعات الأولية، والقواعد التكيفية بناءً على التعلم المستمر والتغذية الراجعة."
            ]
        },
        "note_to_practitioners_header": "٥. ملاحظات للممارسين",
        "note_to_practitioners_subheader_5.1": "٥.١. فوائد نهج Bayesian التكيفي",
        "note_to_practitioners_markdown_5.1_content": """
* **الكفاءة:** يوجه جهد أخذ العينات حيث تشتد الحاجة إليه، مما قد يقلل من أحجام العينات الإجمالية مقارنة بالطرق الثابتة مع تحقيق الدقة المطلوبة.
* **القدرة على التكيف:** يستجيب للبيانات الواردة، مما يجعله مناسبًا للبيئات الديناميكية حيث قد يتقلب الرضا أو حيث تكون المعرفة الأولية منخفضة.
* **الاستخدام الرسمي للمعرفة المسبقة:** يسمح بالدمج المنهجي للبيانات التاريخية أو رؤى الخبراء، والتي يمكن أن تكون مفيدة بشكل خاص مع البيانات الأولية المتفرقة للخدمات الجديدة أو المجموعات الفرعية المحددة.
* **قياس عدم اليقين بشكل بديهي:** توفر فترات الموثوقية تفسيرًا احتماليًا مباشرًا لنطاق المُعلمة، والذي يمكن أن يكون فهمه أسهل لأصحاب المصلحة من فترات الثقة التكرارية.
* **مخرجات غنية:** يوفر توزيعًا لاحقًا كاملاً لكل مُعلمة، مما يوفر رؤية أعمق من مجرد تقدير نقطي وفترة.
""",
        "note_to_practitioners_subheader_5.2": "٥.٢. القيود والاعتبارات",
        "note_to_practitioners_markdown_5.2_content": """
* **التعقيد:** يمكن أن تكون طرق Bayesian أكثر تطلبًا من الناحية المفاهيمية من الأساليب التكرارية التقليدية. يتطلب التنفيذ معرفة متخصصة.
* **اختيار التوزيع الأولي:** يمكن أن يؤثر اختيار التوزيع الأولي على النتائج اللاحقة، خاصة مع أحجام العينات الصغيرة. يتطلب هذا تبريرًا دقيقًا وشفافية. بينما تهدف التوزيعات الأولية "غير المُعلِمة" إلى تقليل هذا التأثير، فإن التوزيعات الأولية غير المُعلِمة حقًا ليست دائمًا واضحة ومباشرة.
* **التكلفة الحسابية:** بينما تعتبر نماذج Beta-Binomial بسيطة حسابيًا، فإن نماذج Bayesian الأكثر تعقيدًا (مثل النماذج الهرمية، النماذج التي تتطلب محاكاة MCMC) يمكن أن تكون كثيفة حسابيًا.
* **الاختلافات في التفسير:** يحتاج الممارسون المطلعون على الإحصاءات التكرارية إلى فهم التفسيرات المختلفة لمخرجات Bayesian (على سبيل المثال، فترات الموثوقية مقابل فترات الثقة).
* **تصور "الصندوق الأسود":** إذا لم يتم شرحها بوضوح، فقد يُنظر إلى الطبيعة التكيفية وحسابات Bayesian على أنها "صندوق أسود" من قبل أولئك غير المطلعين على الطرق. التواصل الواضح هو المفتاح.
""",
        "note_to_practitioners_subheader_5.3": "٥.٣. الافتراضات الرئيسية",
        "note_to_practitioners_markdown_5.3_content": """
* **تمثيلية العينات:** يُفترض أن كل دفعة من البيانات المجمعة تمثل المجتمع (الفرعي) محل الاهتمام *في تلك اللحظة الزمنية*. ستؤثر تحيزات أخذ العينات على صحة التقديرات.
* **ملاءمة النموذج:** يجب أن تعكس دالة الإمكان والتوزيعات الأولية المختارة بشكل معقول عملية توليد البيانات والمعرفة الحالية. بالنسبة لنسب الرضا، غالبًا ما يكون نموذج Beta-Binomial قويًا.
* **الاستقرار (أو التغيير المنمذج):** يُفترض أن المُعلمة الأساسية التي يتم قياسها (مثل معدل الرضا) مستقرة نسبيًا بين التحديثات التكرارية داخل موجة المسح، أو يتم نمذجة التغييرات بشكل صريح (على سبيل المثال، عبر عوامل الخصم أو النماذج الديناميكية). يمكن أن تكون التقلبات السريعة غير المنمذجة صعبة.
* **دقة البيانات:** يفترض أن الردود صادقة ومسجلة بدقة.
""",
        "note_to_practitioners_subheader_5.4": "٥.٤. توصيات عملية",
        "note_to_practitioners_markdown_5.4_content": """
* **ابدأ ببساطة:** ابدأ بمقاييس الرضا الأساسية والنماذج البسيطة (مثل Beta-Binomial). يمكن إضافة التعقيد بشكل متكرر مع اكتساب الخبرة.
* **استثمر في التدريب:** تأكد من أن الفريق المشارك في تنفيذ وتفسير النتائج لديه تدريب كافٍ في إحصاءات Bayesian.
* **الشفافية هي المفتاح:** وثق الخيارات المتعلقة بالتوزيعات الأولية والنماذج والقواعد التكيفية. قم بإجراء تحليلات الحساسية لفهم تأثير اختيارات التوزيعات الأولية المختلفة، خاصة في المراحل المبكرة أو مع بيانات محدودة.
* **المراجعة والتحقق المنتظم:** راجع أداء النماذج بشكل دوري. قارن تقديرات Bayesian بتلك من الطرق التقليدية إذا أمكن، خاصة خلال الفترة الانتقالية. تحقق من صحة الافتراضات.
* **التواصل مع أصحاب المصلحة:** طور طرقًا واضحة لتوصيل المنهجية وفوائدها وتفسير النتائج لأصحاب المصلحة الذين قد لا يكونون إحصائيين.
* **الاختبار التجريبي الشامل:** قبل التنفيذ على نطاق كامل، قم بإجراء دراسات تجريبية شاملة لتحسين العملية واختبار التكنولوجيا وتحديد التحديات غير المتوقعة.
""",
        "interactive_illustration_header": "٦. توضيح تفاعلي: نموذج Beta-Binomial",
        "interactive_illustration_markdown_intro": """
يقدم هذا القسم توضيحًا تفاعليًا بسيطًا لكيفية تحديث توزيع Beta الأولي إلى توزيع Beta لاحق ببيانات جديدة (دالة الإمكان Binomial). هذا هو جوهر تقدير نسبة (مثل معدل الرضا) بطريقة Bayesian.
""",
        "interactive_illustration_prior_beliefs_header": "المعتقدات الأولية",
        "interactive_illustration_prior_beliefs_markdown": "توزيع Beta $Beta(\\alpha, \\beta)$ هو توزيع أولي شائع للنسب. يمكن اعتبار $\\alpha$ بمثابة 'النجاحات' الأولية و $\\beta$ بمثابة 'الإخفاقات' الأولية. $Beta(1,1)$ هو توزيع أولي منتظم (غير مُعلِم).",
        "interactive_illustration_prior_alpha_label": "Prior Alpha (α₀)",
        "interactive_illustration_prior_beta_label": "Prior Beta (β₀)",
        "interactive_illustration_prior_mean_label": "المتوسط الأولي",
        "interactive_illustration_prior_ci_label": "فترة الموثوقية 95% (الأولية)",
        "interactive_illustration_width_label": "العرض",
        "interactive_illustration_new_data_header": "بيانات الاستطلاع الجديدة (دالة الإمكان)",
        "interactive_illustration_new_data_markdown": "أدخل النتائج من دفعة جديدة من الاستطلاعات.",
        "interactive_illustration_num_surveys_label": "عدد الاستطلاعات الجديدة (n)",
        "interactive_illustration_num_satisfied_label": "عدد الراضين في الاستطلاعات الجديدة (k)",
        "interactive_illustration_observed_satisfaction_label": "الرضا الملاحظ في البيانات الجديدة",
        "interactive_illustration_posterior_beliefs_header": "المعتقدات اللاحقة (بعد التحديث)",
        "interactive_illustration_posterior_markdown": "التوزيع اللاحق هو $Beta(\\alpha_0 + k, \\beta_0 + n - k) = Beta({posterior_alpha:.1f}, {posterior_beta:.1f})$",
        "interactive_illustration_posterior_mean_label": "المتوسط اللاحق",
        "interactive_illustration_posterior_ci_label": "فترة الموثوقية 95% (اللاحقة)",
        "interactive_illustration_target_width_label": "عرض فترة الموثوقية المستهدف للإيقاف",
        "interactive_illustration_success_message": "تم تحقيق الدقة المستهدفة! العرض الحالي ({current_width:.3f}) ≤ العرض المستهدف ({target_width:.3f}).",
        "interactive_illustration_warning_message": "لم يتم تحقيق الدقة المستهدفة بعد. العرض الحالي ({current_width:.3f}) > العرض المستهدف ({target_width:.3f}). ضع في اعتبارك المزيد من العينات.",
        "interactive_illustration_plot1_title": "التوزيعات الأولية واللاحقة لمعدل الرضا",
        "interactive_illustration_plot_xlabel": "معدل الرضا (θ)",
        "interactive_illustration_plot_ylabel": "الكثافة",
        "interactive_illustration_discounting_header": "توضيح مفاهيمي: تأثير خصم البيانات الأقدم",
        "interactive_illustration_discounting_markdown": """
يوضح هذا كيف يمكن لعامل الخصم أن يغير تأثير بيانات التوزيع اللاحق 'القديمة' عند استخدامها لتشكيل توزيع أولي لفترة 'جديدة'.
افترض أن 'التوزيع اللاحق' المحسوب أعلاه هو الآن 'بيانات قديمة' من فترة سابقة.
نريد تكوين توزيع أولي جديد للفترة القادمة.
يمثل 'التوزيع الأولي المبدئي' (على سبيل المثال، $Beta(1,1)$) اعتقادًا أساسيًا أقل إفادة.
""",
        "interactive_illustration_discount_factor_label": "عامل الخصم (δ) للبيانات القديمة",
        "interactive_illustration_discount_factor_help": "يتحكم في وزن البيانات القديمة. 1.0 = الوزن الكامل، 0.0 = تجاهل البيانات القديمة، والاعتماد فقط على التوزيع الأولي المبدئي.",
        "interactive_illustration_initial_prior_alpha_discount_label": "Initial Prior Alpha (لفترة جديدة إذا كان الخصم كبيرًا)",
        "interactive_illustration_initial_prior_beta_discount_label": "Initial Prior Beta (لفترة جديدة إذا كان الخصم كبيرًا)",
        "interactive_illustration_new_prior_discount_label": "التوزيع الأولي الجديد للفترة التالية: $Beta({new_prior_alpha:.2f}, {new_prior_beta:.2f})$",
        "interactive_illustration_new_prior_mean_discount_label": "متوسط التوزيع الأولي الجديد",
        "interactive_illustration_plot2_title": "تكوين توزيع أولي جديد مع الخصم",
        "conclusion_header": "٧. الخاتمة",
        "conclusion_markdown": """
يقدم إطار تقدير Bayesian التكيفي المقترح نهجًا متطورًا ومرنًا وفعالًا لتحليل استطلاعات رضا الحجاج. من خلال تحديث المعتقدات بشكل متكرر وتعديل جهود أخذ العينات ديناميكيًا، تعد هذه المنهجية بتقديم رؤى أكثر دقة وفي الوقت المناسب، مما يتيح اتخاذ قرارات أفضل استنارة لتعزيز تجربة الحج.

في حين أنه يقدم مفاهيم جديدة ويتطلب تنفيذًا دقيقًا، فإن الفوائد طويلة الأجل - بما في ذلك الاستخدام الأمثل للموارد والفهم الأعمق لديناميكيات الرضا - كبيرة. يدعو هذا المقترح إلى تنفيذ مرحلي، يبدأ بالمقاييس الأساسية ويبني تدريجيًا التعقيد والنطاق.

نوصي بالمضي قدمًا في مشروع تجريبي لإثبات الفوائد العملية وتحسين الجوانب التشغيلية لهذا النهج التحليلي المتقدم.
"""
    }
}

# --- Proposal Content Functions ---

def introduction_objectives(lang='en'):
    st.header(translations[lang]["introduction_objectives_header"])
    st.markdown(translations[lang]['introduction_objectives_markdown_1'])
    st.subheader(translations[lang]["introduction_objectives_subheader_1.1"])
    st.markdown(translations[lang]['introduction_objectives_markdown_1.1_content'])

def challenges_addressed(lang='en'):
    st.header(translations[lang]["challenges_addressed_header"])
    st.markdown(translations[lang]['challenges_addressed_markdown'])

def bayesian_adaptive_methodology(lang='en'):
    st.header(translations[lang]["bayesian_adaptive_methodology_header"])
    st.markdown(translations[lang]['bayesian_adaptive_methodology_markdown_intro'])
    st.subheader(translations[lang]["bayesian_adaptive_methodology_subheader_3.1"])
    st.markdown(translations[lang]['bayesian_adaptive_methodology_markdown_3.1_content'])
    st.subheader(translations[lang]["bayesian_adaptive_methodology_subheader_3.2"])
    st.markdown(translations[lang]['bayesian_adaptive_methodology_markdown_3.2_content'])
    st.image("https_miro.medium.com_v2_resize_fit_1400_1__f_xL41kP9n2_n3L9yY0gLg.png", caption=translations[lang]["bayesian_adaptive_methodology_image_caption"])
    st.subheader(translations[lang]["bayesian_adaptive_methodology_subheader_3.3"])
    st.markdown(translations[lang]['bayesian_adaptive_methodology_markdown_3.3_content'])
    st.subheader(translations[lang]["bayesian_adaptive_methodology_subheader_3.4"])
    st.markdown(translations[lang]['bayesian_adaptive_methodology_markdown_3.4_content'])
    st.subheader(translations[lang]["bayesian_adaptive_methodology_subheader_3.5"])
    st.markdown(translations[lang]['bayesian_adaptive_methodology_markdown_3.5_content'])

def implementation_roadmap(lang='en'):
    st.header(translations[lang]["implementation_roadmap_header"])
    st.markdown(translations[lang]['implementation_roadmap_markdown'])

    df_data_en = translations["en"]["roadmap_data"]
    df_data_ar = translations["ar"]["roadmap_data"]

    if lang == 'ar':
        df_roadmap = pd.DataFrame({
            translations[lang]["roadmap_df_phase_col"]: df_data_ar["Phase"],
            translations[lang]["roadmap_df_step_col"]: df_data_ar["Step"],
            translations[lang]["roadmap_df_desc_col"]: df_data_ar["Description"]
        })
    else: # lang == 'en'
        df_roadmap = pd.DataFrame({
            translations[lang]["roadmap_df_phase_col"]: df_data_en["Phase"],
            translations[lang]["roadmap_df_step_col"]: df_data_en["Step"],
            translations[lang]["roadmap_df_desc_col"]: df_data_en["Description"]
        })
    st.dataframe(df_roadmap, hide_index=True, use_container_width=True)


def note_to_practitioners(lang='en'):
    st.header(translations[lang]["note_to_practitioners_header"])
    st.subheader(translations[lang]["note_to_practitioners_subheader_5.1"])
    st.markdown(translations[lang]['note_to_practitioners_markdown_5.1_content'])
    st.subheader(translations[lang]["note_to_practitioners_subheader_5.2"])
    st.markdown(translations[lang]['note_to_practitioners_markdown_5.2_content'])
    st.subheader(translations[lang]["note_to_practitioners_subheader_5.3"])
    st.markdown(translations[lang]['note_to_practitioners_markdown_5.3_content'])
    st.subheader(translations[lang]["note_to_practitioners_subheader_5.4"])
    st.markdown(translations[lang]['note_to_practitioners_markdown_5.4_content'])

def interactive_illustration(lang='en'):
    st.header(translations[lang]["interactive_illustration_header"])
    st.markdown(translations[lang]['interactive_illustration_markdown_intro'])

    st.markdown("---")
    col1, col2 = st.columns(2)
    key_suffix = "_" + lang

    with col1:
        st.subheader(translations[lang]["interactive_illustration_prior_beliefs_header"])
        st.markdown(translations[lang]['interactive_illustration_prior_beliefs_markdown'])
        prior_alpha = st.slider(translations[lang]["interactive_illustration_prior_alpha_label"], min_value=0.1, max_value=50.0, value=1.0, step=0.1, key="prior_a" + key_suffix)
        prior_beta = st.slider(translations[lang]["interactive_illustration_prior_beta_label"], min_value=0.1, max_value=50.0, value=1.0, step=0.1, key="prior_b" + key_suffix)

        if prior_alpha > 0 and prior_beta > 0:
            prior_mean = prior_alpha / (prior_alpha + prior_beta)
            st.write(f"{translations[lang]['interactive_illustration_prior_mean_label']}: {prior_mean:.3f}")
            prior_ci = get_credible_interval(prior_alpha, prior_beta)
            st.write(f"{translations[lang]['interactive_illustration_prior_ci_label']}: [{prior_ci[0]:.3f}, {prior_ci[1]:.3f}], {translations[lang]['interactive_illustration_width_label']}: {prior_ci[1]-prior_ci[0]:.3f}")
        else:
            st.warning("Alpha and Beta for prior must be > 0." if lang == 'en' else "يجب أن تكون Alpha و Beta للتوزيع الأولي أكبر من 0.")
            prior_ci = (0,0)

    with col2:
        st.subheader(translations[lang]["interactive_illustration_new_data_header"])
        st.markdown(translations[lang]['interactive_illustration_new_data_markdown'])
        num_surveys = st.slider(translations[lang]["interactive_illustration_num_surveys_label"], min_value=1, max_value=500, value=50, step=1, key="surveys_n" + key_suffix)
        num_satisfied = st.slider(translations[lang]["interactive_illustration_num_satisfied_label"], min_value=0, max_value=num_surveys, value=int(num_surveys/2), step=1, key="surveys_k" + key_suffix)
        num_not_satisfied = num_surveys - num_satisfied
        st.write(f"{translations[lang]['interactive_illustration_observed_satisfaction_label']}: {num_satisfied/num_surveys if num_surveys > 0 else 0:.3f}")

    st.markdown("---")
    st.subheader(translations[lang]["interactive_illustration_posterior_beliefs_header"])

    if prior_alpha > 0 and prior_beta > 0 :
        posterior_alpha, posterior_beta = update_beta_parameters(prior_alpha, prior_beta, num_satisfied, num_not_satisfied)
        posterior_markdown_text = translations[lang]["interactive_illustration_posterior_markdown"]
        try:
            st.markdown(posterior_markdown_text.format(posterior_alpha=posterior_alpha, posterior_beta=posterior_beta))
        except KeyError: # Fallback if placeholders are missing/mismatched
            st.markdown(f"The posterior distribution is $Beta(\\alpha_0 + k, \\beta_0 + n - k) = Beta({posterior_alpha:.1f}, {posterior_beta:.1f})$")

        posterior_mean = posterior_alpha / (posterior_alpha + posterior_beta)
        st.write(f"{translations[lang]['interactive_illustration_posterior_mean_label']}: {posterior_mean:.3f}")
        posterior_ci = get_credible_interval(posterior_alpha, posterior_beta)
        st.write(f"{translations[lang]['interactive_illustration_posterior_ci_label']}: [{posterior_ci[0]:.3f}, {posterior_ci[1]:.3f}], {translations[lang]['interactive_illustration_width_label']}: {posterior_ci[1]-posterior_ci[0]:.3f}")

        target_width = st.number_input(translations[lang]["interactive_illustration_target_width_label"], min_value=0.01, max_value=1.0, value=0.10, step=0.01, key="target_width" + key_suffix)
        current_width = posterior_ci[1] - posterior_ci[0]

        if posterior_alpha <=0 or posterior_beta <=0:
            current_width = float('inf')

        if current_width <= target_width and current_width > 0 :
            st.success(translations[lang]["interactive_illustration_success_message"].format(current_width=current_width, target_width=target_width))
        else:
            st.warning(translations[lang]["interactive_illustration_warning_message"].format(current_width=current_width, target_width=target_width))

        fig, ax = plt.subplots()
        plot_beta_distribution(prior_alpha, prior_beta, "Prior", ax, lang)
        plot_beta_distribution(posterior_alpha, posterior_beta, "Posterior", ax, lang)
        ax.set_title(translations[lang]["interactive_illustration_plot1_title"])
        ax.set_xlabel(translations[lang]["interactive_illustration_plot_xlabel"])
        ax.set_ylabel(translations[lang]["interactive_illustration_plot_ylabel"])
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
    else:
        st.error("Cannot calculate posterior due to invalid prior parameters." if lang == 'en' else "لا يمكن حساب التوزيع اللاحق بسبب معلمات أولية غير صالحة.")
        posterior_alpha, posterior_beta = 1.0, 1.0

    st.markdown("---")
    st.subheader(translations[lang]["interactive_illustration_discounting_header"])
    st.markdown(translations[lang]['interactive_illustration_discounting_markdown'])

    old_posterior_alpha = posterior_alpha
    old_posterior_beta = posterior_beta

    discount_factor = st.slider(translations[lang]["interactive_illustration_discount_factor_label"], min_value=0.0, max_value=1.0, value=0.8, step=0.05,
                                help=translations[lang]["interactive_illustration_discount_factor_help"], key="discount_factor" + key_suffix)
    initial_prior_alpha_discount = st.number_input(translations[lang]["interactive_illustration_initial_prior_alpha_discount_label"], min_value=0.1, value=1.0, step=0.1, key="init_prior_a_disc" + key_suffix)
    initial_prior_beta_discount = st.number_input(translations[lang]["interactive_illustration_initial_prior_beta_discount_label"], min_value=0.1, value=1.0, step=0.1, key="init_prior_b_disc" + key_suffix)

    new_prior_alpha = discount_factor * old_posterior_alpha + (1 - discount_factor) * initial_prior_alpha_discount
    new_prior_beta = discount_factor * old_posterior_beta + (1 - discount_factor) * initial_prior_beta_discount

    new_prior_text_template = translations[lang]["interactive_illustration_new_prior_discount_label"]
    try:
        st.write(new_prior_text_template.format(new_prior_alpha=new_prior_alpha, new_prior_beta=new_prior_beta))
    except KeyError: # Fallback
        st.write(f"New Prior for Next Period: $Beta({new_prior_alpha:.2f}, {new_prior_beta:.2f})$")

    if new_prior_alpha > 0 and new_prior_beta > 0:
        new_prior_mean = new_prior_alpha / (new_prior_alpha + new_prior_beta)
        st.write(f"{translations[lang]['interactive_illustration_new_prior_mean_discount_label']}: {new_prior_mean:.3f}")

        fig2, ax2 = plt.subplots()
        plot_beta_distribution(old_posterior_alpha, old_posterior_beta, "Old Posterior (Data from T-1)", ax2, lang)
        plot_beta_distribution(initial_prior_alpha_discount, initial_prior_beta_discount, "Fixed Initial Prior", ax2, lang)
        plot_beta_distribution(new_prior_alpha, new_prior_beta, f"New Prior (δ={discount_factor})", ax2, lang)
        ax2.set_title(translations[lang]["interactive_illustration_plot2_title"])
        ax2.set_xlabel(translations[lang]["interactive_illustration_plot_xlabel"])
        ax2.set_ylabel(translations[lang]["interactive_illustration_plot_ylabel"])
        ax2.legend()
        ax2.grid(True)
        st.pyplot(fig2)
    else:
        st.error("Cannot plot new prior due to invalid alpha/beta values from discounting." if lang == 'en' else "لا يمكن رسم التوزيع الأولي الجديد بسبب قيم alpha/beta غير صالحة ناتجة عن الخصم.")

def conclusion(lang='en'):
    st.header(translations[lang]["conclusion_header"])
    st.markdown(translations[lang]['conclusion_markdown'])

# --- Streamlit App Structure ---

language_options = {"English": "en", "العربية": "ar"}
selected_language_display = st.sidebar.radio("Language / اللغة", list(language_options.keys()), index=0)
current_lang = language_options[selected_language_display]

if current_lang == 'ar':
    st.markdown("""
        <style>
            /* More robust RTL styling */
            body, .main, .stApp, .block-container {
                direction: rtl !important;
                text-align: right !important;
            }
            /* Sidebar specific styling */
            div[data-testid="stSidebarNav"] { /* Target Streamlit's new sidebar navigation */
                 direction: rtl !important; text-align: right !important;
            }
            div[data-testid="stSidebarNav"] li a { /* Sidebar links */
                 text-align: right !important; display: block; /* Ensure full width for text align */
            }
            .stRadio > label {
                display: flex;
                flex-direction: row-reverse;
                margin-left: auto; /* Push to the right */
                margin-right: 0;
                padding-right: 0; /* Remove any accidental left padding */
                justify-content: flex-end;
                width: 100%; /* Ensure it takes full width to align */
            }
            .stRadio > label > div:first-child { /* The radio button input */
                 margin-right: 0 !important;
                 margin-left: 0.5rem;
            }
             /* Main content headers, paragraphs, lists */
            h1, h2, h3, h4, h5, h6, p, li, ul, ol, label, div[data-testid="stText"], div[data-testid="stMarkdown"] {
                text-align: right !important;
                direction: rtl !important;
            }
            ul, ol {
              padding-right: 30px !important; /* More pronounced padding for list markers */
              padding-left: 0 !important;
              margin-right: 0; /* Reset potential browser defaults */
            }
            li {
              margin-right: 10px !important; /* Indent list items a bit */
              margin-left: 0 !important;
            }
            /* Dataframe text alignment */
            .dataframe th, .dataframe td {
                text-align: right !important;
                direction: rtl !important;
            }
            /* Streamlit buttons */
            .stButton > button {
                direction: rtl; /* Text inside button */
                text-align: right;
            }
            /* Ensure markdown containers and their children are strictly RTL */
            div[data-testid="stMarkdownContainer"], div[data-testid="stMarkdown"] {
                direction: rtl !important;
                text-align: right !important;
            }
            div[data-testid="stMarkdownContainer"] *, div[data-testid="stMarkdown"] * {
                text-align: right !important;
                direction: rtl !important;
            }
            /* Correct alignment for expander headers */
            .st-expanderHeader {
                direction: rtl !important;
                text-align: right !important;
            }
            /* Slider labels */
            .stSlider label {
                 direction: rtl !important;
                 text-align: right !important;
                 width: 100% !important; /* Make label take full width to align text right */
                 display: block !important;
            }
        </style>
    """, unsafe_allow_html=True)

st.title(translations[current_lang]["page_title"])

PAGES_FUNCTIONS = {
    "1. Introduction & Objectives": introduction_objectives,
    "2. Challenges Addressed": challenges_addressed,
    "3. Bayesian Adaptive Methodology": bayesian_adaptive_methodology,
    "4. Implementation Roadmap": implementation_roadmap,
    "5. Note to Practitioners": note_to_practitioners,
    "6. Interactive Illustration": interactive_illustration,
    "7. Conclusion": conclusion
}

page_display_names = [translations[current_lang]["sections"][en_key] for en_key in PAGES_FUNCTIONS.keys()]
display_to_en_key_map = {translations[current_lang]["sections"][en_key]: en_key for en_key in PAGES_FUNCTIONS.keys()}

st.sidebar.title(translations[current_lang]["sidebar_title"])
selected_display_name = st.sidebar.radio(translations[current_lang]["go_to"], page_display_names, key="page_selector_" + current_lang)

selected_en_key = display_to_en_key_map[selected_display_name]
page_function_to_call = PAGES_FUNCTIONS[selected_en_key]
page_function_to_call(lang=current_lang)

st.sidebar.markdown("---")
st.sidebar.info(translations[current_lang]["sidebar_info"])
