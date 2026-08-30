# AI Use Disclosure

This repository documents the capstone project for the Imperial College London
Professional Certificate in ML & AI completed by Srini Rajasekaran.

## How AI Was Used

An AI tool (Claude, Anthropic) was used throughout Stage 2 as an implementation
resource. It wrote and ran Python code for GP fitting, acquisition functions, and
candidate generation; produced evidence workbooks, dashboards, and reflection
drafts; and applied corrections when errors were identified in review.

The working model throughout was that of a practitioner directing a developer.
Specifications, challenges, and approval of all outputs came from the project
owner. This mirrors the relationship between a senior model risk officer and a
quantitative analyst: the analyst implements; the validator specifies requirements,
reviews methodology, identifies gaps, and decides what meets the standard before
anything is used or submitted.

## How the Methodology Evolved

The campaign methodology was not fixed at the start. It developed week by week
through directions from the project owner, often drawing on two parallel sources:
the progression of the course curriculum and a 25-year professional background
in derivatives pricing and model risk.

**Regression as the starting challenger.** The regression framework was introduced
in the first session as the primary decision tool: fit OLS on available data,
act only on statistically significant coefficients, and use that to set the query
direction. This was a deliberate starting point - transparent, auditable, and
consistent with the linear model foundations covered in the early course modules.

**Gaussian Process surrogate.** The GP was brought in as the regression challenger
and gradually became the primary method as data accumulated and linear assumptions
broke down. Kernel selection began with RBF as the default. The project owner
subsequently directed a systematic comparison across RBF and Matern variants,
recognising that different decay structures would matter as the functions revealed
non-stationary or rough surface behaviour. This led to the C2 Kernel Challenger
becoming a mandatory weekly step.

**Non-linearity and collinearity diagnostics.** As the course progressed through
more advanced modelling techniques, the project owner directed the addition of
curvature testing and collinearity checks to the diagnostic battery. These were
introduced in direct alignment with the course curriculum and grounded in the
understanding that linear correlation alone would miss the structural signals
present in higher-dimensional functions.

**Ensemble and bootstrap methods.** The C3 bootstrap ensemble was introduced as
a challenger to the single GP, generating 30 resampled GP posteriors to test
whether the primary surrogate's recommendation was stable or an artefact of a
particular data configuration. This reflected the same logic applied in model
risk practice when stress-testing a model's sensitivity to its calibration sample.

**PCA-style variance decomposition.** Directional analysis of the input space
was introduced to identify which combinations of dimensions explained the most
variation in the output, providing a cross-check on the individual correlation
signals. This was specified in the same methodological spirit as PCA in
risk factor decomposition for structured products.

**Derivatives pricing analogies applied directly.**
Several structural additions came directly from professional derivatives pricing
practice rather than from the course curriculum:

- *Sobol low-discrepancy sequences* were chosen as the primary candidate generator
  because they are the standard tool for quasi-Monte Carlo integration across
  multi-dimensional spaces in derivatives model risk. Uniform random sampling
  and Latin Hypercube were tested and retained only as comparisons.
- *Scenario matrix analysis* holding one dimension fixed and sweeping others
  through a surrogate was drawn from volatility surface construction, where a
  pricing model is used to infer prices at combinations of strike and tenor never
  directly quoted. The GP played the role of the pricing model; the Sobol grid
  played the role of the scenario ladder.
- *Four-corners boundary testing* was introduced as an extreme scenario probe,
  directly analogous to the four-scenario stress grids used in structured product
  risk management to map the edges of the payoff surface before examining the
  interior.
- *Trust region constraints* were framed as domain-of-model-validity bounds: a
  GP surrogate, like a pricing model, should not be trusted far outside the range
  of observations it was calibrated on. Trust regions operationalised this by
  restricting candidate search to the neighbourhood of confirmed observations.
- *Gaussian mixture GP models* were explored as an extension, drawing on the
  use of Gaussian mixture models in volatility surface parameterisation, where
  multi-modal distributions are used to capture regime structure that a single
  Gaussian cannot represent.
- *Local versus global optimum search* and the distinction between grid-based
  and surrogate-guided assessment came from the derivatives practice of running
  both a fast grid pricer for broad coverage and a slower full model for local
  refinement around identified risk concentrations.
- *Stratified topographic assessment* dividing the input space into hypercube
  sub-regions and ranking local peaks was introduced to guard against the same
  risk that arises in derivatives: a model optimised globally may be sitting in
  a locally attractive but globally suboptimal region. The stratification made
  that structure visible.

**Astro-shot probes.** At specific points in the campaign, deliberately aggressive
boundary pushes were directed as high-risk, high-information probes - testing
whether the function had structure beyond the currently observed range. These were
not surrogate-guided; they were deliberate exploratory shots into unsampled
territory, accepted as potentially returning noise but valued for the structural
information a strong result or a flat return would provide.

**Submission decisions and review.** Every candidate coordinate was reviewed
before portal submission. Candidates failing the pre-submission audit were
rejected. All override decisions were classified and logged. Anomalies in the
register were identified during review, not flagged by the AI. Reflection
narratives were directed each week: what findings to lead with, what framing
accurately represented the executed strategy, and what to remove.

## Imperial College London Confirmation

Imperial College London confirmed via support ticket 2821471 (June 2026,
Pooja Pharalay, Programme Support Team) that AI use is permitted provided it
supports the student's own work and learning rather than replacing it, and that
a single disclosure statement at final submission is sufficient.
