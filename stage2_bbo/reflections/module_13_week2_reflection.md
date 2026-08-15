# Module 13.1 — Week 2 Reflection

*Posted to Imperial discussion board. This file is the GitHub record of the reflection submitted for Module 13, Week 2 of the BBO challenge.*

---

Switched from uniform random to Latin Hypercube Sampling after identifying uneven coverage. Fixed shared-seed artefact (F1/F2 had used seed=42, producing identical queries). F6 corrected from Regression to GP-UCB after the fourth gate (predicted y exceeds current best) was found to have been omitted from live analysis. Six of eight improved. F8 regressed: regression passed all gates and predicted 10.38; actual result was 9.651 -- first live disconfirmation of the four-gate framework, logged as an error.

---

*Note: This record captures the key decisions, results, and reasoning for this round as submitted to the Imperial discussion board.*
