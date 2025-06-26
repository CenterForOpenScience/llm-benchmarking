# Task Design Structure for Extract (Easy)

## Objective
Extract key replication components from a research paper and its associated claim/hypotheses to create a structured preregistration document for transparency and fidelity to the original study.

## Input
- **Paper + Claim to Replicate**:
  - The original research paper.
  - The specific claim targeted for replication, which can be:
    - A human-provided statement in their own words.
    - A verbatim statement pulled by a human from the paper’s abstract or full text.
    - Directly from SCORE as "claim 2" (the claim from the abstract).

- **Hypotheses (Claim 3)**:
  - The hypotheses tied to the claim, labeled as "claim 3" in the SCORE framework, extracted from the provided materials and specifying the expected relationships or effects to be tested.

- **Prereg template**

## Output (in Prereg Format)
A structured preregistration document with the following components:

1. **Hypotheses**:
   - A clear restatement of the focal hypothesis (H*) to be tested in the replication, extracted from the provided materials (e.g., the paper or SCORE framework).

2. **Study Design**:
   - **Data**:
     - Source (e.g., survey, database)
     - Specific waves or subsets (if applicable)
     - Sample size
     - Unit of analysis (e.g., individual, household)
     - Access details (e.g., restrictions, request process)
   - **Methods**:
     - Study type (e.g., observational, experimental)
     - Model or statistical approach (e.g., regression type)
     - Outcome variable
     - Focal independent variable(s) (IV)
     - Control variable(s)
     - Tools or software specifics (e.g., R packages)

3. **Expected Results/Evidence (Claim 4)**:
   - Key findings from the original study supporting the claim (labeled "claim 4" in the SCORE framework), including statistical results (e.g., coefficients, p-values) and the direction/significance of the effect.

4. **Additional Notes (if applicable)**:
   - Clarifications or deviations, such as variable recoding, handling of missing data, or inferred details due to under-specification in the original paper.

## Example Using Case Study #2: Baxter et al. (2015)

### Study Reference
- **Original Study**: Baxter, J., Buchler, S., Perales, F., & Western, M. (2015). A Life-Changing Event: First Births and Men’s and Women’s Attitudes to Mothering and Gender Divisions of Labor. *Social Forces*, 93(3), 989–1014.
- **SCORE ID**: 0056
- **Replication Team**: Duan et al.
- **Domain**: Sociology / Social Psychology
- **Study Type**: Observational
- **OSF Project**: https://osf.io/u5r47

### Input
- **Paper + Claim to Replicate**:
  - **Paper**: Baxter et al. (2015) explores how parenthood affects gender role attitudes.
  - **Main Claim**: Among men, becoming a parent leads to more traditional attitudes toward gender roles—specifically, less support for equal sharing of housework and childcare. (This could be a human-provided statement, verbatim from the paper, or "claim 2" from SCORE.)

- **Hypotheses (Claim 3)**:
  - Focal Hypothesis (H*): After becoming fathers, men will be less likely to agree that men and women in dual-earner couples should share housework and childcare equally. (Extracted from the SCORE framework or paper.)

### Output (in Prereg Format)
1. **Hypotheses**:
   - After becoming fathers, men will be less likely to agree that men and women in dual-earner couples should share housework and childcare equally.

2. **Study Design**:
   - **Data**:
     - Source: Household, Income and Labour Dynamics in Australia (HILDA) survey
     - Waves Used: 1, 5, 8, 11, and 15
     - Sample Size: 46,488 observations from 19,983 individuals
     - Unit of Analysis: Individual (ages 18–50) across 5 waves
     - Access: Restricted; requires data access request via ADA Australia
   - **Methods**:
     - Study Type: Observational, longitudinal panel
     - Model: Fixed-effects regression with two-way (time + individual) effects, implemented using the `plm` package in R with a "within" (fixed effects) + time fixed effects model
     - Outcome Variable: Recoded 7-point Likert-scale agreement with: "If both partners work, they should share equally in housework and care" (higher values = more traditional attitudes)
     - Focal IV: Binary indicator for "Male & had 1st birth" (parenthood x male interaction)
     - Controls: Female parenthood, age, marital status, education, religion (with missing flag), years since 1st birth

3. **Expected Results/Evidence (Claim 4)**:
   - Key Finding: The focal hypothesis (H*) was confirmed with a positive and statistically significant coefficient for "Male & had 1st birth":
     - With religion: b = 0.1510, p = 0.0002
     - Without religion: b = 0.1496, p = 0.0003
   - Conclusion: Becoming a father leads men to adopt more traditional gender attitudes, fully supporting the original claim and SCORE focal test.

4. **Additional Notes**:
   - Outcome variable recoded so higher values indicate more traditional attitudes (original scale reversed).
   - Religious variable available only in Wave 1; fixed value and missing flag applied.
   - Controls (e.g., education, marital status) inferred due to under-specification in the original paper.
   - Robustness checks (e.g., religion imputation) confirmed the focal effect’s stability.