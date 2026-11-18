# Interview notes: how to explain this project

## 1. The 30-second pitch
"I built a retail credit risk model on 3.26 lakh real Lending Club loans. First I made an application scorecard
that gives every borrower a probability of default (PD). Then I validated it on a later year (2014) that the model
had never seen. Finally I converted the PDs into IFRS 9 expected credit loss and Basel IRB capital, under a normal
scenario and a stress scenario. The scorecard ranked risk well out of time: default rate goes from 29% in the
lowest band to 4% in the highest, and its Gini of 0.325 beats Lending Club's own grade. After recalibration, the
expected loss came within 2% of the loss that actually happened."

## 2. Key ideas and the maths

**PD, LGD, EAD.** Expected Loss = PD × LGD × EAD.
- **PD**: chance the borrower defaults.
- **LGD**: share of the exposure that is lost after recoveries.
- **EAD**: amount outstanding at the time of default.

Example: PD 10%, LGD 90%, EAD ₹1,00,000 → EL = 0.10 × 0.90 × 1,00,000 = ₹9,000.

**Why 36-month loans only?** Every loan issued up to 2014 had finished its term by the 2017 data snapshot, so we
know for sure whether it defaulted. With 60-month loans, many would still be running and the outcome would not
be known yet.

**Weight of Evidence (WoE).** For each bin: WoE = ln(%Good in bin / %Bad in bin).
- Positive WoE means the bin is safer than average; negative means riskier.
- Example: income above $100K has WoE +0.52 (bad rate 7.8%); income below $25.5K has WoE −0.44 (bad rate 18.2%).
- Why use WoE?
  - It makes the relation with log-odds linear.
  - It handles missing values as their own bin.
  - It removes the effect of outliers.
  - The final model is easy to explain to business and to the regulator.

**Information Value (IV).** IV = Σ (%Good − %Bad) × WoE.
- Rule of thumb: < 0.02 useless, 0.02–0.1 weak, 0.1–0.3 medium, > 0.3 strong.
- My strongest variable is annual income with IV 0.101, then unused credit-card limit (0.088).
- The IVs are on the weak side because this is an already-approved population and the public file has no bureau score.

**Monotonic binning.** I merged neighbouring bins until the bad rate moved in only one direction. Reason: a
scorecard where a higher DTI sometimes gives more points is not acceptable to the business or the regulator.

**Logistic regression.** ln(PD / (1 − PD)) = α + Σ βᵢ·WoEᵢ.
- Because of the WoE convention, every β must be negative. I checked this, and also that every p-value is
  below 0.05.
- If a variable's sign flips, that usually signals multicollinearity, so I also removed variables whose WoEs
  were correlated above 0.6.

**Score scaling.**
- Factor = PDO / ln 2 = 20 / 0.693 = 28.85.
- Offset = 600 − 28.85 × ln 50 = 487.1.
- Score = Offset + Factor × ln(odds good:bad).
- So 600 points means odds of 50:1, and every 20 more points doubles the odds.
- Points for one attribute = −(βᵢ·WoEᵢⱼ + α/n) × Factor + Offset/n, where n = number of variables.

## 3. Validation: what each number means
- **Gini = 2·AUC − 1.**
  - Measures how well the model separates goods from bads; 0 = random, 1 = perfect.
  - Mine is 0.325 on 2014, almost the same as in training (0.330), so there is no overfitting.
- **KS.** The maximum gap between the cumulative distributions of goods and bads. Mine is 0.19.
- **PSI = Σ (Actual% − Expected%) × ln(Actual% / Expected%)** across score bands.
  - Thresholds: < 0.10 stable, 0.10–0.25 monitor, > 0.25 investigate.
  - Mine is 0.003: the 2014 applicants looked like the training applicants.
- **Calibration.** Do predicted PDs match actual default rates?
  - In-time: yes.
  - 2014: predicted 12.5% but actual 14.6%, and the Hosmer–Lemeshow test rejects.
- **Main talking point:** PSI was fine and ranking was fine, but calibration was off. So the borrowers did not
  change; the environment did, because the 2014 vintage was riskier. The fix is recalibration: re-estimate only the
  intercept (+0.188 on log-odds) and keep the scorecard. Rebuilding the whole model is not needed.
- **Benchmark.** Lending Club's own sub-grade has Gini 0.319 on 2014; my scorecard has 0.325. They also had
  the FICO score, which is not in the public data.
- **Challenger.** A LightGBM model on the same features reaches 0.357, but it overfits (0.62 on its own training
  data) and cannot be turned into a points table. So the scorecard is the champion and LightGBM is the benchmark.

## 4. IFRS 9 ECL
- **Stage 1** (performing): 12-month ECL.
- **Stage 2** (significant increase in credit risk, SICR): lifetime ECL.
- **Stage 3** (credit-impaired): lifetime ECL, with interest income on the net amount.
- Common SICR triggers: 30 days past due (backstop), a big jump in PD since origination, watch-list.
- **12-month PD:** 22.6% of all defaults historically happened in year 1, so PD₁₂ = 0.226 × lifetime PD.
- **EAD changes over time.** These loans amortise: at a first-year default about 89% of the loan is still
  outstanding, but averaged over the whole life it is 59%.
- **Discounting:** at the effective interest rate (the loan's own rate), per IFRS 9.
- **Result on the 2014 book** ($1.93 bn): Stage 1 ECL $44.0 M (2.3%), lifetime $114.9 M (6.0%), both with the
  recalibrated PD. The jump from Stage 1 to Stage 2 is why staging matters so much for a bank's P&L.
- **IFRS 9 vs Basel:**
  - IFRS 9 is point-in-time and forward-looking; it uses macro scenarios, probability-weighted.
  - Basel IRB PD is through-the-cycle, a long-run average.
  - IFRS 9 has no regulatory floors; Basel has floors and needs a downturn LGD.
- **India context (very relevant for KPMG FRM):**
  - On 27 April 2026, RBI issued final directions moving scheduled commercial banks from incurred-loss
    provisioning to a forward-looking ECL framework with Stage 1/2/3, effective **1 April 2027**. Draft was
    October 2025.
  - The 90-day NPA rule stays, and there are prudential floors on provisions.
  - Banks are now building PD/LGD/EAD models for this, which is exactly the kind of work FRM consulting teams do.
  - Ind AS 109 is India's version of IFRS 9 (already used by NBFCs).

## 5. Basel IRB capital
- Capital covers **unexpected** loss; provisions (ECL) cover **expected** loss.
- Formula (other retail):
  - R = 0.03·w + 0.16·(1 − w), where w = (1 − e^(−35·PD)) / (1 − e^(−35))
  - K = LGD × N[ (G(PD) + √R · G(0.999)) / √(1 − R) ] − PD × LGD
  - RWA = 12.5 × K × EAD
  - N = normal CDF; G = inverse normal CDF.
- The idea, from the Vasicek one-factor model: the systematic factor is set at its worst 1-in-1000 value, giving
  the conditional PD; subtract expected loss and what remains is unexpected loss.
- Correlation R is lower for retail than for corporates because retail portfolios are diversified.
- PD floor: 0.05% for retail under Basel III.
- My RWA density is about 124% (high because LGD is 91%; there is no collateral).
- **Stress:** I fitted a logistic regression with borrower-risk variables plus a dummy for 2007–08 loans. For the
  same borrower quality, crisis-era loans had 1.37× the default odds (95% CI 1.21–1.56). I applied that to PD
  odds and added 5 pp to LGD. ECL rose 36% but capital only 11%, because the capital formula already assumes a 99.9%
  stress.

## 6. Questions you are likely to get (and short answers)
- **Why logistic regression, not XGBoost?**
  - Regulators and model validation need every coefficient to be explainable, and a points-based scorecard needs a
    linear model.
  - XGBoost might add a little Gini, but you would then need SHAP explanations and a harder validation.
  - It can be used as a challenger model.
- **What is reject inference?** Banks only see outcomes for approved loans. Reject inference estimates how the
  rejected applicants would have behaved (for example, augmentation or parcelling). I did not do it, which is a
  limitation.
- **What is data leakage here?** Using information from after the application date, like recoveries, last payment
  or the interest rate set from the lender's own grade. I used only application-date fields.
- **Why an out-of-time test?** A random split hides time effects. A real bank uses the model on future applicants,
  so 2014 tests exactly that, and it caught the calibration drift.
- **Your LGD is 91%. Is that realistic?** For unsecured personal loans sold to collection agencies, yes. Secured
  loans (home, auto) have far lower LGD because of collateral.
- **What would you add with more data?** Bureau score; behavioural variables for a behaviour scorecard; monthly
  payment history to do proper IFRS 9 staging and roll-rate PD term structures; macro-linked forward-looking PDs
  (unemployment as a scenario variable); downturn LGD.

## 7. How I fine-tuned it (if they ask "how did you improve the model?")
- First version: 7 application variables on 2007–13 loans, out-of-time Gini 0.259.
- I tested one change at a time, always judged on 2014 Gini:
  - Finer bins: almost no gain (+0.001).
  - Lower IV cut-off (0.02 → 0.01): small gain.
  - Credit-bureau variables (card limits, accounts opened in 24 months, months since the last account), which
    Lending Club recorded from 2012: **big gain, to 0.318**.
  - Together: **0.325**, a 25% improvement.
- I did NOT add the borrower's state, even though it added 0.002. Geography can be a proxy for race or
  community, which is a fair-lending problem, and the gain was tiny.
- Lesson: better data beats more tuning.

## 8. Numbers to remember
- 3,26,192 loans; trained on 2012–13 (1,00,679 loans); tested on 1,53,204 loans from 2014.
- 14 variables; Gini 0.325 and KS 0.23 out of time (beats Lending Club's grade at 0.319); PSI 0.003.
- Default rate 29.0% in the lowest band → 4.2% in the highest.
- Calibration: predicted 12.5% vs actual 14.6%; intercept shift +0.188.
- LGD 90.8%; 22.6% of defaults happen in the first year.
- 2014 book $1.93 bn:
  - Stage 1 ECL 2.3%, lifetime 6.0% (recalibrated).
  - Back-test: 102% of realised loss.
  - Adverse: ECL +36%, capital +11%.
