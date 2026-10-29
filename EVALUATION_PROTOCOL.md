# FHEViT Frozen Evaluation Protocol

This file records preregistered evaluation contracts and later amendments. It is a
chronological research record, not one configuration shared by every experiment;
later sections supersede earlier rules when they revise the same cohort or endpoint.
The current scientific conclusions are summarized in [STATUS.md](docs/handoff/STATUS.md)
and claim boundaries in [CLAIMS_AUDIT.md](docs/handoff/CLAIMS_AUDIT.md). Raw patient-level
result files and the historical experiment report are local-only and are not part of a
source release.

## 1. Frozen configuration

| Item | Frozen value | Source of truth |
|---|---|---|
| Clinical states `K` | 8 | `ClinicalCodecConfig.states` |
| Local channel width `L` | 4 | `ClinicalCodecConfig.width` |
| Privacy budget `epsilon` | 3.0 (pure `epsilon`-LDP, `delta = 0`) | `ClinicalCodecConfig.epsilon`, `ClinicalPrivacyChannel` |
| Encrypted multiplicative depth | 3 | `ClinicalMomentEncoder.compile` |
| Ciphertext product budget | 96 (hard reject above) | `ClinicalCodecConfig.max_encrypted_products` |
| Development cohort | 99 images / 20 patients train, 28 images / 7 patients test, patient-disjoint | `artifacts/spatial_quadratic_diagnostic.pt` split |
| Seeds | codec `812`, DP-noise measurement `1941`, bootstrap `42` | code defaults |
| Backends (pluggable) | `qwen2vl-2b`, `internvl2-2b` (add MedGemma / LLaVA-Med only after P1 passes) | `fpavlm/llm` |
| Non-inferiority margin | absolute `0.05` on the primary metric, patient-level | `paired_patient_noninferiority(margin=0.05)` |

The cohort split, seeds and margin above must not be changed to obtain a pass. New
sensitive-attribute or membership confirmation must use held-out patients that were
never inspected during tuning. This early configuration is retained for historical
context; it is not the active NIH protocol described by later sections.

## 2. The single yardstick

Every release mechanism is scored against the **image-independent constant-mean prefix**
on the same patient-disjoint test set, and significance is decided by a **patient-level
bootstrap interval**, never a point estimate:

* Feature transfer: per-study cosine and MSE-reduction-over-mean, aggregated per patient,
  with a 95% bootstrap CI via `fpavlm.evaluation.paired_metric_advantage`. A mechanism
  only "beats the mean" when the CI excludes zero.
* Clinical / report level (P1+): a diagnostic probe (`diagnostic_probe`) and report-level
  automatic metrics, compared to the native VLM with `paired_patient_noninferiority`.

A high absolute cosine is not evidence: the constant-mean prefix already reaches
cosine ≈ 0.774 on this cohort. Only the CI-backed advantage over that baseline counts.

## 2.1 Discriminative ceiling — locked before any log-odds result

Report keyword counts on this cohort sit on the floor for both 2B models. The next
native-ceiling measurement is a forced choice, not another free-text report. It was
fixed before that measurement was run.

| Item | Locked value |
|---|---|
| Question | `Does this chest X-ray show pneumonia? Answer Yes or No.` |
| Score | next-token `logit(Yes) - logit(No)`. Both strings are one token. |
| Image | one pneumonia-labeled image per patient, first in cohort order |
| Population for this probe | every pneumonia-labeled development patient. Nothing is fitted. |
| Bootstrap | 2000 patient resamples, seed 42 |
| Chance line | AUC 0.5, only once both classes exist |
| Release margin | absolute AUC `0.05` |

The margin is the top of the 0.03–0.05 absolute-AUC band used in diagnostic
non-inferiority studies, and it matches the 0.05 already frozen for paired clinical
scores. It is not a loss percentage chosen after seeing a point estimate. A release
is non-inferior to the plaintext native backbone only when the patient-bootstrap
one-sided 95% lower bound of `AUC(release) - AUC(native)` is greater than `-0.05`.
The native ceiling itself is above chance only when its own one-sided 95% lower bound
is greater than 0.5. Neither number is filled in here.

`todo` is not a negative label. The development cohort's labeled patients are all
pneumonia, so a pneumonia AUC is undefined there and must be reported as null.
Confirmation contains two `No Finding` studies, each on a patient who also has
COVID-19 images; they are not an independent negative class. The final holdout stays
sealed and is not opened to manufacture negatives. MedGemma-4B may be used later as
a native-ceiling probe after its local weights and any custom code are hashed. That
probe does not enter the multi-backend claim table and does not pass P1. Release
re-scoring and privacy attacks stay unstarted until a native AUC is defined and its
lower bound is above 0.5.

A Yes/No preference is not why the AUC was undefined. AUC ranks scores across classes.
It stays undefined until a negative class exists. The preference is not calibrated here.

## 2.2 Negative studies — locked before any score on them

Negatives are extra Open-I chest X-ray reports that are not already in
`data/openi_subset`, the development split, the confirmation split, or the final
holdout. The final holdout is not opened. Confirmation patients 173 and 253 stay out
of the negative pool: each has a `No Finding` study and a COVID-19 study.

A report is eligible only when every line below holds. The allow-list is not widened
after eligibility counts or log-odds are seen.

* `problems`, stripped and lowercased, is exactly `normal`.
* `mesh`, stripped and lowercased, is exactly `normal`.
* The impression normalizes onto the allow-list. Normalization is: lowercase; replace
  a redacted `x-` plus two or more `x` with `xray`; replace any remaining word of two
  or more `x` with a space; delete every character that is not a letter, digit, or
  space; collapse whitespace. The result must equal one of: `normal chest`,
  `normal chest xray`, `normal chest radiograph`, `no acute cardiopulmonary abnormality`,
  `no acute cardiopulmonary abnormalities`, `no acute cardiopulmonary finding`,
  `no acute cardiopulmonary findings`, `no acute cardiopulmonary disease`,
  `no acute cardiopulmonary process`, `no acute cardiopulmonary processes`,
  `no evidence of acute cardiopulmonary disease`, `no evidence of acute cardiopulmonary process`,
  `no evidence of acute cardiopulmonary abnormality`, `no evidence of acute cardiopulmonary finding`,
  `no evidence of acute cardiopulmonary findings`, `no acute intrathoracic process`,
  `no acute intrathoracic abnormality`, `no acute intrathoracic finding`,
  `no acute intrathoracic findings`, `no acute pulmonary finding`,
  `no acute pulmonary findings`, `no acute pulmonary disease`,
  `no acute pulmonary abnormality`, `no acute disease`,
  `negative for acute cardiopulmonary abnormality`, `negative for acute cardiopulmonary disease`,
  `lungs are clear`, `clear lungs`, `normal study`.
* One image per report, the first image in document order.
* Eligible reports are ordered by numeric uid. The negative pool is the first 20.
  If fewer than 10 reports qualify, the pool is not repaired by relaxing this section,
  and the AUC is not computed.

## 2.3 Same-source negatives — locked before any score on them

The development pneumonia studies come from `ieee8023/covid-chestxray-dataset` at
commit `78543292f8b01d5e0ed1d0e15dce71949f0657bb`. Open-I is a different collection.
A same-source negative has to come from that COVID metadata file. Hospital location
is recorded, and the file is one multi-hospital compilation rather than one ward.

A patient is eligible only when every line below holds. This section is not widened
after the eligibility count is seen, and no log-odds are computed on a shortfall.

* The patient id is absent from development, confirmation, and the final holdout.
* Every metadata row for that patient has finding exactly `No Finding`.
* At least one row is an X-ray whose view starts with `PA` or `AP`, whose filename
  is present, and whose license is one of the licenses already allowed for the
  privacy cohorts.
* The image is the first such X-ray in `(date, filename)` order.
* Eligible patients are ordered by the existing cohort patient hash. The pool is the
  first 20. If fewer than 10 patients qualify, the pool is not repaired and the AUC
  is not computed.

## 2.4 Etiology inside the development cohort — locked before any score on it

Section 2.3 does not supply a same-source normal class, and that section is not
relaxed. The question in this section is not pneumonia versus normal. It is whether
native log-odds separate COVID-19 pneumonia from the other pneumonia labels already
on the 20 labeled development patients. Confirmation, the final holdout, Open-I, and
any new corpus stay unread. MedGemma is not scored. Release re-scoring is not run.
The cross-collection AUC from section 2.2 does not unlock either of those.

One image per patient, the first development row whose finding contains `pneumonia`.
Rows whose finding does not contain `pneumonia`, including `todo` and `Unknown`, cast
no vote. The votes that remain must be one string, and that string must be one of
the following. Anything else excludes the patient, the exclusion is not repaired, and
the AUC is not computed.

* COVID-19, label 1: `Pneumonia/Viral/COVID-19`
* Other pneumonia, label 0: `Pneumonia`, `Pneumonia/Bacterial/E.Coli`,
  `Pneumonia/Bacterial/Streptococcus`, `Pneumonia/Fungal/Pneumocystis`,
  `Pneumonia/Viral/Varicella`

The locked census is 14 COVID-19 patients and 6 other-pneumonia patients, with zero
exclusions. A different count aborts before the model loads. No second contrast is
added after the scores.

The prompt is `Is this pneumonia caused by COVID-19? Answer Yes or No.` The score is
`logit(Yes) - logit(No)`. `Yes` means COVID-19. Both strings must be one token on the
tokenizer in use, or the run aborts. The patient bootstrap uses 2000 draws and seed
42. A draw that lacks either class is skipped. Fewer than 100 draws that contain both
classes aborts the interval. The native etiology ceiling is above chance only when
the one-sided 95% lower bound of that AUC is greater than 0.5. A bound at or below
0.5 does not authorize a new prompt, a new grouping, or a return to section 2.2 or
2.3.

Hospital location may be copied from the source metadata onto the result row. It is
not an eligibility rule and it is not matched across the two labels. This compilation
is not one hospital, so a bound above 0.5 rejects the Open-I versus COVID collection
explanation and still leaves a hospital-mix limitation on the page.

## Constant shift — closed without a rescore

ROC AUC compares the order of scores. Adding one constant to every score keeps every
pairwise order, and it keeps ties tied. The Yes/No preference is that kind of shift:
it moves both classes together. It does not explain an AUC sitting next to 0.5.
Section 2.4 is not rerun with a subtracted median, a subtracted mean, or any other
constant. A later section must not spend a model call on that check.

## 2.5 NIH ChestX-ray14 probe — locked before any score on it

Sections 2.2 through 2.4 do not supply a baseline that is both same-source and above
chance. This section does not relax them, and it does not add another task on the
27-patient COVID compilation.

The corpus is NIH ChestX-ray14, one clinical center. The label file is
`Data_Entry_2017_v2020.csv`, 112,120 image rows. Images are the PNG members of the
twelve official image archives. A JPEG recompress is not a substitute. CheXpert and
MIMIC-CXR are not downloaded here. MIMIC-CXR needs PhysioNet credentialing, CITI
training, and a signed agreement. CheXpert is issued after Stanford's agreement.
NIH's own FAQ states that use is unrestricted aside from citing Wang et al., CVPR
2017, acknowledging the NIH Clinical Center, and linking to
`https://nihcc.app.box.com/v/ChestXray-NIHCC`.

The question is pneumonia versus `No Finding` on this corpus. Labels are the
released NLP labels. NIH states that `No Finding` is not the same statement as a
radiologist's normal: a study can carry a pattern outside the fourteen names, or an
uncertain mention. That limitation is recorded. It is not repaired by swapping in
another corpus after the scores.

A patient is eligible only when every line below holds. The counts are not widened
after the eligibility census, and no log-odds are computed on a shortfall.

* View Position is `PA` or `AP` on the chosen row.
* Patient Age on the chosen row is an integer from 18 through 90.
* The patient is pneumonia, label 1, when at least one row's finding labels contain
  the token `Pneumonia`, and no row is exactly `No Finding`. Other tokens on the
  chosen row, separated by `|`, stay on the row. They are not an exclusion.
* The patient is `No Finding`, label 0, when every row is exactly `No Finding`.
* A patient with both a `Pneumonia` token and a `No Finding` row is excluded.
* The chosen row is the earliest of the eligible rows in `(Follow-up #, Image Index)`
  order. If that row fails the age or view line, the patient is excluded. A later
  row is not promoted.
* Eligible patients in each class are ordered by `sha256("FHEViT-nih14-v1:{patient_id}")`.
  The probe is the first 14 pneumonia patients and the first 13 `No Finding`
  patients. If either class has fewer than that count, the probe is not repaired
  and the AUC is not computed.

These 27 patient ids are the development probe. They are not reused as a later
confirmation set or final holdout. Those splits are not drawn in this section. The
official NIH `test_list.txt` is not read.

The prompt is the section 2.1 sentence `Does this chest X-ray show pneumonia? Answer Yes or No.`
The score is `logit(Yes) - logit(No)`. The patient bootstrap uses 2000 draws and
seed 42. The native ceiling is above chance only when the one-sided 95% lower bound
is greater than 0.5. A bound at or below 0.5 does not authorize a new prompt, a
comorbidity filter, or a larger redraw inside this section.

This section scores `qwen2vl-2b` and `internvl2-2b` on native tokens. These 27
patients are not opened again for MedGemma. Oracle PCA, the teacher mean, and the
linear bridge are not refit on these images.

## 2.6 NIH replication — locked before any score on it

Section 2.5 is not redrawn. Its 27 patient ids stay sealed. They are not a
confirmation set, not a final holdout, and not members of this sample. Qwen's
section 2.5 one-sided bound sits on 0.5 because the Yes rates on the two classes
are close. That bound is not calibrated, shifted, or otherwise rescued.

Eligibility is unchanged from section 2.5: the same label file, the same age
window, the same view rule, the same earliest-row rule, the same comorbidity
rule, and the same exclusion of a patient who has both `Pneumonia` and
`No Finding`. The order salt stays `sha256("FHEViT-nih14-v1:{patient_id}")`.
The prompt stays `Does this chest X-ray show pneumonia? Answer Yes or No.` The
score stays `logit(Yes) - logit(No)`. The patient bootstrap stays 2000 draws and
seed 42. Above chance still means the one-sided 95% lower bound is greater than
0.5.

Every patient id in the section 2.5 manifest is removed first. The replication
is the next 36 pneumonia patients and the next 36 `No Finding` patients in that
same order, 72 patients in total. If either class runs short after the removal,
the replication is not repaired and the AUC is not computed.

These 72 patient ids are sealed as the replication. They are not reused as a
later confirmation set or final holdout. A result here does not authorize a new
prompt, a comorbidity filter, or a constant-shift rescore. Both local native
backbones are scored. The official NIH `test_list.txt` is not read.

## 2.7 Same-sample release comparison — locked before any refit

The section 2.6 AUCs are not the baseline for release non-inferiority. Every
patient in that manifest already contributed to those AUCs. Using that number
beside a release AUC from different patients would compare two samples.

Section 2.6 patient ids are the development fitting set and nothing else. Section
2.5 patient ids stay out of both sets. Confirmation is a new slice: after those
ids are removed, the same eligibility lines and the same
`sha256("FHEViT-nih14-v1:{patient_id}")` order supply the next 36 pneumonia
patients and the next 36 `No Finding` patients. If either class runs short, the
slice is not repaired and nothing is fit.

The teacher mean, the PCA basis, and the linear bridge are fit on development
images only. No confirmation image and no confirmation score enter the fit.
The prompt stays `Does this chest X-ray show pneumonia? Answer Yes or No.` The
score stays `logit(Yes) - logit(No)`. The bootstrap stays 2000 patient draws
and seed 42. The margin stays the section 2.1 absolute AUC `0.05`.

Native AUC and release AUC are computed on the confirmation patients only, on
the same images. The release is non-inferior only when the patient-bootstrap
one-sided 95% lower bound of `AUC(release) - AUC(native)` is greater than
`-0.05`. A draw that lacks either class is skipped. Fewer than 100 complete
draws aborts the comparison.

The width check is separate from that AUC. Oracle PCA ranks are 4, 16, and 64.
Each rank is a linear bridge from the development PCA code to the native tokens.
It beats the development teacher mean only when the confirmation-patient
two-sided 95% interval of MSE improvement lies above zero. A point estimate is
not enough.

`internvl2-2b` is fit first. `qwen2vl-2b` is not fit until that InternVL2 width
result is on disk. MedGemma is not fit and is not scored in this section. The
0.4 finite-state release is not scored unless at least one oracle PCA rank beats
the mean on the confirmation patients. The official NIH `test_list.txt` is not
read.

## 2.8 Predictor capacity — locked before any NIH score of it

Section 2.7 showed that oracle PCA coordinates can move the native tokens. It
did not show that any function of the image can predict those coordinates. This
section asks only that second question. The COVID learning curve is not
reopened, and its result file is not overwritten.

The backend is `internvl2-2b`. Qwen is not fit here. The target is oracle PCA
rank 4 of the native tokens. The basis is fit once on all 72 development
patients and then held fixed. Confirmation images do not enter the basis.

The predictor is frozen TorchXRayVision `densenet121-res224-all`. Nothing in
that network is trained. The only learned map is a ridge from its 1024 pooled
features to the four coordinates. The penalty, including the intercept, is
chosen by leave-one-patient-out inside the training patients named for that
row. The same penalty grid as the COVID curve is used. No privacy noise is
added. The coordinates are not discretized. The 0.4 codec is not loaded.

The learning curve uses the first 4, 8, 12, 16, 20, 36, and 72 patients of one
permutation of the development ids. The permutation seed is 812. Every row is
reported. No row is dropped after seeing the confirmation scores. The prefix
ridge stays the protocol ridge. Confirmation patients are the only evaluation
patients, at every row.

The gate is the 72-patient row alone. The coordinate fit beats the development
coordinate mean only when the confirmation two-sided 95% interval of MSE
improvement lies above zero. The prefix fit beats the development token mean
on the same rule. A new finite-state alphabet is not started in this section,
whatever those two intervals say. A stronger encoder is not trained in this
section. MedGemma is not fit. The official NIH `test_list.txt` is not read.

## 2.9 Qwen native stability — locked before the redraw

The section 2.6 Qwen log-odds stay as stored. They are not recomputed. The
confirmation native log-odds are recomputed only if the per-patient rows were
not saved, with the same prompt and the same score, on the confirmation images
only. A recomputed confirmation AUC that disagrees with the section 2.7 native
AUC aborts this section. No new NIH slice is downloaded. The 27-patient probe
stays out.

The pooled patients are the 72 development patients and the 72 confirmation
patients, 72 pneumonia and 72 `No Finding`. Two hundred draws, seed 42, each
take 36 pneumonia patients and 36 `No Finding` patients without replacement.
Each draw reports a point AUC. There is no nested bootstrap inside a draw.

Qwen stays a release-comparison backend only when the 2.5 and 97.5 percentiles
of those 200 AUCs both lie strictly above 0.5. An interval that does not clear
that line retires Qwen from the release comparison. The prompt is not changed.
The scores are not calibrated. InternVL2 remains the main line for section 2.8
either way. MedGemma is not fit. The official NIH `test_list.txt` is not read.

## 2.10 Pooled-token self-prediction — locked before any score of it

Section 2.8 selected the intercept for frozen DenseNet at every training size.
This section changes only the features. The target stays InternVL2 oracle PCA
rank 4, with the basis fit once on the 72 development patients. Confirmation
images do not enter the basis.

The predictor is the mean of the 256 frozen native visual tokens, one vector
per image. The unpooled token matrix is not the predictor. Nothing in the
vision tower is trained. The ridge, the penalty grid, the intercept candidate,
and leave-one-patient-out stay as in section 2.8. The training subsets are the
section 2.8 patient lists. Every size is reported. The gate is the 72-patient
row, and it uses the same two-sided 95% interval rule. No privacy noise is
added. The coordinates are not discretized.

A trainable adapter is not started in this section. The PCA target is not
redefined in this section. Qwen is not fit. MedGemma is not fit. The official
NIH `test_list.txt` is not read.

## 2.11 Hash-split covariate balance — locked before the counts

The development and confirmation manifests are the two groups. No patient is
added or removed. The 27-patient probe stays out. A stratified split is not
started in this section.

The primary statistics are age, sex, view, and the subtype omnibus:
development minus confirmation in mean age, in the female proportion, and in
the PA proportion, plus one subtype omnibus. The omnibus is the sum of absolute
differences, among pneumonia patients only, of the bare-`Pneumonia` rate and
of the rates of `Atelectasis`, `Cardiomegaly`, `Effusion`, `Infiltration`,
`Mass`, `Nodule`, `Pneumothorax`, `Consolidation`, `Edema`, `Emphysema`,
`Fibrosis`, `Pleural_Thickening`, and `Hernia`. A token outside that list is
reported and does not enter the omnibus.

The null is a random partition with 36 pneumonia and 36 `No Finding` patients
on each side. Two thousand partitions, seed 42. Each p-value is
`(1 + count of partitions whose absolute statistic is at least the observed
absolute statistic) / 2001`. The split is called imbalanced only when a primary
p-value is below `0.0125`. No covariate is added after the counts are seen.

## 2.12 Later Qwen release comparisons

The section 2.7 Qwen non-inferiority result remains the recorded result for
that one split. A later claim that the Qwen release is non-inferior, or that
it is not, is not made from that one split. It requires a patient-level redraw
of the paired AUC difference, on scores that are not refit for the redraw.
That redraw is not run in this section. The prompt is not changed. This section
does not retire Qwen and does not rescore the symbol.

## 2.13 Pooled nonlinearity — locked before any score of it

Section 2.10 is the linear baseline. This section does not refit the PCA basis
and does not change the 72 development patients or the confirmation patients.
The features stay the mean of the 256 frozen native tokens. The linear probe is
refit with the section 2.8 penalty grid. If its 72-patient prefix improvement
disagrees with the stored section 2.10 value by more than `0.001`, this section
aborts.

The primary check is an RBF kernel ridge. Inside each leave-one-patient-out
fold the pooled features are standardized, and the fold does not see
confirmation rows. The penalty grid is `logspace(-2, 4, 7)`. The gamma grid is
`logspace(-4, 1, 6)`. A tie keeps the larger penalty, then the smaller gamma.
The secondary check is degree-2 polynomial features on 16 principal components
of the pooled features. Those components are fit on the development patients
only, and the following ridge uses the section 2.8 penalty grid.

A nonlinear prefix beats the linear prefix only when the paired two-sided 95%
interval of the MSE-improvement difference lies above zero. Linear capacity is
called insufficient only when the RBF prefix beats the linear prefix. The
polynomial row is reported and does not by itself make that call. A multilayer
perceptron is not started. The PCA target is not redefined. Qwen is not fit.

## 2.14 Unpooled attention pool — locked before any score of it

The target, the patients, and the linear prefix baseline stay those of section
2.13. The predictor is no longer the token mean. Each of the 256 tokens gets a
score from one linear map, the scores go through softmax, and the weighted sum
of tokens goes through one linear map to four coordinates. Training uses the 72
development patients only: AdamW, learning rate `0.001`, weight decay `0.01`,
100 epochs, seed 812. The score vector starts at zero, so the first forward
pass is the token mean. Confirmation scores are not used to pick an epoch.

The attention prefix beats the linear prefix only under the section 2.13 paired
interval rule. That is the only support this section gives to the claim that
global pooling discarded local information. A regional release is not started.
A multilayer perceptron beyond this pool is not started.

## 2.15 Qwen release redraw — the section 2.12 rule, executed once

The redraw uses the confirmation patients only. Development release scores are
not computed. The codec is reproduced only to archive per-patient native and
release log-odds. If either reproduced AUC disagrees with the section 2.7 file
by more than `0.0001`, the redraw aborts and does not overwrite the section 2.7
file. The non-inferiority rule is unchanged: the patient-bootstrap one-sided
95% lower bound of `AUC(release) - AUC(native)` must be greater than `-0.05`.
The prompt is not changed. The scores are not calibrated.

## 2.16 Oracle AUC alignment — locked before any score of it

Sections 2.8 through 2.14 optimize token MSE. This section asks whether the
oracle coordinates themselves are non-inferior on the pneumonia AUC. The
release is the true development PCA code passed through the same linear bridge
as section 2.7. It is not a predicted code. Ranks are 4, 16, and 64. Every
rank is scored. No rank is dropped after the AUC is seen.

The patients, the prompt, and the score stay those of section 2.7. Native AUC
and oracle AUC use the same confirmation images. The margin stays `0.05`. A
rank is non-inferior only when the patient-bootstrap one-sided 95% lower bound
of `AUC(oracle) - AUC(native)` is greater than `-0.05`. The backend is
`internvl2-2b`. Qwen is not fit. Qwen is the secondary line.

If a refit MSE improvement disagrees with the stored section 2.7 value by more
than `0.001`, or the native AUC disagrees with the stored section 2.7 native
AUC by more than `0.0001`, this section aborts. The coordinate path remains
open only when at least one rank is non-inferior. A multilayer perceptron is
not started. Clinical-label training is not started. The RBF grid is not
extended. A regional release is not started.

## 2.17 Oracle-width neighbors — locked before any score of them

Section 2.16 scored ranks 4, 16, and 64. This section adds ranks 8 and 32
only. The release is the true development PCA code through the same linear
bridge as section 2.7. It is not a predicted code. Ranks 4, 16, and 64 are
not refit and are not rescored. Their stored section 2.16 oracle AUC point
estimates are the other three points on the curve.

The patients, the prompt, the score, and the margin stay those of section
2.7. Native AUC and oracle AUC use the same confirmation images. A rank is
non-inferior only when the patient-bootstrap one-sided 95% lower bound of
`AUC(oracle) - AUC(native)` is greater than `-0.05`. The backend is
`internvl2-2b`. Qwen is not fit.

If the native AUC disagrees with the stored section 2.7 native AUC by more
than `0.0001`, this section aborts. The curve peak is the rank in 4, 8, 16,
32, and 64 with the highest oracle AUC point estimate. A tie keeps the
smaller rank. This section does not retarget the predictor. A capacity
diagnostic on a new rank is not started. A privacy-channel redesign is not
started. A multilayer perceptron is not started. Clinical-label training is
not started.

## 2.18 Curve intervals — locked before any resampling of the stored scores

The section 2.7 confirmation patients were used to compare ranks 4, 8, 16,
32, and 64. Those patients are a selection set. This section does not make a
non-inferiority claim and does not sample a new confirmation set.

The inputs are the stored per-patient log-odds. Ranks 4, 16, and 64 come
from the section 2.16 file. Ranks 8 and 32 come from the section 2.17 file.
No model is loaded. No bridge is refit. If patient ids or labels disagree
across the files, or a native log-odds differs by more than `0.0001`, this
section aborts. If a recomputed oracle AUC disagrees with the stored point
estimate by more than `0.000001`, this section aborts.

One patient bootstrap, 2000 draws, seed 42, is shared by every rank. A draw
that loses either pneumonia class is skipped for every rank. Fewer than 100
valid draws aborts the section. The curve band is the two-sided 95% interval
of each rank's oracle AUC. For each pair, smaller rank first, the paired
difference is `AUC(smaller) - AUC(larger)` on those same draws. An interval
includes zero when its lower bound is at most zero and its upper bound is at
least zero. The bootstrap standard deviation of `AUC(oracle) - AUC(native)`
uses divisor `n - 1`. This section does not select a rank. The predictor is
not retargeted. A multilayer perceptron is not started. A privacy-channel
redesign is not started. A regional release is not started.

## 2.19 Non-inferiority sample size — locked before the census

The planning standard error is the median of the five section 2.18 bootstrap
standard deviations. The margin stays `0.05`. The bound is the one-sided 95%
lower bound. The planned true differences are `0` and `+0.03`, each at 80%
power. With `z_0.95` and `z_0.80` the normal quantiles,

`n = 72 * (se * (z_0.95 + z_0.80) / (delta + 0.05))^2`.

The planned size is the smallest even integer at least that large. Power at
72 patients is reported for both differences. The census applies the section
2.5 eligibility rule to the local NIH table and excludes every patient id in
the section 2.5, 2.6, and 2.7 manifests. It also counts pneumonia image rows
and unique patients who have a pneumonia token, before that eligibility rule.
A new confirmation manifest is not written. Images are not downloaded. The
predictor is not retargeted.

## 2.20 Attribute inference — locked before any attack score

The releases are the noiseless development PCA coordinates at ranks 4, 8,
and 16. The control is the mean of the 256 native visual tokens. The
attacker is fit on the 72 development patients and scored on the 72
confirmation patients. Confirmation images do not enter the PCA or the
attacker fit. Sex is female versus male. View is PA versus AP. Both use
logistic regression with `C = 1`, solver `lbfgs`, a standardizer fit on
development only, and seed 42. Age uses ridge regression with penalty `1`
and that standardizer. The age baseline predicts the development mean age.

A patient bootstrap of the confirmation scores, 2000 draws, seed 42,
resamples the frozen predictions and does not refit the attacker. A binary
attribute is above chance only when its one-sided 95% lower bound is greater
than `0.5`. Age beats the training mean only when the two-sided 95% interval
of `baseline MAE - model MAE` lies above zero. For each rank and binary
attribute, the same draws give a two-sided 95% interval of
`AUC(coordinates) - AUC(pooled)`. Ranks are not added after the scores. This
attack does not choose a clinical rank and does not make a non-inferiority
claim. A reconstruction attack is not started. The predictor is not
retargeted. A multilayer perceptron is not started. A privacy-channel
redesign is not started.

## 2.21 View shortcut — locked before any count of the stored slices

Section 2.20 showed that noiseless coordinates predict PA versus AP. This
section asks whether that view also separates pneumonia from `No Finding`
in the slices already scored. It uses the stored confirmation log-odds and
the two manifests. No model is loaded. No bridge is refit. Oracle component
scores are not on disk. This section does not extract features to recreate
them, and component correlations are not computed.

The development manifest and the confirmation manifest are tabulated
separately. The view score is 1 for AP and 0 for PA. The AP-rate difference
is the AP rate among pneumonia patients minus the AP rate among `No Finding`
patients. Its two-sided 95% interval is a patient bootstrap, 2000 draws,
seed 42. The rate is higher only when that interval lies above zero. The
view-only pneumonia AUC uses the same bootstrap. View predicts pneumonia
only when its one-sided 95% lower bound is greater than `0.5`. A slice
supports the shortcut only when its AP rate is higher and view predicts
pneumonia.

PA-only AUCs use the confirmation patients whose stored view is PA, and the
stored native and oracle log-odds at ranks 4, 8, and 16. A PA-only AUC is
above chance only when its one-sided 95% lower bound is greater than `0.5`.
This section does not make a non-inferiority claim and does not select a
rank. Direction nulling is not started. The predictor is not retargeted.
A multilayer perceptron is not started. A privacy-channel redesign is not
started. A regional release is not started.

## 2.22 Image-level pneumonia pool — locked before any new score

Section 2.19 is the reason for this revision. Its census, not an AUC, showed
that the patient-level exclusion of mixed `No Finding` and pneumonia does
not leave enough pneumonia patients for the 80% plan. The sealed section
2.5, 2.6, and 2.7 manifests are not relabeled and are not rewritten.

A negative patient still requires every row to be exactly `No Finding`, with
the same chosen-row age and view rule as section 2.5. A positive patient is
the earliest pneumonia image with view PA or AP; that image's age must be
an integer from 18 through 90. A later pneumonia image is not promoted. A
patient with both a `No Finding` row and a pneumonia image is positive when
that image passes, and is never negative. Patients already in the three
manifests are excluded. Order is the section 2.5 hash.

The zero-difference plan in section 2.19 requires 253 patients in each
class. This section starts a sealed confirmation set only when both
remaining classes have at least 253 patients. The set is the first 253
pneumonia patients and the first 253 `No Finding` patients in that order.
The margin stays `0.05` and is not widened. If either class is short, no
manifest is written, no image is downloaded, and no score is computed.

Section 2.18 did not distinguish the ranks. The primary rank is the
smallest, rank 4. Rank 8 is the only secondary comparison. Ranks 16, 32,
and 64 are not scored. The bridge is fit on the existing development
patients. The new patients are the confirmation sample. A failed download
is not replaced. Scoring waits until every chosen image is stored. The
predictor is not retargeted. A multilayer perceptron is not started. A
privacy-channel redesign is not started. A regional release is not started.
MedGemma is not fit. CheXpert and MIMIC are not downloaded. Direction
nulling is not started. A reconstruction attack is not started.

## 2.23 Scored-set strata — locked before any new count

The 506 patients scored in section 2.22 are no longer a confirmation set.
They are a development analysis set. The 171 patients in the section 2.5,
2.6, and 2.7 manifests were already inspected. They may be used later for
bridge fitting. They are not a fresh confirmation set. Every remaining
image-level pneumonia patient is the final confirmation positive pool. This
section does not download those images and does not score them. Future
confirmation negatives, when a later section samples them, are two per
positive, taken by the section 2.5 hash from `No Finding` patients outside
the inspected manifests. That later sample is not drawn here.

The 0.4 finite-state symbol is not tested again. The noiseless rank-4 and
rank-8 coordinates already failed the margin. A predictor is not retargeted.
A multilayer perceptron is not started. The RBF grid is not extended.
MedGemma is not fit.

This section uses the stored section 2.22 log-odds and the sealed manifest.
No model is loaded. If the recomputed native AUC disagrees with the stored
section 2.22 value by more than `0.000001`, this section aborts. The view
score is 1 for AP and 0 for PA. The AP-rate
rule and the view-only AUC rule are the section 2.21 rules, applied to these
506 patients. PA-only AUCs use stored native and oracle log-odds at ranks 4
and 8. An AUC is above chance only when its one-sided 95% lower bound is
greater than `0.5`. A positive image is bare when its label tokens are
exactly `Pneumonia`, and comorbid otherwise. Native AUC is computed for bare
positives against all `No Finding` patients, and again for comorbid
positives against the same negatives. Comorbidity carries the native AUC
only when the comorbid lower bound is greater than `0.5` and the bare lower
bound is not. Bootstrap draws are 2000, seed 42. This section does not make
a non-inferiority claim and does not select a basis, bridge, or rank.

## 2.24 Prefix-loss grid — locked before any cell is scored

The grid below is fixed before any of its scores. It is not changed after a
cell is seen. The margin stays `0.05`. A cell passes only when the
patient-bootstrap one-sided 95% lower bound of `AUC(release) - AUC(native)`
is greater than `-0.05`. Draws are 2000, seed 42.

The bases are unsupervised PCA and a supervised PLS direction fit to the
pneumonia label. The bridge sizes are the original 72 development patients,
and 5-fold patient cross-fitting on the inspected patients. The ranks are 4,
8, 16, 32, 64, and the training-fold maximum `n_train - 1`. Confirmation
patients reserved in section 2.23 are not used. A passing cell is not scored
on that reserve in this section. If more than one cell passes, the selected
cell is the smallest rank, then the supervised basis, then the 72-patient
bridge. If no cell passes, the global prefix grid failed. A regional release
is not started. A noisy full-token release is not started. The 0.4 symbol is
not tested. A reconstruction attack is not started in this section. A
predictor is not retargeted. A multilayer perceptron is not started.

## 2.25 PA-only endpoint — locked before any new count

Section 2.24 is unchanged and has not been scored. This section is the
revision. The view analysis that motivates it used plaintext scores and
metadata. It did not use a release score to choose the endpoint.

The primary endpoint is PA-only. AP images are exploratory. The chosen image
stays the section 2.22 image: the earliest eligible film, with no later film
promoted. A patient is in the PA-only pool only when that image is PA. An AP
image is not replaced by a later PA image. Negatives use the same rule.

The margin stays `0.05`. A later grid cell passes only when the
patient-bootstrap one-sided 95% lower bound of
`AUC(release) - AUC(native)` on PA-only patients is greater than `-0.05`.
The stored PA-only native AUC is 0.758, so that margin reads as 0.708 on the
absolute scale. The test remains the paired difference. The full set is
reported and is not the pass rule. The view-only prefix is a bridge from the
AP indicator, scored on the full set where view varies. It is constant on
PA-only patients, so it is not a PA-only AUC. The training-fold maximum rank
`n_train - 1` stays in the grid. This section does not score the grid.

If the recomputed PA-only native AUC disagrees with the stored section
2.23 value by more than `0.000001`, this section aborts. The planning
standard error is the larger of the rank-4 and rank-8 bootstrap
standard deviations of the PA-only paired AUC difference on the scored 506,
2000 draws, seed 42, divisor `n - 1`. It scales with the square root of the
PA-positive count. The planned true differences are `0` and `+0.03`, each at
80% power, with the section 2.19 normal approximation. If the reserved
pneumonia patients whose chosen image is PA cannot reach 80% power at a true
difference of `0`, no confirmation sample is drawn and no image is
downloaded. The 0.4 symbol is not tested. A predictor is not retargeted. A
multilayer perceptron is not started. MedGemma is not fit.

## 2.26 Reconstruction ceiling — locked before any attack score

The same linear attacker reads three inputs: the 256 native tokens, their
mean, and PCA coordinates at ranks 4, 8, and 16. It is ridge regression with
penalty `1` from that input to a 64 by 64 grayscale image. Inputs are
standardized on the 72 training patients. Each chest
radiograph is converted to one grayscale channel, resized with bilinear
interpolation, and scaled to `[0, 1]`. LPIPS repeats that channel three
times and maps `[0, 1]` onto `[-1, 1]`. The attacker is
fit on the 72 development patients and scored on the 506 analysis patients.
The reserved confirmation patients are not used. Metrics are PSNR, SSIM, and
LPIPS. A patient bootstrap, 2000 draws, seed 42, resamples the frozen
reconstructions. This attack does not choose a release. The prefix grid is
not scored here. The 0.4 symbol is not tested. A predictor is not
retargeted.

## 2.27 Unlabeled bridge curve — locked before any new score

Section 2.25 has been scored. No cell passed. This section does not rewrite
that grid and does not open the reserved confirmation patients.

The PCA basis, the teacher-token mean, and the bridge do not use the
pneumonia label. A training patient is absent from every NIH evaluation
manifest and from the reserved confirmation list. Finding labels are not
read for selection. The chosen image is the earliest PA or AP film. A later
film is not promoted. Age on that film must be an integer from 18 through
90. Patients are ordered by the existing hash. The prefixes are the first
500, the first 2000, and the first 8000. Each shorter prefix is the start
of the longer one.

The basis is unsupervised PCA. The ranks are 64, 128, 256, and the training
maximum `n_train - 1`. The bridge is the ridge prefix with penalty `1` and
sigma `0`. It is scored on the 506 analysis patients. The pass rule is
unchanged: the patient-bootstrap one-sided 95% lower bound of
`AUC(release) - AUC(native)` on the 368 PA images is greater than `-0.05`.
Draws are 2000, seed 42.

For one rank, the step from 2000 to 8000 is flat when the one-sided 95%
lower bound of `AUC(8000) - AUC(2000)` on those same PA patients is less
than or equal to `0`. A flat curve stops the schedule. No larger prefix is
added. The margin is not widened.

The reserve opens only when a cell in this section passes and power from
that cell's own PA-only bootstrap standard deviation, scaled to the 301
reserved PA positives, is at least 0.8. The deviation uses 2000 draws, seed
42, and divisor `n - 1`. The planned true differences are `0` and `-0.02`,
with the section 2.19 normal approximation. A point estimate near `-0.02`
does not by itself open the reserve. The pass probability of that region
is recorded before any new cell is scored. No reserved image is downloaded
in this section. The 0.4 symbol is not tested. A predictor is not
retargeted. A multilayer perceptron is not started. MedGemma is not fit.

## 2.28 Strong reconstructor — locked before any attack score

The reconstructor trains on the 8000-patient prefix from section 2.27 and
on that prefix's cached features. It is not the section 2.26 linear map.
The inputs are the 256 native tokens, their mean, and PCA coordinates at
ranks 64, 128, 256, and `n_train - 1`. Full tokens enter through a 1 by 1
convolution. A pooled vector or a coordinate vector enters through a linear
layer. Either path yields a 16 by 16 by 128 grid. A ReLU follows. Two
stride-2 transposed convolutions use kernel 4 and padding 1 and end at one
64 by 64 channel, then a sigmoid. Training minimizes mean squared error
with Adam, learning rate `0.001`, 20 epochs, batch size 32, and seed 812.
No epoch or learning rate is chosen after a score.

Images are one grayscale channel, bilinear 64 by 64, scaled to `[0, 1]`.
The baselines are the mean training image and the mean training image of
the same view. LPIPS is not a decision metric. A reconstruction beats a
baseline when the two-sided 95% patient-bootstrap interval of the per-image
SSIM advantage lies above zero. Draws are 2000, seed 42. The primary call
is the view-conditional mean. PSNR is reported and does not decide. This
attack does not choose a release.

Sex and view use the section 2.20 logistic specification, fit on this
8000-patient prefix and scored on the 506 analysis patients. Above chance
means the one-sided 95% lower bound is greater than `0.5`. The reserved
confirmation patients are not used. A near-full-rank release is not given
an epsilon-3 local-DP claim. The privacy statement of this section is
empirical. The 0.4 symbol is not tested. A predictor is not retargeted.

## 2.29 Deviation record — locked with the scored prefixes

Section 2.27 and section 2.28 are unchanged. The first 3000 images of the
8000-patient list are the only unlabeled images on disk. The scored prefixes
are 500 and 2000. The 8000 prefix was not scored.
Two prefixes are not a learning curve.
Do not extrapolate to a larger prefix.

The cell that passed is unsupervised PCA at prefix 2000 and rank
`n_train - 1`. Those coordinates are principal components of the native
InternVL tokens. The depth-3 moment circuit does not compute them. A pass
on that cell is not an FHE result. The patient bootstrap resamples the 368
PA images and does not redraw the training patients.

The reserved confirmation stays closed. A passing development cell does not
download those images. Noise, a student, and CKKS error would be a different
release. The confirmation is used only after the development release is
fixed. The 0.4 symbol is not tested. A predictor is not retargeted.

## 2.30 Training redraw — locked before any redraw score

This section does not change the published section 2.27 cell and does not
open the reserved confirmation.

Five training sets are drawn from the cached first 3000 unlabeled patients.
Each set contains 2000 distinct patients. The generator is NumPy `PCG64`
with seed 42. The five draws are five calls of `choice` without replacement,
in that order. The published hash prefix is not replaced.

Each draw fits unsupervised PCA at rank 1999, then the ridge bridge with
penalty `1` and sigma `0`. Scoring uses the same 506 patients. The pass
rule is unchanged: the one-sided 95% lower bound is greater than `-0.05`
on the 368 PA images. Draws are 2000, seed 42. The published bound is
unstable when at least 3 of the 5 draws fail that rule. The redraw does not
choose a rank. The 0.4 symbol is not tested.

## 2.31 Coordinate frontier — locked before any new cell

This section adds ranks 512 and 1024. It does not rewrite section 2.27.
Section 2.28 stays unfitted, because its 8000-patient prefix is not on disk.

The PCA basis is fit on the first 2000 unlabeled patients. The ranks are
256, 512, 1024, and 1999. After projection, isotropic Gaussian noise is
added to the coordinates. The scales are `0`, `0.25`, `0.5`, and `1` times
the training-set standard deviation of that coordinate. A zero standard
deviation stays zero. The generator is NumPy `PCG64` with seed 812. Cells
follow that rank order and, within a rank, that scale order. Each cell
consumes one training draw and one test draw, including scale `0`. The
bridge is fit on the noisy training coordinates, with penalty `1` and sigma
`0`. The pass rule is greater than `-0.05` on the 368 PA images. Draws are
2000, seed 42.

The reconstructor uses the section 2.28 architecture and trains on the 3000
unlabeled images. Its input is that cell's noisy coordinates. Images are one
grayscale channel, bilinear 64 by 64, scaled to `[0, 1]`. The baselines are
the mean training image and the mean training image of the same view. LPIPS
is not a decision metric. The primary call is the view-conditional mean. A
reconstruction beats that baseline when the two-sided 95% interval of the
per-image SSIM advantage lies above zero. Draws are 2000, seed 42. PSNR is
reported and does not decide.

Sex and view use the section 2.20 logistic specification, fit on the 3000
and scored on the 506. Above chance means greater than `0.5`. They do not
choose the cell.

Among cells whose PA-only lower bound is greater than `-0.05`, the selected
cell has the smallest mean SSIM advantage over the view-conditional mean.
A tie keeps the smaller rank, then the smaller noise scale. That cell has
room for later CKKS error when the next larger noise scale at the same rank
also passes. A selected cell at the largest scale that passes has that room.
A cell at scale `0` whose next scale fails has none. No CKKS circuit is
started here. The reserved confirmation stays closed. The 0.4 symbol is not
tested. This section does not claim an epsilon-3 local-DP release.

## 2.32 Plaintext students — locked before any student score

No CKKS run belongs to this section. The reserved confirmation stays closed.
The target is the clean rank-1999 PCA of the first 2000 unlabeled patients.
The other 1000 cached images are projected onto that basis and may train a
student. The 506 analysis patients do not train a student. The oracle
bridge, penalty `1` and sigma `0`, stays fit on the clean coordinates. A
student does not refit it. The pass rule is greater than `-0.05` on the
368 PA images.

Student 1 freezes the first `L` InternViT blocks, for `L` of 2, 4, and 8.
The head mean-pools that block's tokens and applies one linear map to 1999
coordinates. The blocks stay at their published weights. The head uses Adam,
learning rate `0.001`, 10 epochs, batch size 32, and seed 812. No epoch or
width is chosen after a score.

Student 2 is the registered `medical-polynomial-vit` at its default width
64, 3 layers, 4 by 4 patches, and degree schedule `[1, 1, 1]`. The grayscale
film is repeated to three channels and resized to 224. A new linear head
maps the pooled student tokens to 1999 coordinates. Training uses Adam,
learning rate `0.001`, 20 epochs, batch size 8, and seed 812. The head is
not initialized from a promoted checkpoint.

Each student is scored with the section 2.31 reconstructor on its predicted
coordinates, and with the section 2.20 logistic specification for sex and
view. Student 1 also reports that attack on the mean-pooled block output.
The full spatial grid is not a release in this section. A later CKKS attempt
waits until a student passes the PA-only rule.
If no student passes, no circuit is started. The 0.4 symbol is not tested.

## 2.33 Frontier revision of 2026-10-10 — locked before any frontier score

Section 2.31 and section 2.32 stay in place. This revision is dated
2026-10-10. It is the rule for scores that have not been run. The reserved
confirmation stays closed.

The bridge reports two fits. The mismatch fit trains on clean coordinates,
with penalty `1` and sigma `0`, and noise is added only at test time. The
refit trains on the noisy coordinates. Its penalty is `1 + n * v`, where
`v` is that coordinate's squared noise scale and `n` is the training count.
When every coordinate has the same variance, this is the diagonal form of
`DiagnosticPrefixProjector.fit`. The pass rule uses the refit. The mismatch
fit is a reference and does not pass a cell.

Each rank and each noise scale trains its own reconstructor on that noisy
release. A clean-coordinate reconstructor is not the leakage of a noisy
cell. The noise is scaled by the coordinate standard deviation. There is
no clipping and no sensitivity bound.
This noise is not differential privacy.
It is not an epsilon entry in the 0.3 ledger.

A frontier cell repeats the five section 2.30 training draws. The cell
passes when at least 4 of the 5 refit lower bounds are greater than
`-0.05`. One draw does not pass a cell. The leakage baselines are the
view-conditional mean image and a reconstruction from the
256 native tokens. The token reconstruction is a leakage ceiling, not a candidate
release. A release is recognizable when the lower end of the two-sided
95% interval of the paired SSIM advantage over the view-conditional mean
is greater than `0`. Draws are 2000, seed 42.

The cached 3000 patients may be scored as one nested prefix. That prefix
contains the published first 2000, so it is not an independent point. Its
ranks are 64, 128, 256, and `n_train - 1`. It does not use the five-draw
rule, and it does not open the confirmation. The 8000 prefix stays
unscored.

Before any confirmation image is downloaded, power at a true difference of
`-0.025` is recorded for the 301 reserved PA positives. The standard error
is the passing cell's own PA-only bootstrap standard deviation, scaled with
the section 2.19 approximation. Recording that power does not open the
reserve.

A plaintext pass of student 1 means the frozen early blocks linearly
predict the coordinates. It does not measure polynomial replacements of
softmax, LayerNorm, or GELU. Such a pass does not start a CKKS circuit.
Each student still receives the reconstruction attack above, including the
mean-pooled block output for student 1. Utility alone does not authorize a
circuit.

The cost anchor is one measured block, not an estimate. The block is
`MedFHEBlock`, width 64, 4 tokens, polynomial activation of degree 1,
polynomial norm of degree 2, token mixing on, and CADN off. The CKKS
parameters are `N=32768`, scale bits 28, and depth budget 6, with
relinearization keys. The input seed is 812. One forward is timed. Context
construction is timed separately and is not the block time. A failed
forward is the recorded anchor. The depth budget is not raised in this
section.

## 2.34 Same draws and the depth gate — locked before those scores

Section 2.33 stays in place. This revision is dated 2026-10-10. The
reserved confirmation stays closed.

Every frontier cell uses the same five section 2.30 draws, in that stored
order. A cell does not draw a new seed. The noise generator remains NumPy
`PCG64` with seed 812, and scale `0` still consumes its draw.
The coordinate standard deviation uses divisor `n`.

The noise-zero refit on the published first 2000 patients at rank 1999 is a
regression check. Its PA-only difference must lie within absolute tolerance
`1e-6` of the published difference. At zero noise the penalty on the
unnormalized gram is `1`, the same term as `DiagnosticPrefixProjector.fit`
with sigma `0` and ridge `1`. A miss stops the frontier. The tolerance is
not widened after a miss.

The cost sweep uses widths 32, 64, and 128 and token counts 4, 8, and 16.
The block and the CKKS parameters stay those of the section 2.33 anchor.
One forward is timed at each pair. A failed pair is not replaced. The width
slope is ordinary least squares of log forward time on log width at 4
tokens. The token slope uses width 64. Those slopes are not a full
InternViT time.
TenSEAL does not bootstrap.

The student depth gate uses depth budget `6`. The published block consumed
output depth `5`. Ciphertext-plaintext multiplies count toward that budget.
A student whose blocks would consume more than `6` levels does not enter
the circuit stage, even when its plaintext AUC passes. Replacing softmax,
LayerNorm, or GELU requires a new plaintext PA-only score. The native score
is not reused. No cell passing both gates is an expected result of this
section. That result does not start a CKKS circuit. The reserved
confirmation stays closed.

## 2.35 Depth before any student score — locked before that score

Section 2.34 stays in place. This revision is dated 2026-10-10. The
reserved confirmation stays closed.

The measured parameter set is `N=32768`, scale bits 28, and depth budget
`6`. TenSEAL does not bootstrap. One measured block consumes output depth
`5`. A circuit student is eligible only when its multiplicative depth is
at most `5`. A ciphertext-plaintext multiply adds one level. A
ciphertext-ciphertext multiply adds one level.

Replacing each of the first `L` InternViT blocks by one measured block
costs `5 * L` levels. Those counts are 10, 20, and 40 for `L` of 2, 4,
and 8. Those three students are not scored. A native LayerNorm, softmax,
or GELU is not treated as cheaper than that block.

The scored circuit students predict rank-1999 coordinates. The pixel input
is one grayscale channel, bilinear 64 by 64, scaled to `[0, 1]`. The
linear student is one ciphertext-plaintext map, depth `1`. The square
student appends the elementwise square and then one linear map, depth `2`.
Both use ridge penalty `1` and sigma `0` on the unnormalized gram. Training
uses the cached 3000 images. The targets are that draw's PCA coordinates.
Images in the 3000 outside the draw are projected onto that basis. The
bridge is the oracle bridge of that draw. A student does not refit it. A
student passes when at least 4 of the 5 draws have a PA-only lower bound
greater than `-0.05`.

The same two heads, fed by the mean of the first InternViT block tokens,
are an attribution control. They are not circuit candidates. If every
eligible circuit student fails, the conclusion is no feasible student
at depth at most `5`.

A longer modulus chain in this library still cannot bootstrap. A
bootstrapping library is a different runtime. An interactive client refresh
removes the depth limit and shows intermediate activations to the client.
That path is not a pure FHE claim. The 0.4 symbol is not tested.

## 2.36 Passing-cell attack — locked before any attack score (2026-10-12)

Section 2.28 is unchanged. Its 8000-patient prefix is not on disk. This
attack trains on the 3000 images that exist and scores the 506 analysis
patients. The inputs are the 256 native tokens, rank 1024 and rank 1999.
Both coordinate releases are zero-noise principal components of the first
2000 patients. The other 1000 training images and the 506 are projected
onto that basis. No other rank is added. The architecture, optimizer,
image size, and SSIM rule are the ones in section 2.28. The first
transposed convolution maps 128 channels to 64. The second maps 64 channels
to 1. Each attacker is initialized from seed 812, so training order does
not change a model.

The baselines are the mean training image and the view-conditional mean.
The primary call is the view-conditional mean. A reconstruction beats a
baseline when the two-sided 95% interval of the paired SSIM advantage lies
above zero, which means the lower end is greater than `0`. PSNR is reported
and does not decide. This attack does not choose a release.

Sex, age, and view use the section 2.20 specification, fit on the 3000 and
scored on the 506. The attribute vectors are the token mean, rank 1024, and
rank 1999. The full token matrix is the reconstruction input, not a second
logistic model. The reserved confirmation patients are not used.

## 2.37 Fine noise budget — locked before any new cell (2026-10-12)

The scored frontier already contains scales `0`, `0.25`, `0.5`, and `1`.
Those cells are not rerun. This section adds scales `0.05` and `0.1` at
rank 1024 and rank 1999 only, on the same five draws, with the same refit
penalty and the same rule of at least 4 of the 5. The noise generator is a
new stream, NumPy PCG64 seed 812, ordered by draw, then rank, then scale.
Scale `0` is the published point of the budget and is not drawn again.

The error budget of a rank is the largest of these scales that still
passes. It is the joint tolerance of a student, a polynomial approximation,
and CKKS error. This noise is not differential privacy. The reserved
confirmation patients are not used.

## 2.38 Pixel R² pre-check — locked before the regression (2026-10-12)

The depth-1 and depth-2 students in section 2.35 still run. Before they
do, one unconstrained ridge maps the 64 by 64 pixels to the rank-1999
coordinates of section 2.36. The penalty is `1`. It is fit on the 3000 and
scored on the 506. R² is reported on coordinate segments `0:64`, `64:256`,
`256:1024`, and `1024:1999`. R² uses the training mean of each coordinate
as its baseline. The segment standard deviation is the root mean square of
the per-coordinate training standard deviations, divisor `n`. A segment is
far from the failed `0.25` noise
scale when its root-mean-square error exceeds `0.25` times that segment's
training standard deviation. This comparison is a heuristic. It does not
cancel a student, and it is not a theorem.

## 2.39 Polynomial vision tower — locked before any approximation score (2026-10-12)

This section is not a circuit candidate. It asks whether a runtime that
supports bootstrapping could keep the PA-only margin after the vision
nonlinearities are replaced. The tower is the full InternViT, 24 blocks,
in plaintext, with the published linear weights left unchanged.

GELU is replaced by `0.1365 + 0.5 x + 0.2045 x^2`. Those coefficients are
not trained. Softmax is replaced by the square of the attention scores
divided by their row sum. The division is not a polynomial, so each softmax
counts as one bootstrap. Attention uses the naive path, not flash attention.
LayerNorm keeps its published scale and shift. Its reciprocal square root is
replaced by `1 - u/2 + 3 u^2/8`, where `u` is `variance - 1`. The mean and
the variance are not polynomials, so each LayerNorm counts as one bootstrap.
Query-key normalization is off. Each of the 24 blocks has two
normalizations, one softmax, and one GELU. The projector that emits the 256
release tokens has one LayerNorm and one GELU, with the same replacements.
The pre-written count is 73 bootstraps.
The multiplicative depth is `172`.
The count is not reduced after the score.

The bridge and the PA-only rule are the section 2.36 basis and the section
2.37 pass rule: at least 4 of the 5 draws, lower bound greater than
`-0.05`. The approximation error is then read against the section 2.37
budget. A pass here is not an FHE result. The reserved confirmation
patients are not used.

## 3. Roadmap and acceptance gates

This is the historical 0.3/0.4 development roadmap, not the current project plan.
The 0.4 finite-state continuation was stopped. See [STATUS.md](docs/handoff/STATUS.md)
for current research status and [CLAIMS_AUDIT.md](docs/handoff/CLAIMS_AUDIT.md) for
approved claim boundaries.

### Phase 0 — diagnose the bottleneck (plaintext, no FHE) — **DONE**

`experiments/diagnose_release_sufficiency.py` fits the shared linear bridge from four
release representations to the frozen native tokens and reports CI-backed advantages
over the constant-mean prefix. Result on `qwen2vl-2b`
(`experiments/results/release_sufficiency_qwen2vl-2b.json`, local-only):

| Release | MSE reduction over mean | 95% CI | Beats mean? |
|---|---:|---|:--:|
| constant mean | 0 | [0, 0] | baseline |
| oracle PCA rank 4 | +8.79% | [+6.27%, +11.14%] | yes |
| oracle PCA rank 16 | +11.80% | [+8.47%, +14.86%] | yes |
| oracle PCA rank 64 | +13.78% | [+10.66%, +16.73%] | yes |
| 0.3 quadratic release (clean) | −7.01% | [−11.48%, −3.37%] | no (worse) |
| 0.3 quadratic release + DP | +0.04% | [−0.12%, +0.26%] | no |

**Localization: possibility A (release fidelity).** The *same* linear bridge already
extracts a significant image-dependent signal from as few as four oracle image
dimensions, so the bridge/cohort is not the fundamental blocker; the deployed 0.3
low-rank quadratic release is what fails to carry a bridge-usable signal. Effort
therefore belongs on **release fidelity** (the 0.4 finite-state target), not on
enlarging the projector. The PCA rows are an oracle upper bound, not an achievable
encrypted release.

### Phase 1 — validate the 0.4 core hypothesis (development cohort) — **FAILED, recorded**

`experiments/validate_clinical_release.py` trains the finite-state codec under the frozen
configuration above and scores it on the Phase-0 yardstick. The cohort has no pixel-level
region annotation, so supervision is a documented frozen teacher: the native tower's
16×16 tokens are mean-pooled to the 4×4 grid, grouped into the five public slots, and
each region is reduced to four principal directions with basis and [0, 1] scaling fit on
**training patients only**. Whole-image labels are not copied onto regions. The oracle
PCA rank-4 row reproduces Phase 0 exactly (+8.79%), confirming the same yardstick.

Result on `qwen2vl-2b`
(`experiments/results/clinical_release_validation_qwen2vl-2b.json`, local-only); only
`codec_symbol` is the pre-registered gate:

| Release | MSE reduction over mean | 95% CI | Beats mean? |
|---|---:|---|:--:|
| Constant mean | 0 | [0, 0] | baseline |
| Oracle PCA rank 4 | +8.79% | [+6.27%, +11.14%] | yes |
| 0.4 clean state (before the channel) | −15.50% | [−20.19%, −9.91%] | no (worse) |
| **0.4 sampled symbol (the gate)** | **−8.48%** | **[−10.79%, −6.51%]** | **no (worse)** |
| 0.4 expected posterior | −9.60% | [−12.97%, −5.83%] | no (worse) |

**Gate: not passed.** The certified channel is not the cause — the argmax state alone is
already worse than the constant mean, and the channel only pulls it back toward the mean.
Training state accuracy is 1.0, so the atlas memorized its eight training clusters and
that assignment does not transfer to held-out patients. No test-set search was run.
Privacy confirmation, native non-inferiority and the multi-backend table stay blocked.

### Factorized disease anchor — failed, implementation removed

Same split, ridge, baseline, bootstrap, and image epsilon 3. Declared before the run:
5 slots × 4 quantile states, TorchXRayVision `densenet121-res224-all`, epsilon 3/5 per
slot. The source was removed after the result was recorded. The gate row is
`encoder_symbol` in
 `experiments/results/factorized_release_validation_qwen2vl-2b.json` (local-only).

| Release | MSE change vs mean | 95% CI | Beats mean? |
|---|---:|---|:--:|
| Factorized anchor, no channel | −14.19% | [−16.08%, −12.13%] | no (worse) |
| **Moment readout symbol (the gate)** | **−2.66%** | **[−3.23%, −1.98%]** | **no (worse)** |
| Moment readout expectation | +0.02% | [−0.04%, +0.09%] | no |

The clean medical anchor is already worse than the constant mean, and the moment
readout agrees with that anchor at chance (26.4% held out, 25% chance). Factorizing
the alphabet and importing a frozen disease basis did not put image-dependent signal
through this ridge.

### Native principal coordinates — continuous gate failed

`experiments/validate_manifold_release.py` fits the moment circuit to the four
training-only native principal coordinates. Discretization requires a noiseless MSE
gain of at least +5% with a patient interval that excludes zero. Measured continuous
change: **−6.91%**, 95% CI **[−15.08%, +1.11%]**. Lloyd-Max, the geometric mechanism,
and a one-region budget concentration were not run. The downstream gates stay blocked.

### Operator capacity — held-out transfer failed

`experiments/validate_operator_capacity.py` replaces joint gradient training. A missed
optimum on 99 images cannot prove a family empty, and a wide release beats the
training mean by rank. The shared stem is the top four training cross-covariance
directions of the patches against the aligned native cells (99.64% of that
cross-energy). Anatomical moments and a public Fourier quadratic (78 products, budget
96) are each reduced to four coordinates and scored with the same ridge. A training-only
Gaussian 4-vector is the in-sample rank control. Held-out result on `qwen2vl-2b`
(`experiments/results/operator_capacity_qwen2vl-2b.json`, local-only):

| Release | MSE change vs mean | 95% CI | Beats mean? |
|---|---:|---|:--:|
| Oracle PCA rank 4 | +8.79% | [+6.27%, +11.14%] | yes |
| Anatomical moments, optimal rank 4 | −55.91% | [−131.50%, −5.85%] | no (worse) |
| Fourier quadratic, optimal rank 4 | −289.58% | [−531.24%, −68.15%] | no (worse) |

In sample the oracle gains +15.57%, moments +11.97%, Fourier +14.80%, and the
Gaussian control +4.13%. Both circuits clear the rank artifact on the training
patients. Neither transfers. Discretization was not run. Later phases stay blocked.

### Pretrained-encoder learning curve — intercept at every size

`experiments/validate_pretrained_anchor.py` freezes TorchXRayVision DenseNet-121 and
fits only a ridge onto the fixed oracle PCA-4 coordinates. The penalty, including the
intercept, is chosen by leave-one-patient-out inside each training subset. Subsets are
the first 4, 8, 12, 16 and 20 patients of one seeded permutation. Held-out result on
`qwen2vl-2b`
(`experiments/results/pretrained_anchor_learning_curve_qwen2vl-2b.json`, local-only):

| Training patients | Oracle prefix | 95% CI | Beats mean? |
|---:|---:|---|:--:|
| 4 | −2.86% | [−5.86%, −0.59%] | no (worse) |
| 8 | +5.82% | [+3.94%, +7.64%] | yes |
| 12 | +7.01% | [+4.64%, +9.46%] | yes |
| 16 | +8.10% | [+5.72%, +10.55%] | yes |
| 20 | +8.79% | [+6.27%, +11.14%] | yes |

DenseNet selects the intercept at every row, so its held-out change is 0. The oracle
coordinates improve with sample size. This encoder does not carry the subspace. Later
phases stay blocked.

### Oracle report probe — no clinical gain measured

`experiments/validate_oracle_reports.py` generates a greedy `qwen2vl-2b` report from the
oracle PCA-4 ridge and from the constant-mean prefix, one image for every evaluation
patient. The cohort has no reference report, so RadGraph is not computed. The constant-mean
report is one normal-chest paragraph for all 7 patients. Five oracle reports stay on that
paragraph. Two leave it and do not name the finding string. A nonlinear probe and a
clinical-label target were not run. Later phases stay blocked.

The same generator was then run at ranks 16 and 64
(`experiments/results/oracle_report_width_qwen2vl-2b.json`, local-only). Rank 16 stays on the
normal paragraph for 5/7 patients, the same count as rank 4. Rank 64 stays for 3/7.
Keyword hits stay 0/3 at every rank. No width in {4, 16, 64} names a labeled finding.

`experiments/validate_native_ceiling.py` then changes only the native-injection prompt and
decoder (`experiments/results/native_ceiling_qwen2vl-2b.json`, local-only). Greedy decoding of the
open prompt reproduces 1/3. Five-sample majority vote stays at 1/3. A direct checklist prompt
scores 3/3 and 11/11, and the training-mean prefix produces the same affirmative pneumonia
sentence, as do all 17 unlabeled evaluation images. The higher count is the checklist. The
evaluation split has 3 labeled patients.

`experiments/validate_native_backend.py` repeats the open prompt and native injection on
`internvl2-2b` (`experiments/results/native_ceiling_internvl2-2b.json`, local-only). Affirmative
hits are 0/3, 0/11, and 0/3 by patient majority. The development cohort's 20 labeled
patients are all pneumonia subtypes, so a new split cannot add another disease. MedGemma
was not run. A regional-token release was not started. Later phases stay blocked.

### Phase 2 — adversarial privacy confirmation (historical plan; not current status)

On patient-disjoint holdouts never used for tuning, run reconstruction (PSNR/SSIM/LPIPS)
and attribute/membership/model-extraction attacks against the 0.4 single symbol at the
frozen `epsilon`. **Gate:** reconstruction well below the full-embedding baseline and
attack AUC within preregistered limits. A failed confirmation may not be reused after
redesign. See the current status page before treating this historical plan as active.

### Phase 3 — one-backend clinical utility (historical plan)

For a single backend, compare native VLM vs the FHE-release + LLM pipeline with
report-specific metrics (RadGraph / CheXbert F1) under `paired_patient_noninferiority`,
plus one end-to-end client-side cost measurement. **Gate:** primary metric within the
0.05 margin of the native tower.

### Phase 4 — transferability (historical plan)

Reproduce P1 on a public medical ViT backbone not tuned in-house to show the joint
compiler transfers. Nice-to-have, not a foundation.

## 4. Historical 0.4 guardrails

* No new circuits or modalities before non-inferiority and confirmation. A MedGemma native-ceiling probe is allowed only under §2.1 and does not fill the claim table.
* Do not code `todo` or a sealed holdout study as a pneumonia-negative to force an AUC.
* Do not attribute the software tests or the 0.3 FHE timings to 0.4 clinical performance.
* Do not search hyper-parameters on the test set.
* Do not pursue a whole-model FHE-LLM.
