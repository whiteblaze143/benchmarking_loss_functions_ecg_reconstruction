# Blinded Electrophysiologist ECG Adjudication Packet

**Prepared for:** Dr. Christopher Cheung (Cardiac Electrophysiology Review)
**Cases Included:** 24 blinded rhythm-transition strips (`[-12 s, +18 s]` around annotated AF termination `t = 0 s`, single surface ECG lead, bandpass filtered 0.5–35 Hz)
**Blinding:** Cohort source, patient identifier, upstream rhythm label, and downstream reinitiation timing are completely withheld from all strip images and the adjudication sheet (`ep_review_sheet.csv`).

## Adjudication Instructions
For each strip (`EP_01` through `EP_24`), classify:
1. **`adjudicated_post_rhythm`** (select one):
   - `CONFIRMED_SINUS`: Clear sinus P waves / regular sinus rhythm restored after `t = 0 s`
   - `NON_SINUS_ATRIAL_AFL_AT`: Organized atrial arrhythmia (atrial flutter or atrial tachycardia) without sinus restoration
   - `VENTRICULAR_ECTOPIC_NO_SINUS_RESTORATION`: Ventricular rhythm, bigeminy/trigeminy, or ectopy interruption while underlying atrial fibrillation likely persists
   - `ARTIFACT_OR_NOISE`: Signal dropout, motion artifact, or baseline shift mimicking rhythm transition
   - `INDETERMINATE`: Cannot reliably determine post-transition rhythm from available single-lead strip
2. **`genuine_af_termination`** (`YES`, `NO`, or `INDETERMINATE`)
3. **`reviewer_confidence_1_to_3`** (`1` = low, `2` = moderate, `3` = high)

---

## Blinded ECG Strips

### EP_01
![EP_01](figures/ep_review/EP_01.png)
- `adjudicated_post_rhythm`: `[ ] CONFIRMED_SINUS  [ ] NON_SINUS_ATRIAL_AFL_AT  [ ] VENTRICULAR_ECTOPIC_NO_SINUS_RESTORATION  [ ] ARTIFACT_OR_NOISE  [ ] INDETERMINATE`
- `genuine_af_termination`: `[ ] YES  [ ] NO  [ ] INDETERMINATE` | Confidence (`1–3`): `___` | Notes: `________________________`

### EP_02
![EP_02](figures/ep_review/EP_02.png)
- `adjudicated_post_rhythm`: `[ ] CONFIRMED_SINUS  [ ] NON_SINUS_ATRIAL_AFL_AT  [ ] VENTRICULAR_ECTOPIC_NO_SINUS_RESTORATION  [ ] ARTIFACT_OR_NOISE  [ ] INDETERMINATE`
- `genuine_af_termination`: `[ ] YES  [ ] NO  [ ] INDETERMINATE` | Confidence (`1–3`): `___` | Notes: `________________________`

### EP_03
![EP_03](figures/ep_review/EP_03.png)
- `adjudicated_post_rhythm`: `[ ] CONFIRMED_SINUS  [ ] NON_SINUS_ATRIAL_AFL_AT  [ ] VENTRICULAR_ECTOPIC_NO_SINUS_RESTORATION  [ ] ARTIFACT_OR_NOISE  [ ] INDETERMINATE`
- `genuine_af_termination`: `[ ] YES  [ ] NO  [ ] INDETERMINATE` | Confidence (`1–3`): `___` | Notes: `________________________`

### EP_04
![EP_04](figures/ep_review/EP_04.png)
- `adjudicated_post_rhythm`: `[ ] CONFIRMED_SINUS  [ ] NON_SINUS_ATRIAL_AFL_AT  [ ] VENTRICULAR_ECTOPIC_NO_SINUS_RESTORATION  [ ] ARTIFACT_OR_NOISE  [ ] INDETERMINATE`
- `genuine_af_termination`: `[ ] YES  [ ] NO  [ ] INDETERMINATE` | Confidence (`1–3`): `___` | Notes: `________________________`

### EP_05
![EP_05](figures/ep_review/EP_05.png)
- `adjudicated_post_rhythm`: `[ ] CONFIRMED_SINUS  [ ] NON_SINUS_ATRIAL_AFL_AT  [ ] VENTRICULAR_ECTOPIC_NO_SINUS_RESTORATION  [ ] ARTIFACT_OR_NOISE  [ ] INDETERMINATE`
- `genuine_af_termination`: `[ ] YES  [ ] NO  [ ] INDETERMINATE` | Confidence (`1–3`): `___` | Notes: `________________________`

### EP_06
![EP_06](figures/ep_review/EP_06.png)
- `adjudicated_post_rhythm`: `[ ] CONFIRMED_SINUS  [ ] NON_SINUS_ATRIAL_AFL_AT  [ ] VENTRICULAR_ECTOPIC_NO_SINUS_RESTORATION  [ ] ARTIFACT_OR_NOISE  [ ] INDETERMINATE`
- `genuine_af_termination`: `[ ] YES  [ ] NO  [ ] INDETERMINATE` | Confidence (`1–3`): `___` | Notes: `________________________`

### EP_07
![EP_07](figures/ep_review/EP_07.png)
- `adjudicated_post_rhythm`: `[ ] CONFIRMED_SINUS  [ ] NON_SINUS_ATRIAL_AFL_AT  [ ] VENTRICULAR_ECTOPIC_NO_SINUS_RESTORATION  [ ] ARTIFACT_OR_NOISE  [ ] INDETERMINATE`
- `genuine_af_termination`: `[ ] YES  [ ] NO  [ ] INDETERMINATE` | Confidence (`1–3`): `___` | Notes: `________________________`

### EP_08
![EP_08](figures/ep_review/EP_08.png)
- `adjudicated_post_rhythm`: `[ ] CONFIRMED_SINUS  [ ] NON_SINUS_ATRIAL_AFL_AT  [ ] VENTRICULAR_ECTOPIC_NO_SINUS_RESTORATION  [ ] ARTIFACT_OR_NOISE  [ ] INDETERMINATE`
- `genuine_af_termination`: `[ ] YES  [ ] NO  [ ] INDETERMINATE` | Confidence (`1–3`): `___` | Notes: `________________________`

### EP_09
![EP_09](figures/ep_review/EP_09.png)
- `adjudicated_post_rhythm`: `[ ] CONFIRMED_SINUS  [ ] NON_SINUS_ATRIAL_AFL_AT  [ ] VENTRICULAR_ECTOPIC_NO_SINUS_RESTORATION  [ ] ARTIFACT_OR_NOISE  [ ] INDETERMINATE`
- `genuine_af_termination`: `[ ] YES  [ ] NO  [ ] INDETERMINATE` | Confidence (`1–3`): `___` | Notes: `________________________`

### EP_10
![EP_10](figures/ep_review/EP_10.png)
- `adjudicated_post_rhythm`: `[ ] CONFIRMED_SINUS  [ ] NON_SINUS_ATRIAL_AFL_AT  [ ] VENTRICULAR_ECTOPIC_NO_SINUS_RESTORATION  [ ] ARTIFACT_OR_NOISE  [ ] INDETERMINATE`
- `genuine_af_termination`: `[ ] YES  [ ] NO  [ ] INDETERMINATE` | Confidence (`1–3`): `___` | Notes: `________________________`

### EP_11
![EP_11](figures/ep_review/EP_11.png)
- `adjudicated_post_rhythm`: `[ ] CONFIRMED_SINUS  [ ] NON_SINUS_ATRIAL_AFL_AT  [ ] VENTRICULAR_ECTOPIC_NO_SINUS_RESTORATION  [ ] ARTIFACT_OR_NOISE  [ ] INDETERMINATE`
- `genuine_af_termination`: `[ ] YES  [ ] NO  [ ] INDETERMINATE` | Confidence (`1–3`): `___` | Notes: `________________________`

### EP_12
![EP_12](figures/ep_review/EP_12.png)
- `adjudicated_post_rhythm`: `[ ] CONFIRMED_SINUS  [ ] NON_SINUS_ATRIAL_AFL_AT  [ ] VENTRICULAR_ECTOPIC_NO_SINUS_RESTORATION  [ ] ARTIFACT_OR_NOISE  [ ] INDETERMINATE`
- `genuine_af_termination`: `[ ] YES  [ ] NO  [ ] INDETERMINATE` | Confidence (`1–3`): `___` | Notes: `________________________`

### EP_13
![EP_13](figures/ep_review/EP_13.png)
- `adjudicated_post_rhythm`: `[ ] CONFIRMED_SINUS  [ ] NON_SINUS_ATRIAL_AFL_AT  [ ] VENTRICULAR_ECTOPIC_NO_SINUS_RESTORATION  [ ] ARTIFACT_OR_NOISE  [ ] INDETERMINATE`
- `genuine_af_termination`: `[ ] YES  [ ] NO  [ ] INDETERMINATE` | Confidence (`1–3`): `___` | Notes: `________________________`

### EP_14
![EP_14](figures/ep_review/EP_14.png)
- `adjudicated_post_rhythm`: `[ ] CONFIRMED_SINUS  [ ] NON_SINUS_ATRIAL_AFL_AT  [ ] VENTRICULAR_ECTOPIC_NO_SINUS_RESTORATION  [ ] ARTIFACT_OR_NOISE  [ ] INDETERMINATE`
- `genuine_af_termination`: `[ ] YES  [ ] NO  [ ] INDETERMINATE` | Confidence (`1–3`): `___` | Notes: `________________________`

### EP_15
![EP_15](figures/ep_review/EP_15.png)
- `adjudicated_post_rhythm`: `[ ] CONFIRMED_SINUS  [ ] NON_SINUS_ATRIAL_AFL_AT  [ ] VENTRICULAR_ECTOPIC_NO_SINUS_RESTORATION  [ ] ARTIFACT_OR_NOISE  [ ] INDETERMINATE`
- `genuine_af_termination`: `[ ] YES  [ ] NO  [ ] INDETERMINATE` | Confidence (`1–3`): `___` | Notes: `________________________`

### EP_16
![EP_16](figures/ep_review/EP_16.png)
- `adjudicated_post_rhythm`: `[ ] CONFIRMED_SINUS  [ ] NON_SINUS_ATRIAL_AFL_AT  [ ] VENTRICULAR_ECTOPIC_NO_SINUS_RESTORATION  [ ] ARTIFACT_OR_NOISE  [ ] INDETERMINATE`
- `genuine_af_termination`: `[ ] YES  [ ] NO  [ ] INDETERMINATE` | Confidence (`1–3`): `___` | Notes: `________________________`

### EP_17
![EP_17](figures/ep_review/EP_17.png)
- `adjudicated_post_rhythm`: `[ ] CONFIRMED_SINUS  [ ] NON_SINUS_ATRIAL_AFL_AT  [ ] VENTRICULAR_ECTOPIC_NO_SINUS_RESTORATION  [ ] ARTIFACT_OR_NOISE  [ ] INDETERMINATE`
- `genuine_af_termination`: `[ ] YES  [ ] NO  [ ] INDETERMINATE` | Confidence (`1–3`): `___` | Notes: `________________________`

### EP_18
![EP_18](figures/ep_review/EP_18.png)
- `adjudicated_post_rhythm`: `[ ] CONFIRMED_SINUS  [ ] NON_SINUS_ATRIAL_AFL_AT  [ ] VENTRICULAR_ECTOPIC_NO_SINUS_RESTORATION  [ ] ARTIFACT_OR_NOISE  [ ] INDETERMINATE`
- `genuine_af_termination`: `[ ] YES  [ ] NO  [ ] INDETERMINATE` | Confidence (`1–3`): `___` | Notes: `________________________`

### EP_19
![EP_19](figures/ep_review/EP_19.png)
- `adjudicated_post_rhythm`: `[ ] CONFIRMED_SINUS  [ ] NON_SINUS_ATRIAL_AFL_AT  [ ] VENTRICULAR_ECTOPIC_NO_SINUS_RESTORATION  [ ] ARTIFACT_OR_NOISE  [ ] INDETERMINATE`
- `genuine_af_termination`: `[ ] YES  [ ] NO  [ ] INDETERMINATE` | Confidence (`1–3`): `___` | Notes: `________________________`

### EP_20
![EP_20](figures/ep_review/EP_20.png)
- `adjudicated_post_rhythm`: `[ ] CONFIRMED_SINUS  [ ] NON_SINUS_ATRIAL_AFL_AT  [ ] VENTRICULAR_ECTOPIC_NO_SINUS_RESTORATION  [ ] ARTIFACT_OR_NOISE  [ ] INDETERMINATE`
- `genuine_af_termination`: `[ ] YES  [ ] NO  [ ] INDETERMINATE` | Confidence (`1–3`): `___` | Notes: `________________________`

### EP_21
![EP_21](figures/ep_review/EP_21.png)
- `adjudicated_post_rhythm`: `[ ] CONFIRMED_SINUS  [ ] NON_SINUS_ATRIAL_AFL_AT  [ ] VENTRICULAR_ECTOPIC_NO_SINUS_RESTORATION  [ ] ARTIFACT_OR_NOISE  [ ] INDETERMINATE`
- `genuine_af_termination`: `[ ] YES  [ ] NO  [ ] INDETERMINATE` | Confidence (`1–3`): `___` | Notes: `________________________`

### EP_22
![EP_22](figures/ep_review/EP_22.png)
- `adjudicated_post_rhythm`: `[ ] CONFIRMED_SINUS  [ ] NON_SINUS_ATRIAL_AFL_AT  [ ] VENTRICULAR_ECTOPIC_NO_SINUS_RESTORATION  [ ] ARTIFACT_OR_NOISE  [ ] INDETERMINATE`
- `genuine_af_termination`: `[ ] YES  [ ] NO  [ ] INDETERMINATE` | Confidence (`1–3`): `___` | Notes: `________________________`

### EP_23
![EP_23](figures/ep_review/EP_23.png)
- `adjudicated_post_rhythm`: `[ ] CONFIRMED_SINUS  [ ] NON_SINUS_ATRIAL_AFL_AT  [ ] VENTRICULAR_ECTOPIC_NO_SINUS_RESTORATION  [ ] ARTIFACT_OR_NOISE  [ ] INDETERMINATE`
- `genuine_af_termination`: `[ ] YES  [ ] NO  [ ] INDETERMINATE` | Confidence (`1–3`): `___` | Notes: `________________________`

### EP_24
![EP_24](figures/ep_review/EP_24.png)
- `adjudicated_post_rhythm`: `[ ] CONFIRMED_SINUS  [ ] NON_SINUS_ATRIAL_AFL_AT  [ ] VENTRICULAR_ECTOPIC_NO_SINUS_RESTORATION  [ ] ARTIFACT_OR_NOISE  [ ] INDETERMINATE`
- `genuine_af_termination`: `[ ] YES  [ ] NO  [ ] INDETERMINATE` | Confidence (`1–3`): `___` | Notes: `________________________`
