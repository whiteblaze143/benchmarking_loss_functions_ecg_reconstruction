dont forget to consider the Ansari et al model too :[Skip to main content](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#main)[Become a member](http://professional.heart.org/professional/membership/membership-tiers.jsp)


what can we use to imrpove our model think SOTA futurisc acadmeic phd high level researcher novelty is key Development of the 3DRECON-QT ModelThe 3DRECON-QT model utilizes an encoder-decoder framework trained via a multi-task
learning objective. This design forces the model to learn a physiologically grounded latent
representation by simultaneously reconstructing the full 12-lead ECG and predicting the scalar
QT interval from a single input lead.The overall architecture takes in an input of a 10-second single-lead ECG waveform, sampled at
500Hz. The model employs an encoder (SE-ResNeXT) that distills the input signal into a latent
representation. Two separate heads operate simultaneously, one for 12-lead ECG
reconstruction, and one for QT interval prediction (Figure 2). The reconstruction head
synthesizes the standard 12-lead ECG by treating each lead as a unique spatial ‘viewpoint’ of
the heart’s electrical activity with respect to the thorax. A ThetaEncoder first generates a spatial
embedding from a target ECG lead’s position, mathematically represented as a spherical
coordinate pair (𝜃𝜃, 𝜙𝜙). The spatial information is then fused with the latent representation of the
input single-lead ECG, effectively conditioning it with the information of the target viewpoint. An
ECGDecoder then translates this conditioned representation into the corresponding waveform.
The QT prediction head bypasses spatial conditioning and instead uses the complete latent
representation to preserve available spatiotemporal information. This latent representation is
then fed into a TransformerDecoder, which uses self-attention to identify salient features and
their relationships across the cardiac cycle for a globally-informed QT assessment. The2resulting embedding is passed to a final Multi-Layer Perceptron (MLP) regressor to output the
scalar QT prediction.The model uses a weighted combination of two loss functions, one for the reconstruction and
one for the QT regression. Optimization details included a stochastic gradient descent optimizer
with an initial learning rate of 1e-3 with cosine annealing with warm restarts. The regularization
was L2 weight decay of 1e-5 and batch size of 128. To prevent overfitting, the model was
trained for up to 100 epochs with an early stopping protocol. The final checkpoint was selected
based on the best Pearson correlation for QTc prediction on the validation set. The model was
implemented in Python using the PyTorch framework (v2.3.1).
[Volunteer](http://professional.heart.org/professional/registration/volunteerForm.jsp)
[Donate](https://mygiving.heart.org/-/XZTZBDFD?s_src=20U2W1UEMG&s_subsrc=ahajournals_top_donate_button)
[Journals](https://www.ahajournals.org/action/showPublications)
[Browse](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#)
[Resources](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#)
[Information](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#)
[Alerts](https://www.ahajournals.org/action/showPreferences?menuTab=Alerts)
[Shopping cart with0item](https://www.ahajournals.org/action/showCart?FlowID=1)
[Sign in](https://www.ahajournals.org/action/ssostart?idp=https://ahasso.heart.org&redirectUri=%2Fdoi%2F10.1161%2FCIRCULATIONAHA.125.077494)[REGISTER](https://www.ahajournals.org/action/ssostart?idp=https://ahasso.heart.org&redirectUri=%2Fdoi%2F10.1161%2FCIRCULATIONAHA.125.077494)
[Current Issue](https://www.ahajournals.org/toc/circ/current)
[Archive](https://www.ahajournals.org/loi/circ)
[Journal Information](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#)
[Features](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#)
Reference #1
Research Article
Originally Published 8 November 2025
Free Access
Deep Learning–Based Continuous QT Monitoring to Identify High-Risk Prolongation Events After Class III Antiarrhythmic Initiation
[Rayan A. Ansari, BS](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#con1) [https://orcid.org/0000-0001-6153-7272](https://orcid.org/0000-0001-6153-7272), [Sabyasachi Bandyopadhyay, PhD](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#con2) [https://orcid.org/0009-0003-4825-646X](https://orcid.org/0009-0003-4825-646X), [Rishi K. Trivedi, MD, PhD](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#con3), [Kelly A. Brennan, MS](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#con4) [https://orcid.org/0000-0002-7698-7353](https://orcid.org/0000-0002-7698-7353), [Xichong Liu, MD](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#con5) [https://orcid.org/0000-0002-8772-5623](https://orcid.org/0000-0002-8772-5623), [Prasanth Ganesan, PhD](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#con6) [https://orcid.org/0000-0002-1885-0690](https://orcid.org/0000-0002-1885-0690), [J. Weston Hughes, PhD](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#con7) [https://orcid.org/0009-0001-9810-6094](https://orcid.org/0009-0001-9810-6094), … Show All … , and [Albert J. Rogers, MD, MBA](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#con15) [https://orcid.org/0000-0001-6585-534X](https://orcid.org/0000-0001-6585-534X) [rogersaj@stanford.edu](mailto:rogersaj@stanford.edu)[Author Info & Affiliations](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#tab-contributors)
Circulation
[Volume 153, Number 1](https://www.ahajournals.org/toc/circ/153/1)
[https://doi.org/10.1161/CIRCULATIONAHA.125.077494](https://doi.org/10.1161/CIRCULATIONAHA.125.077494)
[1,2126](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#)Metrics
Total Downloads1,212
Last 12 Months1,212
Total Citations6
Last 12 Months6
[Abstract](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#abstract)
[Clinical Perspective](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#box-1-sec-1)
[Methods](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#sec-1)
[Results](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#sec-2)
[Discussion](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#sec-3)
[Acknowledgments](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#acknowledgments)
[Footnote](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#footnotes)
[Supplemental Material](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#supplementary-materials)
[References](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#bibliography)
[eLetters](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#comments)
[Information & Authors](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#core-collateral-info)
[Metrics & Citations](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#core-collateral-metrics)
[View Options](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#core-collateral-fulltext-options)
[References](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#core-collateral-references)
[Figures](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#core-collateral-figures)
[Tables](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#core-collateral-tables)
[Share](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#core-collateral-share)
Abstract
BACKGROUND:
Drug-induced QT prolongation after successful inpatient loading of class III antiarrhythmics may occur during routine outpatient care. Insertable cardiac monitors offer continuous signals but are limited by single-lead configuration. We hypothesized that a spatially aware deep learning system (3DRECON-QT) can reconstruct spatial information from a single lead vector to quantify QT/QTc and identify high-risk prolongation.
METHODS:
We developed 3DRECON-QT using a multitask encoder–decoder that ingests a 10-s single-lead signal, reconstructs 12 leads, and predicts QT/QTc. The model was developed using 12-lead ECGs with clinician-adjudicated QT/RR from a large health system and tested in an external center with different ECG hardware. Continuous monitoring performance was assessed in a public dofetilide‐loading data set with serial ECGs. In a real-world cohort of outpatients on dofetilide or sotalol presenting to the hospital or emergency room for any reason, rates of ventricular arrhythmias and QT prolongation were assessed. Device validation was tested in patients with insertable cardiac monitor recordings paired with clinical 12-lead ECGs.
RESULTS:
3DRECON-QT classified prolonged QTc from single-lead signals with area under the receiver operating characteristics curve, 0.942 (mean absolute error, 17.5 ms) in the internal test set and 0.943 (mean absolute error, 21.1 ms) externally. During continuous dofetilide monitoring, predictions correlated with ground truth (r, 0.851; mean absolute error, 17.8 ms; area under the receiver operating characteristics curve, 0.936 for prolonged QTc, 0.816 for ≥15% QTc rise). QTc prediction from true insertable cardiac monitor recordings showed r=0.824 and mean absolute error, 17.5 ms. In outpatients on class III antiarrhythmics (n=1676), 16.5% had high-risk QTc prolongation. Ventricular arrhythmia events were 3.97% versus 0.86% without prolongation (adjusted odds ratio, 4.24 [95% CI, 1.81–9.90]). 3DRECON-QT detected these events with area under the receiver operating characteristics curve 0.94 (F1 score, 0.60).
CONCLUSIONS:
A single-lead, deep-learning approach can achieve guideline-level measurement accuracy, enable continuous QTc surveillance from nonstandard ECG vectors, and identify clinically meaningful outpatient QTc prolongation associated with a >4-fold increase in serious ventricular arrhythmias. This strategy may enhance safety monitoring after class III antiarrhythmic initiation and support targeted intervention.
Clinical Perspective
What Is New?
•
A spatially aware deep learning model (3DRECON-QT) can reconstruct 12-lead electrocardiographic information from a single-lead, insertable cardiac monitor signal and accurately predict QT/QTc intervals.
•
External validation across health systems and hardware platforms showed robust generalization to identify prolonged QT.
•
In a real-world cohort of patients on class III antiarrhythmic drugs, 3DRECON-QT can identify QT prolongation that is frequent and associated with a 4-fold higher risk of ventricular arrhythmias.
What Are the Clinical Implications?
•
Continuous ambulatory QT monitoring may be achieved using existing cardiac monitor infrastructure without new hardware or significant patient effort, enabling enhanced drug safety surveillance.
•
Ambulatory surveillance may detect high-risk QT prolongation events after discharge from class III antiarrhythmic initiation to guide early intervention and dose adjustment.
•
This approach may provide the groundwork to shorten or obviate inpatient drug initiation for selected patients and support broader remote safety monitoring across other QT-prolonging medications.
The QT interval is a critical electrocardiographic marker for assessing the risk of life-threatening arrhythmias, including torsades de pointes and sudden cardiac death.[1](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#core-collateral-R1),[2](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#core-collateral-R2) However, its measurement requires a 12-lead ECG, which is impractical for continuous, long-term monitoring in ambulatory care settings.[3](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#core-collateral-R3)
Patients initiated on QT-prolonging drugs such as dofetilide or sotalol are monitored as inpatients, but QT prolongation may occur between periodic 12-lead ECGs, and may fluctuate beyond discharge in patients with electrolyte fluctuations such as those with renal dysfunction or heart failure.[4](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#core-collateral-R4) Insertable cardiac monitors (ICMs) provide a platform for continuous rhythm monitoring, yet their nonstandard, single-lead configuration precludes direct QT interval measurement according to society guidelines.[5](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#core-collateral-R5)
We hypothesized that 3DRECON-QT, a spatially informed, multitask architecture applied to single-lead ECG signals, can accurately reconstruct the 12-lead ECG to identify clinically relevant QTc prolongation and reveal hidden burdens of arrhythmia risk. We have previously shown that machine learning of intracardiac electrograms can identify patients at risk for sudden cardiac arrest and suggests cellular phenotypes associated with that risk.[6](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#core-collateral-R6) We have also presented a deep learning model of the 12-lead ECG alone to identify patients with wall motion abnormalities.[7](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#core-collateral-R7) This study extends these principles to reconstruct a multilead ECG from a nonstandard, narrowly spaced ECG signal (as obtained from an ICM or chest-worn patch monitor) to quantitatively measure QT interval in a broad group of patients. Further, in a patient population with class III antiarrhythmic (dofetilide or sotalol) exposure, we assess the incidence QT prolongation and the risk of significant arrhythmia events after successful load of the medications.
To test this, we aimed to: (1) develop and validate 3DRECON-QT using large and diverse internal and external ECG data sets, respectively; (2) assess its ability to detect drug-induced QT prolongation during continuous monitoring; and (3) quantify the real-world clinical risk associated with outpatient QT prolongation and validate the ability of 3DRECON-QT to identify high-risk events in this real-life cohort.
Methods
Study Populations and Data Sets
Model development was performed at Stanford University and tested at an independent external site (Cedars-Sinai Medical Center). The study was approved by the institutional review boards of both centers. A waiver of consent was granted for retrospective analysis. Across these sites and a public database, 5 distinct patient cohorts were established. These comprised the Stanford Model Development Cohort, Cedars-Sinai External Validation Cohort, Continuous QT Monitoring Cohort (ECG Effects of Ranolazine, Dofetilide, Verapamil, and Quinidine PhysioNet Dataset), Stanford Real-World Class III Antiarrhythmic Exposure Cohort, and the Stanford ICM Recording Cohort. The data that support the findings of this study are available from the corresponding author upon reasonable request. A dataset overview and model input for each analysis is available in [Table S1](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#supplementary-materials).
Stanford Model Development Cohort
We analyzed 794 813 12-lead ECG data obtained from 258 205 unique patients in the Stanford Hospital ECG database (iECG, Phillips, Amsterdam, The Netherlands). The data collection period extended from 2005 to 2018. Patients were randomly divided into training (80%), validation (10%), and testing (10%) cohorts to ensure all ECGs from any single patient were assigned exclusively to one data partition. The test set (10%) included 78 245 12-lead ECG recordings from 25 821 patients not previously seen by the model during training. Exclusion criteria included records in legacy formats before system migration, poor signal quality (>70% of leads 0-valued or flatlined), incomplete or absent machine annotations, and missing QT or RR interval measurements. All rhythms, including atrial and ventricular paced ECGs, were included. The consort diagram for the development and validation cohorts is provided in [Figure 1](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#F1).
Figure 1. Consort diagram showing data flow for 12-lead ECGs in the development (Stanford) and external testing cohorts (Cedars-Sinai). QT indicates QT interval; and RR, R–R interval.
Cedars-Sinai External Validation Cohort
ECGs (N=1 048 575) from a geographically distinct health system and alternate ECG acquisition equipment (MUSE-GE system) were collected to test model generalization between January 1, 1900, and December 31, 2023. Exactly 100 000 (from 65 291 unique patients) ECG recordings were selected from all consecutive ECGs with complete lead data and calculated measures. Patient characteristics of the Stanford Model Development and the Cedars-Sinai External Validation cohorts are provided in [Table 1](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#T1) (by patient) and in [Table S2](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#supplementary-materials) (by ECG recording).
Table 1. Patient Characteristics of the Development Cohort and External Testing Cohort
Variable
Train(n=206 563)
Validation (n=25 821)
Test(n=25 821)
External (n=65 291)
Age
58.3±18.0
58.4±17.9
58.4±18.0
63.6±18.2
Male
101 151 (49.0%)
12 649 (49.0%)
12 533 (48.5%)
34 794 (53.3%)
Race
AT/AF indicates atrial tachycardia or atrial fibrillation; CAD, coronary artery disease; HF, heart failure; and HTN, hypertension.
*Other includes patient-reported race entries that did not fit predefined EHR categories, including multiracial identities and free-text responses that could not be mapped to standard classifications.
Continuous QT Monitoring Cohort (ECGRDVQ PhysioNet Data Set)
A total of 1056 serial 12-lead ECGs from 22 healthy patients undergoing dofetilide loading, were each monitored for 24 hours after drug initiation.[8](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#core-collateral-R8) This was used specifically for the simulated continuous monitoring analysis. Device information for the ECGRDVQ data set was not specified.
Stanford Real-World Class III Antiarrhythmic Exposure Cohort
A separate retrospective clinical-encounter cohort of Stanford patients treated with dofetilide or sotalol, included 72 919 ECGs from 1676 unique patients across 2083 outpatient encounters (Stanford, 2008–2024). This cohort was used to assess the real-world clinical problem and validate 3DRECON-QT in the target population.
Stanford ICM Recording Cohort
The Medtronic CareLink database was accessed for all patients with an ICM implant (LINQ I and II, Medtronic, Minneapolis, MN) between 2016 and 2024. Only patients whose final transmission occurred during the period were accessible through the archives of older studies. Transmissions from a total of 387 patients were available. The timestamps from the presenting rhythm were compared with 12-lead ECGs acquired during routine clinical care. To ensure comparable physiological states between the 12-lead ECG and the ICM recordings, we included pairs obtained within ±7 days and with a heart rate difference |Δ heart rate |≤20 beats per minute. Twenty-six patients met these inclusion criteria.
ECG Recording and Preprocessing
All ECG recordings used in the Stanford Model Development Cohort and the Real-World Class III Antiarrhythmic Cohort were acquired using the Philips iECG system. Cedars-Sinai external testing recordings were obtained through the MUSE-GE system. We extracted 10-s, 12-lead ECG segments and standardized all signals to a 500-Hz sampling rate. A 0-phase, fifth-order Butterworth bandpass filter (0.5–40 Hz) was applied to remove baseline wander and high-frequency noise.
In all analyses, the input to 3DRECON-QT is a single-lead, narrowly spaced chest wall signal. For model development, external testing, continuous monitoring, and real-world class III analyses, the input vector was a derived ICM signal calculated voltage differential between precordial leads V3 and V2. True ICM recordings were the input for the Stanford ICM Recording analysis. Twelve-lead ECGs were used as the target for reconstruction and the basis for gold standard QT labels. A summary of model input and 12-lead ECG use is provided in [Table S1](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#supplementary-materials).
We standardized each recording by subtracting the record-wide mean and dividing by its SD, so that all leads entered the network with zero mean and unit variance. We excluded recordings containing >70% 0-valued samples or any lead with both 0 mean and 0 SD. In addition, recordings lacking cardiologist-interpreted QT or RR annotations were removed. The CONSORT diagram for the development and external test sets is provided in [Figure 2](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#F2).
QT Annotation
Ground truth QT and RR interval measurements for our development and external validation cohorts were extracted from the clinical ECG systems after adjudication by cardiologists during routine clinical care. QTc values were calculated from the measured QT intervals and RR intervals using the Bazett formula.[9](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#core-collateral-R9) Across both data sets, a prolonged QT was defined according to the corrected QT interval and the QRS duration according to the dofetilide Food and Drug Administration drug insert labeling.[10](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#core-collateral-R10) For a narrow QRS (≤100 ms), a prolonged QTc was defined as >500 ms. For a wide QRS (>100 ms), a QTc >550 ms was determined to be prolonged.[11](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#core-collateral-R11) Based on a mixed-model pooled variance calculation ([Detailed Methods](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#supplementary-materials)), the interobserver variance (caused by reader bias) is 47.1 ms2 (SD, 6.86 ms) whereas the within-observer variance (caused by physiological variability, repeat measures, noise, etc) is 2392.7 ms2. This results in a variance ratio of 0.02, or a proportion of variance of 0.02, suggesting minimal effect of interobserver labeling on QT assessment and clinically robust QT labels ([Figure S1](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#supplementary-materials)).
Development of 3DRECON-QT Model
The 3DRECON-QT model is a deep learning architecture designed to reconstruct a full 12-lead ECG and estimate the QT interval from a single-lead input. It employs an encoder and decoder framework with a multitask learning objective, encouraging the network to learn physiologically and spatially meaningful latent features that jointly inform both waveform reconstruction and QT prediction ([Figure 2](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#F2)).
A 10-s single-lead ECG signal with 500-Hz sampling is processed through a squeeze-and-excitation ResNeXt encoder, which extracts a compact latent representation of cardiac electrical activity.[12](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#core-collateral-R12),[13](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#core-collateral-R13) Two task-specific “heads” operate in parallel: a reconstruction head, which generates the 12-lead ECG by conditioning on the spatial coordinates (θ and ϕ) of each lead,[14](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#core-collateral-R14) and a QT prediction head, which applies a transformer decoder to the shared latent representation to estimate the QT interval.[15](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#core-collateral-R15) Model optimization used a weighted composite loss across the 2 tasks, stochastic gradient descent with cosine annealing, and early stopping based on validation performance.
Implementation was performed in Python (PyTorch v2.3.1).[16](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#core-collateral-R16) Detailed architectural, training, and hyperparameter details are provided in the [Supplemental Methods](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#supplementary-materials).
Continuous QT Monitoring Analysis (ECGRDVQ)
To assess the performance of 3DRECON-QT to detect changes in QT interval during continuous monitoring, we validated the model in the Continuous QT Monitoring Cohort (ECGRDVQ PhysioNet Dataset), which investigates the pharmacokinetic profile of dofetilide taken with oral ingestion during safety studies of the medication.[8](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#core-collateral-R8) We assessed 3DRECON-QT for detecting drug-induced QT prolongation using 24-hour ECGs from the ECGRDVQ dofetilide-loading data set (simulating ambulatory monitoring). The input signal is the same as described in the "ECG Recording and Preprocessing" section. We examined the temporal trends of aggregated mean predicted and ground truth QT/QTc. We performed classification analysis for prolonged QT interval, using the thresholds as described above for dofetilide loading and, secondarily, a ≥15% increase in QTc from baseline.[17](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#core-collateral-R17)
Real-World Class III Antiarrhythmic Exposure Analysis
To assess the clinical impact of an enhanced monitoring strategy in a high-risk patient population, we conducted a retrospective encounter-based analysis using the Stanford Real-World Class III Antiarrhythmic Exposure Cohort, a data set comprising 1676 unique patients receiving outpatient dofetilide or sotalol treatment at Stanford between 2008 and 2024 at the time of a hospital or emergency room encounter for any reason. We categorized the primary diagnosis from the encounter using the Clinical Classifications Software Refined for International Classification of Diseases, Tenth Revision, clinical modification codes to express which types of clinical encounters were associated with prolonged QT events on the first ECG of each encounter.[18](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#core-collateral-R18) We then established a case-control framework for serious arrhythmic events (torsades de pointes, ventricular fibrillation, and sudden cardiac death), calculating adjusted odds ratios (ORs) using the Fisher exact test. Last, we ensured that QT assessments in this high-risk population were appropriately classified using 3DRECON-QT performance on this clinical data set using classification analysis and receiver operator characteristics.
Stanford ICM Recording Analysis
We evaluated 3DRECON-QT using Medtronic LINQ ICM signals paired with a clinical 12-lead ECG obtained within ±7 days. Analyses were performed at the patient level (one paired QTc value per patient). For each patient, 3DRECON-QT generated predictions from the real ICM recordings, with the paired 12-lead ECG providing the reference QTc value. We repeated the QTc analysis using the derived ICM signal from the paired 12-lead ECG. Agreement with the reference QTc across patients was assessed using Pearson correlation coefficient and the mean absolute error (MAE) between the prediction and ground truth. We also compared waveform morphology between the real and simulated ICM signals and between the corresponding, model-predicted 12-lead reconstructions. R-peaks were detected using the Neurokit2 Python package, after which individual beats were segmented, temporally aligned to the R-peak, and normalized before overlay.[19](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#core-collateral-R19)
Statistical Analysis
Statistical analysis was performed in Python (v3.8) using SciPy and Scikit-learn.[20](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#core-collateral-R20) Descriptive statistics were reported as mean±SD or median [interquartile range]. For QT prediction performance, agreement between predicted and ground truth values was assessed using Bland-Altman analysis, Pearson correlation coefficient (r), and linear regression to calculate R², and MAE. Classification performance for prolonged QTc detection was evaluated using sensitivity, specificity, positive predictive value, negative predictive value, F1 score, area under the receiver operating characteristics curve (AUROC), and area under the precision-recall curve (AUPRC). CIs (95% CI) were generated using nonparametric bootstraps (1000 iterations, random sampling with replacement at the ECG level). Sensitivity analysis used patient-level clustered bootstrap with bias-corrected and accelerated intervals to account for multiple ECGs per patient. Association between QTc prolongation and clinical events was assessed using the Fisher exact test for univariate analysis and multivariable logistic regression to calculate adjusted ORs controlling for age, heart failure, atrial fibrillation, drug class, and laboratory values (serum creatinine, magnesium, and potassium). P values <0.05 were considered statistically significant.
Results
Participants
There were no significant differences between patient characteristics across the internal splits of the Stanford Model Development Cohort (aggregate demographics: mean age, 61.9 years; 52.2% male; race or ethnicity, White 57.6%, Asian 13.4%, Black 5.7%, Hispanic or Latino 12.7%). The Cedars-Sinai External Validation Cohort contained older individuals with a higher proportion of Black individuals compared with the Stanford Model Development Cohort ([Table 1](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#T1)).
3DRECON-QT Model Performance
Performance on the Development and External Validation Cohorts
Prolonged-QT classification using derived ICM recordings achieved an AUROC of 0.942 (95% CI, 0.941–0.943), AUPRC of 0.632 (95% CI, 0.626–0.639), and an accuracy of 0.957 (95% CI, 0.955–0.957) in the test partition of the Stanford Model Development Cohort. In the Cedars-Sinai External Validation Cohort, performance was similarly high, with an AUROC of 0.943 (95% CI, 0.941–0.945), AUPRC of 0.781 (95% CI, 0.774–0.789), and an accuracy of 0.925 (95% CI, 0.923–0.926) ([Figure 3A](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#F3)).
Figure 2. Overview of the 3DRECON-QT architecture for QT interval prediction from a single ICM recording. The model encodes spatial representation of the query lead (ThetaEncoder) and fuses it with the encoded ICM signal (SE-ResNext) and uses a transformer decoder for QT interval prediction (L2 loss) and an ECGDecoder for reconstruction (L1 loss). 1D indicates 1-dimensional; B, Bazett’s formula; Conv, convolution; ICM, insertable cardiac monitor; QTc, corrected QT interval; RR, R–R interval; SE-ResNeXt, Squeeze-and-Excitation ResNeXt network; ThetaEncoder, spherical angular encoder; and Z1 Conv/Z2 Conv, first and second convolutional blocks in the Z-stack feature extractor.
Figure 3. Model performance for QTc prediction and classification of prolonged QT. A, Receiver operating characteristics curves for classification of prolonged QTc from derived ICM recordings in the development (Stanford) and external testing (Cedars-Sinai) cohorts. B, Scatter plot comparing ground truth QT (clinical 12-lead ECG) and model-predicted QT (from derived ICM signal) in the development cohort. C, Bland-Altman analysis of QTc residuals (all samples from the test set) with bias (red dashed line) and limits of agreement (1.96 * SD, blue dashed lines). AUC indicates area under the curve; QTc, corrected QT interval; and ROC, receiver operating characteristics.
The median ground truth QTc was 411 ms (interquartile range, 391–441 ms). In the test split of the Stanford Model Development Cohort, the model-predicted median QTc was 418 ms (interquartile range, 401–441 ms). The Bland-Altman analysis for QTc revealed a bias of –5.46±29.38 ms ([Figure 3C](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#F3)).
To qualify 12-lead ECG reconstructions systematically, we analyzed performance across the Stanford Model Development test set, achieving mean Pearson correlation of 0.70 across all 12 leads ([Table S3](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#supplementary-materials)). Saliency analysis of the output signals shows activation across leads at clinically expected time points, including QRS onset and T wave offset along with accurate 12-lead reconstructions ([Figure S2](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#supplementary-materials)). To evaluate the contribution of the spatial encoding to QT prediction, we compared 3DRECON-QT with an ablated variant without the theta (spatial angle) encoder. The model with spatial encoding achieved QTc MAE of 25.9 ms and Pearson r of 0.73, whereas the ablated model without spatial encodings showed markedly degraded performance (QTc MAE, 39.23 ms; Pearson r, 0.04), despite similar training loss convergence ([Figure S3](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#supplementary-materials)).
A linear regression of the predicted QT (uncorrected) resulted in an R2 value of 0.747 (95% CI, 0.742–0.752) and a Pearson correlation coefficient of 0.864 (95% CI, 0.861–0.867; [Figure 3B](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#F3)). This represented a MAE of 17.52 ms (95% CI, 17.38–17.64) for the predicted QT interval in the test split of the Stanford Model Development Cohort.
In the Cedars-Sinai External Validation Cohort, the MAE for QT prediction was 21.14 ms (95% CI, 20.99–21.29). The prolonged QTc classification revealed an AUROC of 0.943 (95% CI, 0.941–0.945) and an AUPRC of 0.781 (95% CI, 0.774–0.789). See [Table 2](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#T2) for full test characteristics.
Table 2. Test Characteristics in the Holdout Internal Test Set and the External Test Set
Metric
Stanford
Stanford 95% CI
Cedars
Cedars 95% CI
Regression
MAE
17.521
17.381–17.641
21.135
20.985–21.288
R²
0.747
0.742–0.752
0.738
0.732–0.744
AUC indicates area under the curve; AUPRC, area under the precision-recall curve; AUROC, area under the receiver operating characteristics curve; MAE, mean absolute error; and NPV, negative predictive value.
A sensitivity analysis using patient-level clustered bootstraps using bias-corrected and accelerated intervals is presented in [Table S4](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#supplementary-materials) and showed minimal change from standard bootstrapping.
Performance in Continuous QT Monitoring Cohort
In patients from the Continuous QT Monitoring Cohort (ECGRDVQ PhysioNet Dataset), 3DRECON-QT demonstrated a statistically significant correlation between predicted QT intervals and ground truth values (Pearson r=0.851; P<0.001; [Figure 4](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#F4)). For serial QT measurements over the duration of recording, the MAE was 17.8 ms for QT interval. For detecting prolonged QTc using the absolute risk threshold (see Methods, “QT Annotation”), the model achieved an AUROC of 0.936 (95% CI, 0.900–0.967) and AUPRC of 0.395. For the secondary task of detecting a ≥15% increase in QTc from baseline, the model achieved an AUROC of 0.816 (95% CI, 0.790–0.843) and AUPRC of 0.538.
Figure 4. Continuous QTc tracking and performance evaluation. A, Continuous monitoring of QTc over time in patients receiving a single dose of dofetilide, including response to drug administration, with comparison of predicted and ground truth values (continuous 12-lead ECG). B, Receiver operating characteristics curve evaluating model performance for classification of QTc prolongation events. C, Precision-recall curve assessing detection of relative QTc increases from baseline. AUC indicates area under the curve; QTc, corrected QT interval; and ROC, receiver operating characteristics.
Validation of QTc and 12-Lead Morphology From Medtronic LINQ ICM Recordings
In the Stanford ICM Recording Cohort, 26 patients met criteria for time Δ (±7 days) between the ICM recording and the clinical reference 12-lead ECG. [Figure 5A](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#F5) shows beat-wise morphological comparison between an actual ICM recording and simulated ICM from the 12-lead ECG. 3DRECON-QT predicted QTc values from the ICM recording with a Pearson correlation of r=0.824 (95% CI, 0.67–0.92) and mean absolute error of 17.5 ms (95% CI, 12.4–23.2) with the gold standard QTc ([Figure 5B](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#F5)). A comparison for the same patient for the ICM and 12-lead reconstructions is shown in [Figure 5C](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#F5). Using derived ICM signals from the reference 12-lead ECG for the same cohort had a Pearson r=0.937 (95% CI, 0.885–0.970) and mean absolute error of 18.6 ms (95% CI, 14.1–23.6) with the gold standard QTc measurement. The difference in mean absolute error between the 2 methods was –1.1 ms (95% CI, –5.5 to 3.5).
Figure 5. Validation of QTc and 12-lead morphology from Medtronic LINQ ICM recordings. A, Beat-wise morphological comparison between real ICM recordings (red) and simulated ICM signals derived from the 12-lead ECG (blue) from a patient in ICM cohort. Individual beats are beat-detected and aligned to account for differences in heart rates. Average beats are styled with solid lines. B, Concordance between QTc values predicted from ICM recordings and gold-standard 12-lead QTc, with regression line (dashed red) and line of equality (solid black). Patients with wide QRS are labeled in red circles and those with narrow QRS in blue circles. C, Lead-wise comparison of 12 standard ECG leads reconstructed from the real ICM (red) and the derived ICM (blue) with average beats styled in solid lines. ICM indicates insertable cardiac monitor; and QTc, corrected QT interval.
Clinical Impact in a High-Risk Outpatient Cohort
Baseline characteristics for patients in the Stanford Real-World Class III Antiarrhythmic Cohort are provided in [Table S5](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#supplementary-materials). In this cohort, 1 in 6 patients (16.5%, 277 of 1676) had at least one documented episode of high-risk QTc prolongation after their initial inpatient loading on presentation to medical care for any indication. These events occurred during outpatient visits for a wide range of primary diagnoses, not just arrhythmia management follow-up. [Figure 6A](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#F6) shows that approximately one-third of admissions with prolonged QT were for noncardiac causes, whereas one-third of cases occurred during a primary visit for arrhythmia evaluation, and one-third for non-arrhythmic cardiac evaluation.
Figure 6. Clinical impact of prolonged QTc in a high-risk outpatient cohort. A, Distribution of primary visit diagnoses among outpatients with documented QTc prolongation while on class III antiarrhythmic therapy (dofetilide or sotalol), demonstrating that prolonged QTc events occurred across a wide range of cardiac and noncardiac clinical encounters. B, Receiver operating characteristic curve showing model performance of 3DRECON-QT for detection of prolonged QTc in this high-risk cohort. AUC indicates area under the curve; QTc, corrected QT interval; and ROC, receiver operating characteristics.
The overall incidence of ventricular arrhythmic event–related hospitalization (torsades de points, ventricular fibrillation, and sudden cardiac death) in the cohort was 1.37%. This was increased for patients with prolonged QTc compared with those without (3.97% versus 0.86%; univariate OR, 4.78; P<0.05). After adjusting for age, heart failure, atrial fibrillation, drug class (sotalol/dofetilide), and laboratory values including creatinine, potassium, and magnesium, the risk of event remained 4-fold (OR, 4.24; 95% CI, 1.81–9.90; P<0.05).
For patients with an active outpatient dofetilide or sotalol prescription, 3DRECON-QT accurately identified prolonged QTc based on the derived ICM signal with an AUROC of 0.94 and an F1 score of 0.60 ([Figure 6B](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#F6)).
Discussion
3DRECON-QT accurately predicts QT/QTc within clinical tolerances (mean error, 18–21 ms) with AUROC 0.94 for classifying prolonged QT from a single-lead, nonstandard ECG signal. The system was robust to patient populations across 2 health systems with varying demographics. Further, when applied to a continuous monitoring scenario during pharmacological therapy, the model predicted QTc changes subsequently validated by observed serial measurements. Moreover, the model revealed QT prolongation in patients evaluated for noncardiac conditions, which could easily be missed in routine practice.
In our real-world analysis, 1 in 6 patients receiving class III antiarrhythmic medications presented from the ambulatory setting with dangerous QTc prolongation for encounters related to both cardiac and noncardiac causes. These episodes confer a >4-fold increase in serious ventricular arrhythmia events despite previous successful in-hospital drug loading. Overall event rates of ventricular arrhythmia in patients with and without prolonged QTc were 0.86% and 3.97%, respectively, which is consistent with rates reported in randomized trials[21](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#core-collateral-R21),[22](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#core-collateral-R22) and systematic review.[4](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#core-collateral-R4) The causal relationship between QTc prolongation and arrhythmia events was not adjudicated in this study. Out-of-hospital arrests that did not reach our center would be underreported by this analysis. This analysis underscores the unmet clinical need of safe ambulatory monitoring of these medications.
Previous evidence that successful loading at a given dose of antiarrhythmic dose does not guarantee safety of that dose at future encounters.[23](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#core-collateral-R23) Current guidelines require a 12-lead ECG for QT assessment at least every 3 to 6 months for patients on class III antiarrhythmic agents.[5](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#core-collateral-R5),[24](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#core-collateral-R24) However, a recent statement from the Heart Rhythm Society highlights that although single-lead wearable devices can provide serial QT interval measurements, their accuracy and clinical utility remain inferior to standard 12-lead ECGs.[25](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#core-collateral-R25) Our data show how 3DRECON-QT may address this gap by providing near real-time QT safety monitoring from existing device infrastructure without additional leads, patient visits, or patient compliance burden.[26](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#core-collateral-R26) Cloud-based artificial intelligence pipelines are already in use (such as AccuRhythm AI, Medtronic, Minneapolis, MN) to filter out false-positive events in rhythm detection upstream of the provider portal, which provides a vantage point for immediate implementation. Last, this paradigm could possibly shorten or obviate mandatory hospital initiation of these medications for low-risk patients through hybrid inpatient-outpatient safety protocols and by providing objective triggers for dose reduction, electrolyte assessment, or drug discontinuation.[27](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#core-collateral-R27)
The multitask encoder in 3DRECON-QT learns latent representations that preserve ventricular repolarization vectors, explaining high-fidelity QT inference despite limited spatial input. Previous deep-learning approaches have estimated QT from lead I or smartwatch tracings but lacked external validation.[28](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#core-collateral-R28) Further, the more fixed morphology of the lead I recording is less generalizable to insertable or chest-worn recorders that have variable and nonstandard recording vectors. To achieve long-term, continuous monitoring, a more flexible model is required. In addition, the training data involve the largest QT assessment study to date with strict patient-level splits, which contributed to its robust performance in multi-institution external testing.
Limitations of this study include its retrospective design and the fact that QT labels are derived from clinician-reviewed ECGs during routine clinical care rather than those specifically ordered for QT assessment. Simultaneous ICM and 12-lead ECG recordings are not available because of the proprietary nature of the ICM recordings. Our method of training 3DRECON by deriving vectors approximating the ICM allows us to overcome this limitation. Physiological and positional factors such as circadian variation, posture, and rate dependency are known to influence QT measurements and may affect the relationship between the ICM and 12-lead ECG signals. The broad inclusion of patient states in the development and external testing sites supports the generalizability of the model. The high correlation of the QTc predictions from the true ICM recordings in our data set supports the use of the V3-V2 vector surrogate for training and the generalizability of the 3DRECON-QT network. Prospective simultaneous collection of 12-lead signals with digital ICM recordings would provide further clinical validation and the opportunity to improve model performance through fine tuning. Twelve-lead reconstructions are only validated for measurement of the QT interval. Last, the clinical impact for patients receiving other QT-prolonging agents such as antipsychotics, antiemetics, antiepileptics, and others is not assessed in this study and may have a different benefit threshold compared with class III antiarrhythmics to deploy this monitoring strategy.
Future work may involve deployment in patient populations with preexisting ICMs or chest-worn patches or consumer devices to detect safety concerns. In our high-risk cohort, precision exceeding 75% indicates a modest false-positive burden, with fewer than 2 notifications needed to identify one episode of prolonged QTc event. Notifications for prolongation events may be resolved by a 12-lead ECG and medication review. Conversely, the sensitivity of >80% from a single measurement may offer a substantial improvement in safety over current sporadic clinic-based monitoring. Although continuous monitoring introduces additional challenges such as motion artifacts, future deployment may incorporate noise-filtering strategies. Studying appropriate patient selection is critical to increase drug safety without creating undue health care use.
Conclusions
3DRECON-QT demonstrates that deep-learning reconstruction of latent spatial information from single-lead signals can meet guideline-level QT accuracy. This model may address a substantial hidden burden of high-risk prolongation after class III antiarrhythmic initiation and other QT-prolonging drugs.
Acknowledgments
A.J.R., R.A., and S.B. conceived of the study. S.B., J.W.H., R.A., and A.J.R. were involved in acquiring the Stanford data set in a format appropriate for this study and preprocessing. R.T. and D.O. performed data acquisition, preparation, and analysis at Cedars Sinai Medical Center. S.B., R.A., A.J.R., S.M.N., M.P., A.P., P.G., X.L., K.B., and T.C. performed data analysis and interpretation. Clinical expertise was provided by P.J.W., E.A., A.P., M.P., and S.M.N. Article editing was provided by S.M.N., T.C., and M.P. All authors reviewed the article and approved the final version.
Footnote
Nonstandard Abbreviations and Acronyms
AT/AF
atrial tachycardia or atrial fibrillation
AUPRC
area under the precision-recall curve
AUROC
area under the receiver operating characteristics curve
ICM
insertable cardiac monitor
MAE
mean absolute error
Supplemental Material
File (10.1161.circulationaha.125.077494_stard_checklist.pdf)
[Download](https://www.ahajournals.org/doi/suppl/10.1161/CIRCULATIONAHA.125.077494/suppl_file/10.1161.circulationaha.125.077494_stard_checklist.pdf)
159.45 KB
File (10.1161.circulationaha.125.077494_strobe_checklist.pdf)
[Download](https://www.ahajournals.org/doi/suppl/10.1161/CIRCULATIONAHA.125.077494/suppl_file/10.1161.circulationaha.125.077494_strobe_checklist.pdf)
153.08 KB
File (10.1161.circulationaha.125.077494_supplement.pdf)
[Download](https://www.ahajournals.org/doi/suppl/10.1161/CIRCULATIONAHA.125.077494/suppl_file/10.1161.circulationaha.125.077494_supplement.pdf)
566.61 KB
File (2025_1101_ai_enabled_continuous_qt_monitoring_from_icms_supplement.pdf)
Checklists
Detailed Methods
Tables S1–S5
Figures S1–S3
[Download](https://www.ahajournals.org/doi/suppl/10.1161/CIRCULATIONAHA.125.077494/suppl_file/2025_1101_ai_enabled_continuous_qt_monitoring_from_icms_supplement.pdf)
566.86 KB
File (circ-2025-077494-s01.pdf)
[Download](https://www.ahajournals.org/doi/suppl/10.1161/CIRCULATIONAHA.125.077494/suppl_file/circ-2025-077494-s01.pdf)
159.45 KB
File (circ-2025-077494-s02.pdf)
[Download](https://www.ahajournals.org/doi/suppl/10.1161/CIRCULATIONAHA.125.077494/suppl_file/circ-2025-077494-s02.pdf)
153.08 KB
References
1.
Drew BJ, Ackerman MJ, Funk M, Gibler WB, Kligfield P, Menon V, Philippides GJ, Roden DM, Zareba W; American Heart Association Acute Cardiac Care Committee of the Council on Clinical Cardiology, the Council on Cardiovascular Nursing, and the American College of Cardiology Foundation. Prevention of torsade de pointes in hospital settings: a scientific statement from the American Heart Association and the American College of Cardiology Foundation. Circulation. 2010;121:1047–1060. doi: 10.1161/CIRCULATIONAHA.109.192704
[Go to Citation](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#core-R1-1)
[Crossref](https://www.ahajournals.org/servlet/linkout?suffix=e_1_3_3_2_2&dbid=4&doi=10.1161%2FCIRCULATIONAHA.125.077494&key=10.1161%2FCIRCULATIONAHA.109.192704&site=aha-site)
[PubMed](https://pubmed.ncbi.nlm.nih.gov/20142454/)
[Google Scholar](https://scholar.google.com/scholar_lookup?title=Prevention+of+torsade+de+pointes+in+hospital+settings%3A+a+scientific+statement+from+the+American+Heart+Association+and+the+American+College+of+Cardiology+Foundation.&author=BJ+Drew&author=MJ+Ackerman&author=M+Funk&author=WB+Gibler&author=P+Kligfield&author=V+Menon&author=GJ+Philippides&author=DM+Roden&author=W+Zareba&publication_year=2010&journal=Circulation&pages=1047-1060&doi=10.1161%2FCIRCULATIONAHA.109.192704&pmid=20142454)
2.
Sandau KE, Funk M, Auerbach A, Barsness GW, Blum K, Cvach M, Lampert R, May JL, McDaniel GM, Perez MV, et al; American Heart Association Council on Cardiovascular and Stroke Nursing; Council on Clinical Cardiology; and Council on Cardiovascular Disease in the Young. Update to practice standards for electrocardiographic monitoring in hospital settings: a scientific statement from the American Heart Association. Circulation. 2017;136:e273–e344. doi: 10.1161/CIR.0000000000000527
[Go to Citation](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#core-R2-1)
[Crossref](https://www.ahajournals.org/servlet/linkout?suffix=e_1_3_3_3_2&dbid=4&doi=10.1161%2FCIRCULATIONAHA.125.077494&key=10.1161%2FCIR.0000000000000527&site=aha-site)
[PubMed](https://pubmed.ncbi.nlm.nih.gov/28974521/)
[Google Scholar](https://scholar.google.com/scholar_lookup?title=Update+to+practice+standards+for+electrocardiographic+monitoring+in+hospital+settings%3A+a+scientific+statement+from+the+American+Heart+Association.&author=KE+Sandau&author=M+Funk&author=A+Auerbach&author=GW+Barsness&author=K+Blum&author=M+Cvach&author=R+Lampert&author=JL+May&author=GM+McDaniel&author=MV+Perez&publication_year=2017&journal=Circulation&pages=e273-e344&doi=10.1161%2FCIR.0000000000000527&pmid=28974521)
3.
Steinberg JS, Varma N, Cygankiewicz I, Aziz P, Balsam P, Baranchuk A, Cantillon DJ, Dilaveris P, Dubner SJ, El-Sherif N, et al. 2017 ISHNE-HRS expert consensus statement on ambulatory ECG and external cardiac monitoring/telemetry. Heart Rhythm. 2017;14:e55–e96. doi: 10.1016/j.hrthm.2017.03.038
[Go to Citation](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#core-R3-1)
[Crossref](https://doi.org/10.1016/j.hrthm.2017.03.038)
[PubMed](https://pubmed.ncbi.nlm.nih.gov/28495301/)
[Google Scholar](https://scholar.google.com/scholar_lookup?title=2017+ISHNE-HRS+expert+consensus+statement+on+ambulatory+ECG+and+external+cardiac+monitoring%2Ftelemetry.&author=JS+Steinberg&author=N+Varma&author=I+Cygankiewicz&author=P+Aziz&author=P+Balsam&author=A+Baranchuk&author=DJ+Cantillon&author=P+Dilaveris&author=SJ+Dubner&author=N+El-Sherif&publication_year=2017&journal=Heart+Rhythm&pages=e55-e96&doi=10.1016%2Fj.hrthm.2017.03.038&pmid=28495301)
4.
Tisdale JE, Chung MK, Campbell KB, Hammadah M, Joglar JA, Leclerc J, Rajagopalan B; American Heart Association Clinical Pharmacology Committee of the Council on Clinical Cardiology and Council on Cardiovascular and Stroke Nursing. Drug-induced arrhythmias: a scientific statement from the American Heart Association. Circulation. 2020;142:e214–e233. doi: 10.1161/CIR.0000000000000905
[Crossref](https://www.ahajournals.org/servlet/linkout?suffix=e_1_3_3_5_2&dbid=4&doi=10.1161%2FCIRCULATIONAHA.125.077494&key=10.1161%2FCIR.0000000000000905&site=aha-site)
[PubMed](https://pubmed.ncbi.nlm.nih.gov/32929996/)
[Google Scholar](https://scholar.google.com/scholar_lookup?title=Drug-induced+arrhythmias%3A+a+scientific+statement+from+the+American+Heart+Association.&author=JE+Tisdale&author=MK+Chung&author=KB+Campbell&author=M+Hammadah&author=JA+Joglar&author=J+Leclerc&author=B+Rajagopalan&publication_year=2020&journal=Circulation&pages=e214-e233&doi=10.1161%2FCIR.0000000000000905&pmid=32929996)
eLetters
eLetters should relate to an article recently published in the journal and are not a forum for providing unpublished data. Comments are reviewed for appropriate use of tone and language. Comments are not peer-reviewed. Acceptable comments are posted to the journal website only. Comments are not published in an issue and are not indexed in PubMed. Comments should be no longer than 500 words and will only be posted online. References are limited to 10. Authors of the article cited in the comment will be invited to reply, as appropriate.
Comments and feedback on AHA/ASA Scientific Statements and Guidelines should be directed to the AHA/ASA Manuscript Oversight Committee via its [Correspondence](https://professional.heart.org/en/guidelines-and-statements/correspondence) page.
Information & Authors
InformationAuthors
Information
Published In
Circulation
[Volume 153 • Number 1 • 6 January 2026](https://www.ahajournals.org/toc/circ/153/1)
Pages: 35 - 46
PubMed: [41460938](https://pubmed.ncbi.nlm.nih.gov/41460938/)
Copyright
© 2025 American Heart Association, Inc.
Versions
You are viewing the most recent version of this article.
[8 November 2025: Ahead of Print](https://www.ahajournals.org/history/fe12dccd-d60e-4810-9d19-1e8123c4b010/circulationaha.125.077494.9024660.pdf)
History
Received: 15 September 2025
Accepted: 28 October 2025
Published online: 8 November 2025
Published in print: 6 January 2026
Permissions
Request permissions for this article.
Keywords
[anti-arrhythmia agents](https://www.ahajournals.org/action/doSearch?AllField=anti-arrhythmia+agents&stemming=yes&publication=circ)
[arrhythmias, cardiac](https://www.ahajournals.org/action/doSearch?AllField=arrhythmias%2C+cardiac&stemming=yes&publication=circ)
[deep learning](https://www.ahajournals.org/action/doSearch?AllField=deep+learning&stemming=yes&publication=circ)
[drug-related side effects and adverse reactions](https://www.ahajournals.org/action/doSearch?AllField=drug-related+side+effects+and+adverse+reactions&stemming=yes&publication=circ)
[electrocardiography](https://www.ahajournals.org/action/doSearch?AllField=electrocardiography&stemming=yes&publication=circ)
[monitoring, ambulatory](https://www.ahajournals.org/action/doSearch?AllField=monitoring%2C+ambulatory&stemming=yes&publication=circ)
[Torsades de Pointes](https://www.ahajournals.org/action/doSearch?AllField=Torsades+de+Pointes&stemming=yes&publication=circ)
Subjects
[Arrhythmias](https://www.ahajournals.org/topic/aha-collection/10001)
[Electrocardiology (ECG)](https://www.ahajournals.org/topic/aha-collection/10125)
[Sudden Cardiac Death](https://www.ahajournals.org/topic/aha-collection/10005)
Metrics & Citations
MetricsCitations6
Metrics
[Downloads](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#)
[Citations](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#)
No data available.
0100200Nov 2025Dec 2025Jan 2026Feb 2026Mar 2026Apr 2026May 2026Jun 2026Jul 2026Aug 2026
1,212
6
[Total](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#)
[First 90 Days](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#)
[6 Months](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#)
Total number of downloads and citations
[See more details](https://www.altmetric.com/details.php?domain=www.ahajournals.org&citation_id=184318552)
[Picked up by 2 news outlets](https://www.altmetric.com/details.php?domain=www.ahajournals.org&citation_id=184318552&tab=news)
[On 1 Facebook pages](https://www.altmetric.com/details.php?domain=www.ahajournals.org&citation_id=184318552&tab=facebook)
8 readers on Mendeley
View Options
View options
PDF with Supplements
PDF/EPUB
Figures
Figure 1. Consort diagram showing data flow for 12-lead ECGs in the development (Stanford) and external testing cohorts (Cedars-Sinai). QT indicates QT interval; and RR, R–R interval.
[Go to Figure](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#F1)Open in Viewer
Figure 2. Overview of the 3DRECON-QT architecture for QT interval prediction from a single ICM recording. The model encodes spatial representation of the query lead (ThetaEncoder) and fuses it with the encoded ICM signal (SE-ResNext) and uses a transformer decoder for QT interval prediction (L2 loss) and an ECGDecoder for reconstruction (L1 loss). 1D indicates 1-dimensional; B, Bazett’s formula; Conv, convolution; ICM, insertable cardiac monitor; QTc, corrected QT interval; RR, R–R interval; SE-ResNeXt, Squeeze-and-Excitation ResNeXt network; ThetaEncoder, spherical angular encoder; and Z1 Conv/Z2 Conv, first and second convolutional blocks in the Z-stack feature extractor.
[Go to Figure](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#F2)Open in Viewer
Figure 3. Model performance for QTc prediction and classification of prolonged QT. A, Receiver operating characteristics curves for classification of prolonged QTc from derived ICM recordings in the development (Stanford) and external testing (Cedars-Sinai) cohorts. B, Scatter plot comparing ground truth QT (clinical 12-lead ECG) and model-predicted QT (from derived ICM signal) in the development cohort. C, Bland-Altman analysis of QTc residuals (all samples from the test set) with bias (red dashed line) and limits of agreement (1.96 * SD, blue dashed lines). AUC indicates area under the curve; QTc, corrected QT interval; and ROC, receiver operating characteristics.
[Go to Figure](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#F3)Open in Viewer
Figure 4. Continuous QTc tracking and performance evaluation. A, Continuous monitoring of QTc over time in patients receiving a single dose of dofetilide, including response to drug administration, with comparison of predicted and ground truth values (continuous 12-lead ECG). B, Receiver operating characteristics curve evaluating model performance for classification of QTc prolongation events. C, Precision-recall curve assessing detection of relative QTc increases from baseline. AUC indicates area under the curve; QTc, corrected QT interval; and ROC, receiver operating characteristics.
[Go to Figure](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#F4)Open in Viewer
Figure 5. Validation of QTc and 12-lead morphology from Medtronic LINQ ICM recordings. A, Beat-wise morphological comparison between real ICM recordings (red) and simulated ICM signals derived from the 12-lead ECG (blue) from a patient in ICM cohort. Individual beats are beat-detected and aligned to account for differences in heart rates. Average beats are styled with solid lines. B, Concordance between QTc values predicted from ICM recordings and gold-standard 12-lead QTc, with regression line (dashed red) and line of equality (solid black). Patients with wide QRS are labeled in red circles and those with narrow QRS in blue circles. C, Lead-wise comparison of 12 standard ECG leads reconstructed from the real ICM (red) and the derived ICM (blue) with average beats styled in solid lines. ICM indicates insertable cardiac monitor; and QTc, corrected QT interval.
[Go to Figure](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#F5)Open in Viewer
Figure 6. Clinical impact of prolonged QTc in a high-risk outpatient cohort. A, Distribution of primary visit diagnoses among outpatients with documented QTc prolongation while on class III antiarrhythmic therapy (dofetilide or sotalol), demonstrating that prolonged QTc events occurred across a wide range of cardiac and noncardiac clinical encounters. B, Receiver operating characteristic curve showing model performance of 3DRECON-QT for detection of prolonged QTc in this high-risk cohort. AUC indicates area under the curve; QTc, corrected QT interval; and ROC, receiver operating characteristics.
[Go to Figure](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#F6)Open in Viewer
Tables
Table 1. Patient Characteristics of the Development Cohort and External Testing Cohort
[Go to Table](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#T1)Open in Viewer
Table 2. Test Characteristics in the Holdout Internal Test Set and the External Test Set
[Go to Table](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#T2)Open in Viewer
Media
Share
Share
Share article link
https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494
Share
References
References
1.
Drew BJ, Ackerman MJ, Funk M, Gibler WB, Kligfield P, Menon V, Philippides GJ, Roden DM, Zareba W; American Heart Association Acute Cardiac Care Committee of the Council on Clinical Cardiology, the Council on Cardiovascular Nursing, and the American College of Cardiology Foundation. Prevention of torsade de pointes in hospital settings: a scientific statement from the American Heart Association and the American College of Cardiology Foundation. Circulation. 2010;121:1047–1060. doi: 10.1161/CIRCULATIONAHA.109.192704
[Go to Citation](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#core-R1-1)
[Crossref](https://www.ahajournals.org/servlet/linkout?suffix=e_1_3_3_2_2&dbid=4&doi=10.1161%2FCIRCULATIONAHA.125.077494&key=10.1161%2FCIRCULATIONAHA.109.192704&site=aha-site)
[PubMed](https://pubmed.ncbi.nlm.nih.gov/20142454/)
[Google Scholar](https://scholar.google.com/scholar_lookup?title=Prevention+of+torsade+de+pointes+in+hospital+settings%3A+a+scientific+statement+from+the+American+Heart+Association+and+the+American+College+of+Cardiology+Foundation.&author=BJ+Drew&author=MJ+Ackerman&author=M+Funk&author=WB+Gibler&author=P+Kligfield&author=V+Menon&author=GJ+Philippides&author=DM+Roden&author=W+Zareba&publication_year=2010&journal=Circulation&pages=1047-1060&doi=10.1161%2FCIRCULATIONAHA.109.192704&pmid=20142454)
2.
Sandau KE, Funk M, Auerbach A, Barsness GW, Blum K, Cvach M, Lampert R, May JL, McDaniel GM, Perez MV, et al; American Heart Association Council on Cardiovascular and Stroke Nursing; Council on Clinical Cardiology; and Council on Cardiovascular Disease in the Young. Update to practice standards for electrocardiographic monitoring in hospital settings: a scientific statement from the American Heart Association. Circulation. 2017;136:e273–e344. doi: 10.1161/CIR.0000000000000527
[Go to Citation](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#core-R2-1)
[Crossref](https://www.ahajournals.org/servlet/linkout?suffix=e_1_3_3_3_2&dbid=4&doi=10.1161%2FCIRCULATIONAHA.125.077494&key=10.1161%2FCIR.0000000000000527&site=aha-site)
[PubMed](https://pubmed.ncbi.nlm.nih.gov/28974521/)
[Google Scholar](https://scholar.google.com/scholar_lookup?title=Update+to+practice+standards+for+electrocardiographic+monitoring+in+hospital+settings%3A+a+scientific+statement+from+the+American+Heart+Association.&author=KE+Sandau&author=M+Funk&author=A+Auerbach&author=GW+Barsness&author=K+Blum&author=M+Cvach&author=R+Lampert&author=JL+May&author=GM+McDaniel&author=MV+Perez&publication_year=2017&journal=Circulation&pages=e273-e344&doi=10.1161%2FCIR.0000000000000527&pmid=28974521)
3.
Steinberg JS, Varma N, Cygankiewicz I, Aziz P, Balsam P, Baranchuk A, Cantillon DJ, Dilaveris P, Dubner SJ, El-Sherif N, et al. 2017 ISHNE-HRS expert consensus statement on ambulatory ECG and external cardiac monitoring/telemetry. Heart Rhythm. 2017;14:e55–e96. doi: 10.1016/j.hrthm.2017.03.038
[Go to Citation](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#core-R3-1)
[Crossref](https://doi.org/10.1016/j.hrthm.2017.03.038)
[PubMed](https://pubmed.ncbi.nlm.nih.gov/28495301/)
[Google Scholar](https://scholar.google.com/scholar_lookup?title=2017+ISHNE-HRS+expert+consensus+statement+on+ambulatory+ECG+and+external+cardiac+monitoring%2Ftelemetry.&author=JS+Steinberg&author=N+Varma&author=I+Cygankiewicz&author=P+Aziz&author=P+Balsam&author=A+Baranchuk&author=DJ+Cantillon&author=P+Dilaveris&author=SJ+Dubner&author=N+El-Sherif&publication_year=2017&journal=Heart+Rhythm&pages=e55-e96&doi=10.1016%2Fj.hrthm.2017.03.038&pmid=28495301)
4.
Tisdale JE, Chung MK, Campbell KB, Hammadah M, Joglar JA, Leclerc J, Rajagopalan B; American Heart Association Clinical Pharmacology Committee of the Council on Clinical Cardiology and Council on Cardiovascular and Stroke Nursing. Drug-induced arrhythmias: a scientific statement from the American Heart Association. Circulation. 2020;142:e214–e233. doi: 10.1161/CIR.0000000000000905
[Crossref](https://www.ahajournals.org/servlet/linkout?suffix=e_1_3_3_5_2&dbid=4&doi=10.1161%2FCIRCULATIONAHA.125.077494&key=10.1161%2FCIR.0000000000000905&site=aha-site)
[PubMed](https://pubmed.ncbi.nlm.nih.gov/32929996/)
[Google Scholar](https://scholar.google.com/scholar_lookup?title=Drug-induced+arrhythmias%3A+a+scientific+statement+from+the+American+Heart+Association.&author=JE+Tisdale&author=MK+Chung&author=KB+Campbell&author=M+Hammadah&author=JA+Joglar&author=J+Leclerc&author=B+Rajagopalan&publication_year=2020&journal=Circulation&pages=e214-e233&doi=10.1161%2FCIR.0000000000000905&pmid=32929996)
5.
Joglar JA, Chung MK, Armbruster AL, Benjamin EJ, Chyou JY, Cronin EM, Deswal A, Eckhardt LL, Goldberger ZD, Gopinathannair R, et al; Peer Review Committee Members. 2023 ACC/AHA/ACCP/HRS guideline for the diagnosis and management of atrial fibrillation: a report of the American College of Cardiology/American Heart Association Joint Committee on Clinical Practice Guidelines. Circulation. 2024;149:e1–e156. doi: 10.1161/CIR.0000000000001193
[Crossref](https://www.ahajournals.org/servlet/linkout?suffix=e_1_3_3_6_2&dbid=4&doi=10.1161%2FCIRCULATIONAHA.125.077494&key=10.1161%2FCIR.0000000000001193&site=aha-site)
[PubMed](https://pubmed.ncbi.nlm.nih.gov/38033089/)
[Google Scholar](https://scholar.google.com/scholar_lookup?title=2023+ACC%2FAHA%2FACCP%2FHRS+guideline+for+the+diagnosis+and+management+of+atrial+fibrillation%3A+a+report+of+the+American+College+of+Cardiology%2FAmerican+Heart+Association+Joint+Committee+on+Clinical+Practice+Guidelines.&author=JA+Joglar&author=MK+Chung&author=AL+Armbruster&author=EJ+Benjamin&author=JY+Chyou&author=EM+Cronin&author=A+Deswal&author=LL+Eckhardt&author=ZD+Goldberger&author=R+Gopinathannair&publication_year=2024&journal=Circulation&pages=e1-e156&doi=10.1161%2FCIR.0000000000001193&pmid=38033089)
6.
Rogers AJ, Selvalingam A, Alhusseini MI, Krummen DE, Corrado C, Abuzaid F, Baykaner T, Meyer C, Clopton P, Giles W, et al. Machine learned cellular phenotypes in cardiomyopathy predict sudden death. Circ Res. 2021;128:172–184. doi: 10.1161/CIRCRESAHA.120.317345
[Go to Citation](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#core-R6-1)
[Crossref](https://www.ahajournals.org/servlet/linkout?suffix=e_1_3_3_7_2&dbid=4&doi=10.1161%2FCIRCULATIONAHA.125.077494&key=10.1161%2FCIRCRESAHA.120.317345&site=aha-site)
[PubMed](https://pubmed.ncbi.nlm.nih.gov/33167779/)
[Google Scholar](https://scholar.google.com/scholar_lookup?title=Machine+learned+cellular+phenotypes+in+cardiomyopathy+predict+sudden+death.&author=AJ+Rogers&author=A+Selvalingam&author=MI+Alhusseini&author=DE+Krummen&author=C+Corrado&author=F+Abuzaid&author=T+Baykaner&author=C+Meyer&author=P+Clopton&author=W+Giles&publication_year=2021&journal=Circ+Res&pages=172-184&doi=10.1161%2FCIRCRESAHA.120.317345&pmid=33167779)
7.
Rogers AJ, Bhatia NK, Bandyopadhyay S, Tooley J, Ansari R, Thakkar V, Xu J, Soto JT, Tung JS, Alhusseini MI, et al. Identification of cardiac wall motion abnormalities in diverse populations by deep learning of the electrocardiogram. NPJ Digit Med. 2025;8:21. doi: 10.1038/s41746-024-01407-y
[Go to Citation](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#core-R7-1)
[Crossref](https://doi.org/10.1038/s41746-024-01407-y)
[PubMed](https://pubmed.ncbi.nlm.nih.gov/39799179/)
[Google Scholar](https://scholar.google.com/scholar_lookup?title=Identification+of+cardiac+wall+motion+abnormalities+in+diverse+populations+by+deep+learning+of+the+electrocardiogram.&author=AJ+Rogers&author=NK+Bhatia&author=S+Bandyopadhyay&author=J+Tooley&author=R+Ansari&author=V+Thakkar&author=J+Xu&author=JT+Soto&author=JS+Tung&author=MI+Alhusseini&publication_year=2025&journal=NPJ+Digit+Med&pages=21&doi=10.1038%2Fs41746-024-01407-y&pmid=39799179)
8.
Johannesen L, Vicente J, Mason JW, Sanabria C, Waite-Labott K, Hong M, Guo P, Lin J, Sørensen JS, Galeotti L, et al. Differentiating drug-induced multichannel block on the electrocardiogram: randomized study of dofetilide, quinidine, ranolazine, and verapamil. Clin Pharmacol Ther. 2014;96:549–558. doi: 10.1038/clpt.2014.155
[Crossref](https://doi.org/10.1038/clpt.2014.155)
[PubMed](https://pubmed.ncbi.nlm.nih.gov/25054430/)
[Google Scholar](https://scholar.google.com/scholar_lookup?title=Differentiating+drug-induced+multichannel+block+on+the+electrocardiogram%3A+randomized+study+of+dofetilide%2C+quinidine%2C+ranolazine%2C+and+verapamil.&author=L+Johannesen&author=J+Vicente&author=JW+Mason&author=C+Sanabria&author=K+Waite-Labott&author=M+Hong&author=P+Guo&author=J+Lin&author=JS+S%C3%B8rensen&author=L+Galeotti&publication_year=2014&journal=Clin+Pharmacol+Ther&pages=549-558&doi=10.1038%2Fclpt.2014.155&pmid=25054430)
9.
Bazett HC. An analysis of the time‐relations of electrocardiograms. Ann Noninvasive Electrocardiol. 1997;2:177–194. doi: 10.1111/j.1542-474x.1997.tb00325.x
[Go to Citation](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#core-R9-1)
[Crossref](https://doi.org/10.1111/j.1542-474x.1997.tb00325.x)
[Google Scholar](https://scholar.google.com/scholar_lookup?title=An+analysis+of+the+time%E2%80%90relations+of+electrocardiograms.&author=HC+Bazett&publication_year=1997&journal=Ann+Noninvasive+Electrocardiol&pages=177-194&doi=10.1111%2Fj.1542-474x.1997.tb00325.x)
10.
Tikosyn (dofetilide) [package insert]. Pfizer Inc.; 2014. Accessed June 5, 2025. [https://www.accessdata.fda.gov/drugsatfda_docs/label/2013/020931s007lbl.pdf](https://www.accessdata.fda.gov/drugsatfda_docs/label/2013/020931s007lbl.pdf)
[Go to Citation](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#core-R10-1)
[Google Scholar](https://scholar.google.com/scholar_lookup?title=Tikosyn+%28dofetilide%29+%5Bpackage+insert%5D.&publication_year=2014)
11.
Pedersen OD, Bagger H, Keller N, Marchant B, Køber L, Torp-Pedersen C. Efficacy of dofetilide in the treatment of atrial fibrillation-flutter in patients with reduced left ventricular function: a Danish investigation of arrhythmia and mortality on dofetilide (diamond) substudy. Circulation. 2001;104:292–296. doi: 10.1161/01.cir.104.3.292
[Go to Citation](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#core-R11-1)
[Crossref](https://www.ahajournals.org/servlet/linkout?suffix=e_1_3_3_12_2&dbid=4&doi=10.1161%2FCIRCULATIONAHA.125.077494&key=10.1161%2F01.CIR.104.3.292&site=aha-site)
[PubMed](https://pubmed.ncbi.nlm.nih.gov/11457747/)
[Google Scholar](https://scholar.google.com/scholar_lookup?title=Efficacy+of+dofetilide+in+the+treatment+of+atrial+fibrillation-flutter+in+patients+with+reduced+left+ventricular+function%3A+a+Danish+investigation+of+arrhythmia+and+mortality+on+dofetilide+%28diamond%29+substudy.&author=OD+Pedersen&author=H+Bagger&author=N+Keller&author=B+Marchant&author=L+K%C3%B8ber&author=C+Torp-Pedersen&publication_year=2001&journal=Circulation&pages=292-296&doi=10.1161%2F01.CIR.104.3.292&pmid=11457747)
12.
Hu J, Shen L, Sun G. Squeeze-and-excitation networks. 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition. IEEE; 2018:7132–7140.
[Go to Citation](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#core-R12-1)
[Google Scholar](https://scholar.google.com/scholar_lookup?title=Squeeze-and-excitation+networks.&author=J+Hu&author=L+Shen&author=G+Sun&publication_year=2018&pages=7132-7140)
13.
Xie S, Girshick R, Dollar P, Tu Z, He K. Aggregated residual transformations for deep neural networks. 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). IEEE; 2017:1492–1500.
[Go to Citation](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#core-R13-1)
[Google Scholar](https://scholar.google.com/scholar_lookup?title=Aggregated+residual+transformations+for+deep+neural+networks.&author=S+Xie&author=R+Girshick&author=P+Dollar&author=Z+Tu&author=K+He&publication_year=2017&pages=1492-1500)
14.
Chen J, Zheng X, Yu H, Chen DZ, Wu J. Electrocardio panorama: synthesizing new ECG views with self-supervision. Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence.- IJCAI Organization; 2021:3597–3605.
[Go to Citation](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#core-R14-1)
[Google Scholar](https://scholar.google.com/scholar_lookup?title=Electrocardio+panorama%3A+synthesizing+new+ECG+views+with+self-supervision.&author=J+Chen&author=X+Zheng&author=H+Yu&author=DZ+Chen&author=J+Wu&publication_year=2021&pages=3597-3605)
15.
Vaswani A, Shazeer N, Parmar N, Uszkoreit J, Jones L, Gomez AN, Kaiser L, Polosukhin I. Attention is all you need [published online June 12, 2017]. arXiv [csCL]. 2017; doi: 10.48550/arXi.1706.03762
[Go to Citation](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#core-R15-1)
[Crossref](https://doi.org/10.48550/arXi.1706.03762)
[Google Scholar](https://scholar.google.com/scholar_lookup?title=Attention+is+all+you+need+%5Bpublished+online+June+12%2C+2017%5D.&author=A+Vaswani&author=N+Shazeer&author=N+Parmar&author=J+Uszkoreit&author=L+Jones&author=AN+Gomez&author=L+Kaiser&author=I+Polosukhin&publication_year=2017&journal=arXiv+%5BcsCL%5D&doi=10.48550%2FarXi.1706.03762)
16.
Paszke A, Gross S, Massa F, Lerer A, Bradbury J, Chanan G, Killeen T, Lin Z, Gimelshein N, Antiga L, et al. PyTorch: an imperative style, high-performance deep learning library [published online December 3, 2019]. arXiv [csLG]. 2019; doi: 10.48550/arXi.1912.01703
[Go to Citation](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#core-R16-1)
[Crossref](https://doi.org/10.48550/arXi.1912.01703)
[Google Scholar](https://scholar.google.com/scholar_lookup?title=PyTorch%3A+an+imperative+style%2C+high-performance+deep+learning+library+%5Bpublished+online+December+3%2C+2019%5D.&author=A+Paszke&author=S+Gross&author=F+Massa&author=A+Lerer&author=J+Bradbury&author=G+Chanan&author=T+Killeen&author=Z+Lin&author=N+Gimelshein&author=L+Antiga&publication_year=2019&journal=arXiv+%5BcsLG%5D&doi=10.48550%2FarXi.1912.01703)
17.
Alam R, Aguirre AD, Stultz CM. QTNet: deep learning for estimating QT intervals using a single lead ECG. Annu Int Conf IEEE Eng Med Biol Soc. 2023;2023:1–4. doi: 10.1109/EMBC40787.2023.10341204
[Go to Citation](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#core-R17-1)
[Crossref](https://doi.org/10.1109/EMBC40787.2023.10341204)
[PubMed](https://pubmed.ncbi.nlm.nih.gov/38261472/)
[Google Scholar](https://scholar.google.com/scholar_lookup?title=QTNet%3A+deep+learning+for+estimating+QT+intervals+using+a+single+lead+ECG.&author=R+Alam&author=AD+Aguirre&author=CM+Stultz&publication_year=2023&journal=Annu+Int+Conf+IEEE+Eng+Med+Biol+Soc&pages=1-4&doi=10.1109%2FEMBC40787.2023.10341204&pmid=38261472)
18.
Agency for Healthcare Research and Quality, Healthcare Cost and Utilization Project (HCUP). Clinical Classifications Software Refined (CCSR) for ICD-10-CM Diagnoses. Accessed June 5, 2025. [https://www.hcup-us.ahrq.gov/toolssoftware/ccsr/ccs_refined.jsp](https://www.hcup-us.ahrq.gov/toolssoftware/ccsr/ccs_refined.jsp)
[Go to Citation](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#core-R18-1)
[Google Scholar](https://scholar.google.com/scholar_lookup?title=Clinical+Classifications+Software+Refined+%28CCSR%29+for+ICD-10-CM+Diagnoses)
19.
Makowski D, Pham T, Lau ZJ, Brammer JC, Lespinasse F, Pham H, Schölzel C, Chen SHA. NeuroKit2: a Python toolbox for neurophysiological signal processing. Behav Res Methods. 2021;53:1689–1696. doi: 10.3758/s13428-020-01516-y
[Go to Citation](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#core-R19-1)
[Crossref](https://doi.org/10.3758/s13428-020-01516-y)
[PubMed](https://pubmed.ncbi.nlm.nih.gov/33528817/)
[Google Scholar](https://scholar.google.com/scholar_lookup?title=NeuroKit2%3A+a+Python+toolbox+for+neurophysiological+signal+processing.&author=D+Makowski&author=T+Pham&author=ZJ+Lau&author=JC+Brammer&author=F+Lespinasse&author=H+Pham&author=C+Sch%C3%B6lzel&author=SHA+Chen&publication_year=2021&journal=Behav+Res+Methods&pages=1689-1696&doi=10.3758%2Fs13428-020-01516-y&pmid=33528817)
20.
Virtanen P, Gommers R, Oliphant TE, Haberland M, Reddy T, Cournapeau D, Burovski E, Peterson P, Weckesser W, Bright J, et al; SciPy 1.0 Contributors. SciPy 1.0: fundamental algorithms for scientific computing in Python. Nat Methods. 2020;17:261–272. doi: 10.1038/s41592-019-0686-2
[Go to Citation](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#core-R20-1)
[Crossref](https://doi.org/10.1038/s41592-019-0686-2)
[PubMed](https://pubmed.ncbi.nlm.nih.gov/32015543/)
[Google Scholar](https://scholar.google.com/scholar_lookup?title=SciPy+1.0%3A+fundamental+algorithms+for+scientific+computing+in+Python.&author=P+Virtanen&author=R+Gommers&author=TE+Oliphant&author=M+Haberland&author=T+Reddy&author=D+Cournapeau&author=E+Burovski&author=P+Peterson&author=W+Weckesser&author=J+Bright&publication_year=2020&journal=Nat+Methods&pages=261-272&doi=10.1038%2Fs41592-019-0686-2&pmid=32015543)
21.
Køber L, Bloch Thomsen PE, Møller M, Torp-Pedersen C, Carlsen J, Sandøe E, Egstrup K, Agner E, Videbaek J, Marchant B, et al; Danish Investigations of Arrhythmia and Mortality on Dofetilide (DIAMOND) Study Group. Effect of dofetilide in patients with recent myocardial infarction and left-ventricular dysfunction: a randomised trial. Lancet. 2000;356:2052–2058. doi: 10.1016/s0140-6736(00)03402-4
[Go to Citation](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#core-R21-1)
[Crossref](https://doi.org/10.1016/s0140-6736\(00\)03402-4)
[PubMed](https://pubmed.ncbi.nlm.nih.gov/11145491/)
[Google Scholar](https://scholar.google.com/scholar_lookup?title=Effect+of+dofetilide+in+patients+with+recent+myocardial+infarction+and+left-ventricular+dysfunction%3A+a+randomised+trial.&author=L+K%C3%B8ber&author=PE+Bloch+Thomsen&author=M+M%C3%B8ller&author=C+Torp-Pedersen&author=J+Carlsen&author=E+Sand%C3%B8e&author=K+Egstrup&author=E+Agner&author=J+Videbaek&author=B+Marchant&publication_year=2000&journal=Lancet&pages=2052-2058&doi=10.1016%2Fs0140-6736%2800%2903402-4&pmid=11145491)
22.
Torp-Pedersen C, Møller M, Bloch-Thomsen PE, Køber L, Sandøe E, Egstrup K, Agner E, Carlsen J, Videbaek J, Marchant B, et al. Dofetilide in patients with congestive heart failure and left ventricular dysfunction. Danish Investigations of Arrhythmia and Mortality on Dofetilide Study Group. N Engl J Med. 1999;341:857–865. doi: 10.1056/NEJM199909163411201
[Go to Citation](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#core-R22-1)
[Crossref](https://doi.org/10.1056/NEJM199909163411201)
[PubMed](https://pubmed.ncbi.nlm.nih.gov/10486417/)
[Google Scholar](https://scholar.google.com/scholar_lookup?title=Dofetilide+in+patients+with+congestive+heart+failure+and+left+ventricular+dysfunction.+Danish+Investigations+of+Arrhythmia+and+Mortality+on+Dofetilide+Study+Group.&author=C+Torp-Pedersen&author=M+M%C3%B8ller&author=PE+Bloch-Thomsen&author=L+K%C3%B8ber&author=E+Sand%C3%B8e&author=K+Egstrup&author=E+Agner&author=J+Carlsen&author=J+Videbaek&author=B+Marchant&publication_year=1999&journal=N+Engl+J+Med&pages=857-865&doi=10.1056%2FNEJM199909163411201&pmid=10486417)
23.
Cho JH, Youn SJ, Moore JC, Kyriakakis R, Vekstein C, Militello M, Poe SM, Wolski K, Tchou PJ, Varma N, et al. Safety of oral dofetilide reloading for treatment of atrial arrhythmias. Circ Arrhythm Electrophysiol. 2017;10:1–9. doi: 10.1161/circep.117.005333
[Go to Citation](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#core-R23-1)
[Crossref](https://www.ahajournals.org/servlet/linkout?suffix=e_1_3_3_24_2&dbid=4&doi=10.1161%2FCIRCULATIONAHA.125.077494&key=10.1161%2FCIRCEP.117.005333&site=aha-site)
[Google Scholar](https://scholar.google.com/scholar_lookup?title=Safety+of+oral+dofetilide+reloading+for+treatment+of+atrial+arrhythmias.&author=JH+Cho&author=SJ+Youn&author=JC+Moore&author=R+Kyriakakis&author=C+Vekstein&author=M+Militello&author=SM+Poe&author=K+Wolski&author=PJ+Tchou&author=N+Varma&publication_year=2017&journal=Circ+Arrhythm+Electrophysiol&pages=1-9&doi=10.1161%2FCIRCEP.117.005333)
24.
Rautaharju PM, Surawicz B, Gettes LS, Bailey JJ, Childers R, Deal BJ, Gorgels A, Hancock EW, Josephson M, Kligfield P, et al; American Heart Association Electrocardiography and Arrhythmias Committee, Council on Clinical Cardiology. AHA/ACCF/HRS recommendations for the standardization and interpretation of the electrocardiogram: part IV: the ST segment, T and U waves, and the QT interval: a scientific statement from the American Heart Association Electrocardiography and Arrhythmias Committee, Council on Clinical Cardiology; the American College of Cardiology Foundation; and the Heart Rhythm Society. Endorsed by the International Society for Computerized Electrocardiology. J Am Coll Cardiol. 2009;53:982–991. doi: 10.1016/j.jacc.2008.12.014
[Go to Citation](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#core-R24-1)
[Crossref](https://doi.org/10.1016/j.jacc.2008.12.014)
[PubMed](https://pubmed.ncbi.nlm.nih.gov/19281931/)
[Google Scholar](https://scholar.google.com/scholar_lookup?title=AHA%2FACCF%2FHRS+recommendations+for+the+standardization+and+interpretation+of+the+electrocardiogram%3A+part+IV%3A+the+ST+segment%2C+T+and+U+waves%2C+and+the+QT+interval%3A+a+scientific+statement+from+the+American+Heart+Association+Electrocardiography+and+Arrhythmias+Committee%2C+Council+on+Clinical+Cardiology%3B+the+American+College+of+Cardiology+Foundation%3B+and+the+Heart+Rhythm+Society.+Endorsed+by+the+International+Society+for+Computerized+Electrocardiology.&author=PM+Rautaharju&author=B+Surawicz&author=LS+Gettes&author=JJ+Bailey&author=R+Childers&author=BJ+Deal&author=A+Gorgels&author=EW+Hancock&author=M+Josephson&author=P+Kligfield&publication_year=2009&journal=J+Am+Coll+Cardiol&pages=982-991&doi=10.1016%2Fj.jacc.2008.12.014&pmid=19281931)
25.
Wan EY, Ghanbari H, Akoum N, Itzhak Attia Z, Asirvatham SJ, Chung EH, Dagher L, Al-Khatib SM, Stuart Mendenhall G, McManus DD, et al. HRS white paper on clinical utilization of digital health technology. Cardiovasc Digit Health J. 2021;2:196–211. doi: 10.1016/j.cvdhj.2021.07.001
[Go to Citation](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#core-R25-1)
[Crossref](https://doi.org/10.1016/j.cvdhj.2021.07.001)
[PubMed](https://pubmed.ncbi.nlm.nih.gov/35265910/)
[Google Scholar](https://scholar.google.com/scholar_lookup?title=HRS+white+paper+on+clinical+utilization+of+digital+health+technology.&author=EY+Wan&author=H+Ghanbari&author=N+Akoum&author=Z+Itzhak+Attia&author=SJ+Asirvatham&author=EH+Chung&author=L+Dagher&author=SM+Al-Khatib&author=G+Stuart+Mendenhall&author=DD+McManus&publication_year=2021&journal=Cardiovasc+Digit+Health+J&pages=196-211&doi=10.1016%2Fj.cvdhj.2021.07.001&pmid=35265910)
26.
Giudicessi JR, Schram M, Bos JM, Galloway CD, Shreibati JB, Johnson PW, Carter RE, Disrud LW, Kleiman R, Attia ZI, et al. Artificial intelligence-enabled assessment of the heart rate corrected QT interval using a mobile electrocardiogram device. Circulation. 2021;143:1274–1286. doi: 10.1161/CIRCULATIONAHA.120.050231
[Go to Citation](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#core-R26-1)
[Crossref](https://www.ahajournals.org/servlet/linkout?suffix=e_1_3_3_27_2&dbid=4&doi=10.1161%2FCIRCULATIONAHA.125.077494&key=10.1161%2FCIRCULATIONAHA.120.050231&site=aha-site)
[PubMed](https://pubmed.ncbi.nlm.nih.gov/33517677/)
[Google Scholar](https://scholar.google.com/scholar_lookup?title=Artificial+intelligence-enabled+assessment+of+the+heart+rate+corrected+QT+interval+using+a+mobile+electrocardiogram+device.&author=JR+Giudicessi&author=M+Schram&author=JM+Bos&author=CD+Galloway&author=JB+Shreibati&author=PW+Johnson&author=RE+Carter&author=LW+Disrud&author=R+Kleiman&author=ZI+Attia&publication_year=2021&journal=Circulation&pages=1274-1286&doi=10.1161%2FCIRCULATIONAHA.120.050231&pmid=33517677)
27.
Tisdale J, Wroblewski HA, Kingery JR, Trujillo TN, Overholser BR, Kovacs RJ. A risk score to predict QT interval prolongation in hospitalized patients. J Am Coll Cardiol. 2011;57:E112. doi: 10.1016/s0735-1097(11)60112-5
[Go to Citation](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#core-R27-1)
[Crossref](https://doi.org/10.1016/s0735-1097\(11\)60112-5)
[Google Scholar](https://scholar.google.com/scholar_lookup?title=A+risk+score+to+predict+QT+interval+prolongation+in+hospitalized+patients.&author=J+Tisdale&author=HA+Wroblewski&author=JR+Kingery&author=TN+Trujillo&author=BR+Overholser&author=RJ+Kovacs&publication_year=2011&journal=J+Am+Coll+Cardiol&pages=E112&doi=10.1016%2Fs0735-1097%2811%2960112-5)
28.
Alam R, Aguirre A, Stultz CM. Detecting QT prolongation from a single-lead ECG with deep learning. PLOS Digit Health. 2024;3:e0000539. doi: 10.1371/journal.pdig.0000539
[Go to Citation](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#core-R28-1)
[Crossref](https://doi.org/10.1371/journal.pdig.0000539)
[PubMed](https://pubmed.ncbi.nlm.nih.gov/38917157/)
[Google Scholar](https://scholar.google.com/scholar_lookup?title=Detecting+QT+prolongation+from+a+single-lead+ECG+with+deep+learning.&author=R+Alam&author=A+Aguirre&author=CM+Stultz&publication_year=2024&journal=PLOS+Digit+Health&pages=e0000539&doi=10.1371%2Fjournal.pdig.0000539&pmid=38917157)
[View full text](https://www.ahajournals.org/doi/full/10.1161/CIRCULATIONAHA.125.077494)|[Download PDF](https://www.ahajournals.org/doi/pdf/10.1161/CIRCULATIONAHA.125.077494?download=true)
Now Reading:
Deep Learning–Based Continuous QT Monitoring to Identify High-Risk Prolongation Events After Class III Antiarrhythmic Initiation
[Track Citations](https://www.ahajournals.org/action/addCitationAlert?doi=10.1161%2FCIRCULATIONAHA.125.077494)
[Add to favorites](https://www.ahajournals.org/personalize/addFavoritePublication?doi=10.1161%2FCIRCULATIONAHA.125.077494)
[Share](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#)
[PREVIOUS ARTICLERandomized Controlled Trial of Mechanical Thrombectomy With Anticoagulation Versus Anticoagulation Alone for Acute Intermediate-High Risk Pulmonary Embolism: Primary Outcomes From the STORM-PE Trial](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077232)[NEXT ARTICLEN-Palmitoyl Glutamine Is a Candidate Mediator of Cardiorespiratory Fitness](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.074187)
[Circulation](https://www.ahajournals.org/journal/circ)
[Browse](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#)
[Collections](https://www.ahajournals.org/collections)
[Subject Terms](https://www.ahajournals.org/subjects)
[AHA Journal Podcasts](https://www.ahajournals.org/podcasts)
[Trend Watch](https://www.ahajournals.org/trend-watch)
[Resources](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#)
[CME](https://www.ahajournals.org/cme)
[Journal Metrics](https://www.ahajournals.org/metrics)
[Early Career Resources](https://www.ahajournals.org/early-career)
[AHA Journals @ Meetings](https://www.ahajournals.org/meetings)
[Information](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#)
[For Authors](https://www.ahajournals.org/author-hub)
[For Reviewers](https://www.ahajournals.org/for-reviewers)
[For Subscribers](https://www.ahajournals.org/custserv)
[For International Users](https://www.ahajournals.org/international-users)
[Arteriosclerosis, Thrombosis, and Vascular Biology](https://www.ahajournals.org/journal/atvb)
[Circulation](https://www.ahajournals.org/journal/circ)
[Circulation Research](https://www.ahajournals.org/journal/res)
[Hypertension](https://www.ahajournals.org/journal/hyp)
[Stroke](https://www.ahajournals.org/journal/str)
[Journal of the American Heart Association](https://www.ahajournals.org/journal/jaha)
[Circulation: Arrhythmia and Electrophysiology](https://www.ahajournals.org/journal/circep)
[Circulation: Cardiovascular Imaging](https://www.ahajournals.org/journal/circimaging)
[Circulation: Cardiovascular Interventions](https://www.ahajournals.org/journal/circinterventions)
[Circulation: Population Health and Outcomes](https://www.ahajournals.org/journal/circpopoutcomes)
[Circulation: Genomic and Precision Medicine](https://www.ahajournals.org/journal/circgen)
[Circulation: Heart Failure](https://www.ahajournals.org/journal/circheartfailure)
[Stroke: Vascular and Interventional Neurology](https://www.ahajournals.org/journal/svin)
[Annals of Internal Medicine: Clinical Cases](https://www.ahajournals.org/aimcc)
This page is managed by Wolters Kluwer Health, LLC and/or its affiliates or subsidiaries. [Wolters Kluwer Privacy Policy](https://www.ovid.com/global/privacy-policy)
Manage Cookie Preferences
Back to top
National Center7272 Greenville Ave.Dallas, TX 75231
Customer Service1-800-AHA-USA-1
1-800-242-8721
HoursMonday - Friday: 7 a.m. – 7 p.m. CT
Saturday: 9 a.m. - 5 p.m. CT
Closed on Sundays
Tax Identification Number13-5613797
[ABOUT US](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#)
[About the AHA/ASA](https://www.heart.org/en/about-us)
[Annual report](https://www.heart.org/annualreport)
[AHA Financial Information](https://www.heart.org/en/about-us/aha-financial-information)
[Careers](https://heart.jobs/?utm_campaign=heart.org-Footer&vs=2896&utm_medium=Other&utm_source=heart.org-Footer)
[International Programs](https://www.heart.org/en/about-us/international-programs)
[Latest Heart and Stroke News](https://www.heart.org/en/news)
[AHA/ASA Media Newsroom](https://newsroom.heart.org/)
[GET INVOLVED](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#)
[Donate](https://mygiving.heart.org/-/XZTZBDFD?s_src=20U2W1UEMG&s_subsrc=ahajournals_footer_donatenow)
[Advocate](http://www.yourethecure.org/)
[Volunteer](https://www.heart.org/HEARTORG/volunteer/volunteerForm.jsp)
[ShopHeart](https://www.shopheart.org/?a=aha-heart.org-bottom-navigation&utm_source=heart.org&utm_medium=referral&utm_campaign=aha-heart.org-bottom-navigation)
[ShopCPR](https://shopcpr.heart.org/)
[OUR SITES](https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.125.077494#)
[American Heart Association](https://www.heart.org/en/)
[American Stroke Association](https://www.stroke.org/en/)
[CPR & ECC](https://cpr.heart.org/en/)
[Go Red For Women](https://www.goredforwomen.org/en/)
[More Sites](https://www.heart.org/en/about-us/aha-asa-website-directory)
[AHA Careers](https://heart.jobs/?utm_campaign=heart.org-Footer&vs=2896&utm_medium=Other&utm_source=heart.org-Footer)
[AHA Privacy Policy](https://www.heart.org/en/about-us/statements-and-policies/privacy-statement)
[Medical Advice Disclaimer](https://www.heart.org/en/about-us/statements-and-policies/medical-advice)
[Copyright Policy](https://www.heart.org/en/about-us/statements-and-policies/copyright)
[Accessibility Statement](https://www.heart.org/en/about-us/statements-and-policies/accessibility-statement)
[Ethics Policy](https://www.heart.org/en/about-us/statements-and-policies/ethics-policy)
[Conflict of Interes Policy](https://www.heart.org/en/about-us/statements-and-policies/conflict-of-interest-policy)
[Linking Policy](https://www.heart.org/en/about-us/statements-and-policies/linking-policy)
[Whistleblower Policy](https://www.heart.org/en/about-us/statements-and-policies/whistleblower-policy)
[Content Editorial Guidelines](https://www.heart.org/en/about-us/editorial-guidelines)
[Diversity](https://www.heart.org/en/about-us/diversity-inclusion)
[Suppliers & Providers](https://www.heart.org/en/about-us/procurement-services/procurement-services-department)
[State Fundraising Notices](https://www.heart.org/en/about-us/statements-and-policies/state-fundraising-notices)
©2026 American Heart Association, Inc. All rights reserved. Unauthorized use prohibited.The American Heart Association is a qualified 501(c)(3) tax-exempt organization.*Red Dress ™ DHHS, Go Red ™ AHA ; National Wear Red Day® is a registered trademark.