#!/usr/bin/env python3
import sys

def main():
    with open("slides/generate_pptx.py", "r") as f:
        code = f.read()

    old_slide8_marker = "    # Slide 8: Empirical Validation of ECG-AIM: Benchmark Performance"
    if old_slide8_marker not in code:
        print("Marker not found, maybe already updated?")
        if "Slide 8: Figure 2: ECG-AIM Multi-Task Training Pipeline" in code:
            print("Already updated!")
            return
        sys.exit(1)

    split_idx = code.find(old_slide8_marker)
    prefix = code[:split_idx]

    prefix = prefix.replace(
        'set_notes(s7, "Alex, this is the architectural engine that powers our system: ECG-AIM champion checkpoint conv15e_A0_wave_noSSL_gated_add. It combines continuous Morlet wavelets with a TimeSformer spatio-temporal encoder, gated-add fusion, and an axial decoder capable of predicting both 12-lead waveforms and explicit fiducials.")',
        'set_notes(s7, "Alex, this is the architectural engine that powers our system: ECG-AIM champion checkpoint conv15e_A0_wave_noSSL_gated_add. It combines continuous Morlet wavelets with a TimeSformer spatio-temporal encoder, gated-add fusion, and an axial decoder capable of predicting both 12-lead waveforms and explicit fiducials. Now let us examine Figure 2 to understand how this model is actually trained on 12-lead data and the multi-task loss formulation.")'
    )
    prefix = prefix.replace("(12 slides)", "(14 slides)")

    new_slides_code = '''    # =========================================================================
    # Slide 8: Figure 2: ECG-AIM Multi-Task Training Pipeline
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "Figure 2: ECG-AIM Multi-Task Training Pipeline", "Simulating Single-Lead Input from 12-Lead Records and Multi-Objective Supervision")
    add_footer(s8, 8)

    fig2_train_path = "slides/figures/ecg_aim_training_pipeline.png"
    if os.path.exists(fig2_train_path):
        s8.shapes.add_picture(fig2_train_path, Inches(2.05), Inches(1.7), Inches(9.23), Inches(5.15))

    set_notes(s8, "Alex and Chris, this figure illustrates exactly how we trained the ECG-AIM champion model before deploying it to single-lead signals. Notice the training setup: First, on the left, we ingest full 12-lead hospital records from PTB-XL and MIMIC-IV. We simulate single-lead wrist input by masking out the other 11 leads, so the model only observes Lead I. Second, the single-lead signal passes through our dual-branch wavelet transformer backbone to predict both continuous 12-lead waveforms and P-QRS-T boundary delineations. Third, look at the loss formulation on the right: we supervise the waveform reconstruction with a Combinatorial Composite Loss (MSE, Pearson r correlation, Wyatt ST-T phase loss, and spectral loss) plus Einthoven limb consistency (Lead I + Lead III = Lead II), alongside multi-task Cross-Entropy and Dice loss on the delineation head. Most importantly, look at the bottom callout: in training, the single lead is a synthetic proxy derived from hospital ECGs on supine patients. It lacks dry-electrode contact noise, wrist impedance, and wearable DSP filtering. This fundamental training gap is the exact scientific motivation for why we need our Sunnybrook prospective paired trial data to calibrate and validate the model on real clinical hardware!")

    # =========================================================================
    # Slide 9: Empirical Validation of ECG-AIM: Benchmark Performance
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "Empirical Validation of ECG-AIM: Benchmark Performance", "conv15e_A0_wave_noSSL_gated_add Evaluated on 175,890 Patient Observations")
    add_footer(s9, 9)

    rows_emp = [
        ("Clinical Evaluation Dimension", "Factorial Baseline", "Stanford 3DRECON-QT", "ECG-AIM (Ours)", "Statistical Contrast / Rigor"),
        ("Precordial Chest Mean Pearson r", "0.570", "0.732", "0.776", "Delta = +0.206 (p < 10^-300), Ranked #3 of 55"),
        ("Anterior Lead V3 Pearson r", "0.382", "0.670", "0.739", "Delta = +0.357 (p < 10^-300), Anterior LV wall view"),
        ("QRS Duration Error (MAE / Median)", "13.85 ms", "10.26 ms", "10.05 / 8.0 ms", "Sub-10ms precision; tracks depolarization"),
        ("Conduction Delay Concordance (>120ms)", "90.7%", "94.3%", "94.3%", "aOR = 1.84 (p < 10^-17), Specificity: 96.5%"),
        ("Sokolow-Lyon LVH Voltage MAE", "1.13 mV", "0.63 mV", "0.72 mV", "ICC = 0.714, Quantifies chamber hypertrophy"),
        ("EchoNext Structural Heart Disease", "68.8%", "46.5% (Collapse)", "71.6%", "Preserves chamber echocardiographic pathology")
    ]

    t_shape9 = s9.shapes.add_table(7, 5, Inches(0.8), Inches(1.8), Inches(11.7), Inches(2.7))
    table9 = t_shape9.table
    table9.columns[0].width = Inches(3.2)
    table9.columns[1].width = Inches(1.8)
    table9.columns[2].width = Inches(2.1)
    table9.columns[3].width = Inches(1.8)
    table9.columns[4].width = Inches(2.8)

    for r_idx, row in enumerate(rows_emp):
        for c_idx, val in enumerate(row):
            cell = table9.cell(r_idx, c_idx)
            cell.text = val
            p = cell.text_frame.paragraphs[0]
            p.font.name = "Arial"
            p.font.size = Pt(10)
            if r_idx == 0:
                p.font.bold = True
                p.font.color.rgb = WHITE
                cell.fill.solid()
                cell.fill.fore_color.rgb = NAVY
            else:
                cell.fill.solid()
                cell.fill.fore_color.rgb = LIGHT_BG if r_idx % 2 == 1 else WHITE
                if c_idx == 0:
                    p.font.bold = True
                    p.font.color.rgb = NAVY
                elif c_idx == 2 and "Collapse" in val:
                    p.font.bold = True
                    p.font.color.rgb = CRIMSON
                elif c_idx == 3:
                    p.font.bold = True
                    p.font.color.rgb = NAVY
                else:
                    p.font.color.rgb = SLATE_DARK

    # Left callout card: Wavelet Advantage
    c1_9 = s9.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(4.7), Inches(5.7), Inches(2.0))
    c1_9.fill.solid()
    c1_9.fill.fore_color.rgb = LIGHT_BG
    c1_9.line.color.rgb = NAVY
    c1_9.line.width = Pt(1.5)

    tb1_9 = s9.shapes.add_textbox(Inches(1.0), Inches(4.8), Inches(5.3), Inches(1.8))
    tf1_9 = tb1_9.text_frame
    tf1_9.word_wrap = True
    p = tf1_9.paragraphs[0]
    p.text = "Key Technical Advantage: Morlet Wavelets"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = NAVY

    p_w1 = tf1_9.add_paragraph()
    p_w1.text = "Continuous Morlet wavelets preserve high-frequency QRS spikes and ST-T repolarization without baseline distortion, outperforming Stanford static CNN on chest leads (r = 0.776 vs 0.732)."
    p_w1.font.size = Pt(10.5)
    p_w1.font.color.rgb = SLATE_DARK
    p_w1.space_before = Pt(4)

    p_w2 = tf1_9.add_paragraph()
    p_w2.text = "Patient-clustered MMRM confirms sub-10ms precision (median error 8.0 ms, Delta = -3.72 ms vs factorial baseline, p < 10^-300)."
    p_w2.font.size = Pt(10)
    p_w2.font.color.rgb = SLATE_MUTED
    p_w2.space_before = Pt(3)

    # Right callout card: Ansari Collapse
    c2_9 = s9.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(4.7), Inches(5.7), Inches(2.0))
    c2_9.fill.solid()
    c2_9.fill.fore_color.rgb = LIGHT_BG
    c2_9.line.color.rgb = CRIMSON
    c2_9.line.width = Pt(1.5)

    tb2_9 = s9.shapes.add_textbox(Inches(7.0), Inches(4.8), Inches(5.3), Inches(1.8))
    tf2_9 = tb2_9.text_frame
    tf2_9.word_wrap = True
    p = tf2_9.paragraphs[0]
    p.text = "Overcoming Ansari Structural Collapse"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = CRIMSON

    p_a1 = tf2_9.add_paragraph()
    p_a1.text = "Ansari scalar QT loss causes structural collapse on EchoNext (46.5% overall, dropping to 33.3% on reduced LVEF <= 45% and 30.2% on severe wall thickening)."
    p_a1.font.size = Pt(10.5)
    p_a1.font.color.rgb = SLATE_DARK
    p_a1.space_before = Pt(4)

    p_a2 = tf2_9.add_paragraph()
    p_a2.text = "ECG-AIM retains 71.6% concordance across 1,000 paired exams, preserving 3D ventricular geometry and echocardiographic pathology."
    p_a2.font.size = Pt(10)
    p_a2.font.color.rgb = SLATE_MUTED
    p_a2.space_before = Pt(3)

    set_notes(s9, "Alex and Chris, this slide presents the strictly empirical benchmark across all 55 models evaluated with patient-clustered MMRM and GEE. Notice our champion model: conv15e_A0_wave_noSSL_gated_add. Compared to the Stanford 3DRECON-QT model published in Circulation, our model achieves higher precordial chest lead correlation—0.776 vs 0.732—and an anterior Lead V3 correlation of 0.739 vs 0.670. Crucially, look at the bottom row: Ansari model collapsed to 46.5% concordance on echocardiographic structural heart disease in EchoNext because their scalar QT loss ignored chamber geometry. Our wavelet-gated transformer maintains 71.6% structural concordance and tracks QRS duration with a median error of just 8.0 milliseconds!")

    # =========================================================================
    # Slide 10: Data Collection Methodology: Dual-Stream Protocol
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    add_header(s10, "Data Collection Methodology: Dual-Stream Protocol", "Seamless Integration: Daily At-Home Surveillance + In-Clinic Paired Anchors")
    add_footer(s10, 10)

    c1 = s10.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.8), Inches(5.7), Inches(4.8))
    c1.fill.solid()
    c1.fill.fore_color.rgb = LIGHT_BG
    c1.line.color.rgb = NAVY
    c1.line.width = Pt(1.5)

    tb = s10.shapes.add_textbox(Inches(1.1), Inches(2.0), Inches(5.1), Inches(4.3))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Stream A: Ambulatory Home Surveillance"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = NAVY

    items_s10_1 = [
        "At-Home Daily ECG: Patient records daily 30s smartwatch ECG (Lead I) for 6 months post-ablation.",
        "Continuous Gold Standard: 14-day continuous CardioSTAT patch worn at 3 months and 6 months post-procedure.",
        "Primary Outcome Harvested: Exact time to recurrent AF, cumulative daily AF burden (% time in AF), and asymptomatic episode yield.",
        "Data Pipeline: Automated upload via SRI custom dashboard directly onto secure Sunnybrook servers behind the hospital firewall."
    ]
    for it in items_s10_1:
        p = tf.add_paragraph()
        p.text = f"- {it}"
        p.font.size = Pt(12)
        p.font.color.rgb = SLATE_DARK
        p.space_before = Pt(8)

    c2 = s10.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.8))
    c2.fill.solid()
    c2.fill.fore_color.rgb = LIGHT_BG
    c2.line.color.rgb = CYAN
    c2.line.width = Pt(1.5)

    tb2 = s10.shapes.add_textbox(Inches(7.1), Inches(2.0), Inches(5.1), Inches(4.3))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    p = tf2.paragraphs[0]
    p.text = "Stream B: In-Clinic Paired Anchor Protocol"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = CYAN

    items_s10_2 = [
        "Hospital Touchpoints: Routine visits already mandated in clinical care:",
        "  * Visit 1: Baseline pre-ablation / pre-cardioversion workup",
        "  * Visit 2: Index procedure day (post-ablation acute restoration)",
        "  * Visit 3: 3-month routine follow-up clinic visit",
        "  * Visit 4: 6-month study closeout clinic visit",
        "Synchronous Acquisition: Clinical 12-lead ECG is recorded simultaneously with smartwatch Lead I (delta t = 0 min).",
        "Patient Burden: Zero extra hospital visits, zero additional blood draws, adds <= 5 minutes per routine clinic encounter."
    ]
    for it in items_s10_2:
        p = tf2.add_paragraph()
        p.text = f"- {it}"
        p.font.size = Pt(11.5)
        p.font.color.rgb = SLATE_DARK
        p.space_before = Pt(6)

    set_notes(s10, "Chris and Alex, this is the exact data collection methodology. It operates as two synchronized streams: Stream A is the ambulatory home surveillance from Protocol v1.1. Stream B is the in-clinic paired protocol during routine hospital visits. Patients are already at Sunnybrook for four standard visits. When they get their standard 12-lead ECG, the research coordinator records a 30-second smartwatch tracing at the exact same moment. It adds less than 5 minutes to routine care, requires zero extra visits, and creates perfectly synchronized ground-truth pairs!")

    # =========================================================================
    # Slide 11: The Paired In-Clinic Evaluation Set: The Gold Standard
    # =========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    add_header(s11, "The Paired In-Clinic Evaluation Set", "The Most Critical Dataset: Breaking the Synthetic Proxy Trap")
    add_footer(s11, 11)

    rows_eval = [
        ("Methodological Dimension", "Stanford 3DRECON-QT (Ansari 2026)", "Sunnybrook Wearable Trial (Our Evaluation Set)"),
        ("Hardware Evaluation Cohort", "Retrospective CareLink archive (N = 26 patients).", "Prospective cohort: 96 patients, ~384 paired recordings."),
        ("Temporal Synchronization", "+/- 7 days between ICM and 12L (|delta HR| <= 20 bpm).", "delta t = 0 synchronous: Captured in same clinical exam room."),
        ("Physiological Confounding", "Electrolytes, posture, autonomic tone vary over 7 days.", "Identical resting state: Supine, same autonomic state, same HR."),
        ("Clinical Transitions Captured", "Static cross-sectional outpatient snapshots.", "Longitudinal transitions: Baseline AF -> Acute ablation -> Healed."),
        ("Scientific Role", "Device proof-of-concept for publication.", "The Gold Standard Benchmark: Bridges synthetic AI to real wearables.")
    ]

    t_shape11 = s11.shapes.add_table(6, 3, Inches(0.8), Inches(1.8), Inches(11.7), Inches(3.8))
    table11 = t_shape11.table
    table11.columns[0].width = Inches(2.7)
    table11.columns[1].width = Inches(4.5)
    table11.columns[2].width = Inches(4.5)

    for r_idx, row in enumerate(rows_eval):
        for c_idx, val in enumerate(row):
            cell = table11.cell(r_idx, c_idx)
            cell.text = val
            p = cell.text_frame.paragraphs[0]
            p.font.name = "Arial"
            p.font.size = Pt(11)
            if r_idx == 0:
                p.font.bold = True
                p.font.color.rgb = WHITE
                cell.fill.solid()
                cell.fill.fore_color.rgb = NAVY
            else:
                p.font.color.rgb = SLATE_DARK
                cell.fill.solid()
                cell.fill.fore_color.rgb = LIGHT_BG if r_idx % 2 == 1 else WHITE
                if c_idx == 0:
                    p.font.bold = True

    card_eval = s11.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(5.8), Inches(11.7), Inches(1.0))
    card_eval.fill.solid()
    card_eval.fill.fore_color.rgb = LIGHT_BG
    card_eval.line.color.rgb = SUCCESS_GREEN
    card_eval.line.width = Pt(1.5)

    tb = s11.shapes.add_textbox(Inches(1.0), Inches(5.9), Inches(11.3), Inches(0.8))
    tf = tb.text_frame
    p = tf.paragraphs[0]
    p.text = "Why This Evaluation Set is the Crown Jewel: 99% of published ECG AI papers test only on synthetic derived leads from 12-lead files. That only tests math, not wearable physics (dry electrodes, wrist impedance, DSP filters). Our prospective paired dataset is the first true clinical-hardware benchmark for wearable 12-lead reconstruction."
    p.font.size = Pt(11.5)
    p.font.bold = True
    p.font.color.rgb = SUCCESS_GREEN

    set_notes(s11, "Alex, this slide explains why our computing science group needs this data so desperately: this evaluation set is the crown jewel of the entire project! Ansari and Rogers had to search retrospectively through old CareLink transmissions and only found 26 patients within +/- 7 days. In our trial, we are prospectively acquiring exact delta-t equals zero pairs across 96 patients at 4 distinct time points—yielding nearly 400 paired recordings! Furthermore, this evaluation set captures true longitudinal transitions: pre-ablation AF, acute sinus restoration, and 3-to-6-month healed rhythm. This completely breaks the synthetic proxy trap that invalidates other AI papers.")

    # =========================================================================
    # Slide 12: Figure 3: Integrated Clinical-AI Trial Pipeline
    # =========================================================================
    s12 = prs.slides.add_slide(blank_layout)
    add_header(s12, "Figure 3: Integrated Clinical-AI Trial Pipeline", "Tri-Modal Data Acquisition Engine, Domain Adaptation, and Dual Validation")
    add_footer(s12, 12)

    fig3_path = "slides/figures/paired_trial_model_integration.png"
    if os.path.exists(fig3_path):
        s12.shapes.add_picture(fig3_path, Inches(0.8), Inches(1.8), Inches(6.2), Inches(4.9))

    c_right = s12.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.3), Inches(1.8), Inches(5.2), Inches(4.9))
    c_right.fill.solid()
    c_right.fill.fore_color.rgb = LIGHT_BG
    c_right.line.color.rgb = BORDER_COLOR
    c_right.line.width = Pt(1)

    tb = s12.shapes.add_textbox(Inches(7.5), Inches(1.9), Inches(4.8), Inches(4.7))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "4-Stage Integration Engine"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = NAVY

    stages = [
        ("Stage 1: Foundation Pretraining", "Trained on PTB-XL, MIMIC-IV, and Icentia11k to embed general cardiac electrophysiology priors."),
        ("Stage 2: Tri-Modal Data Engine", "Smartwatch daily rhythm + in-clinic paired Lead I; 14-day continuous CardioSTAT patch; Hospital 12-lead machine."),
        ("Stage 3: Domain Adaptation", "Estimates device transfer function H(omega) and tunes low-rank adapters (LoRA) on paired in-clinic signals."),
        ("Stage 4: Dual Clinical Endpoints", "Primary Efficacy: AF recurrence & burden vs. CardioSTAT. Secondary AI: 12-lead reconstruction & biomarker recovery.")
    ]

    for title, desc in stages:
        p = tf.add_paragraph()
        p.text = f"- {title}: {desc}"
        p.font.size = Pt(11)
        p.font.color.rgb = SLATE_DARK
        p.space_before = Pt(8)

    set_notes(s12, "Alex and Chris, this is Figure 3: the master architecture diagram showing how the clinical trial and AI engineering integrate seamlessly. Stage 1 is foundation models pretrained on big public data. Stage 2 is our tri-modal data acquisition engine. Stage 3 uses paired in-clinic data to calibrate the hardware transfer function and adapt the model. Stage 4 shows the clear separation of endpoints: primary clinical AF surveillance vs. secondary translational 12-lead reconstruction.")

    # =========================================================================
    # Slide 13: Sample Size, Statistical Power & Clustered Modeling
    # =========================================================================
    s13 = prs.slides.add_slide(blank_layout)
    add_header(s13, "Sample Size, Statistical Power & Clustered Modeling", "Precision-Driven Sizing for Clinical Trial and Clustered Secondary AI Aim")
    add_footer(s13, 13)

    c1 = s13.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.8), Inches(5.7), Inches(4.8))
    c1.fill.solid()
    c1.fill.fore_color.rgb = LIGHT_BG
    c1.line.color.rgb = NAVY
    c1.line.width = Pt(1.5)

    tb = s13.shapes.add_textbox(Inches(1.1), Inches(2.0), Inches(5.1), Inches(4.3))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Primary Clinical Power (Protocol v1.1)"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = NAVY

    items_s13_1 = [
        "Primary Outcome: Time to AF recurrence detection.",
        "Hypothesis: Smartwatch detects recurrent AF at higher or non-inferior rate vs. 14-day patch.",
        "Assumptions: AF recurrence rate 20% with patch vs. 25% with smartwatch.",
        "Target Power: 80% power, alpha = 0.05, 10% non-inferiority margin -> N = 96 patients.",
        "Vanguard Cohort: Initial N = 20 consecutive patients to verify protocol adherence and pipeline."
    ]
    for it in items_s13_1:
        p = tf.add_paragraph()
        p.text = f"- {it}"
        p.font.size = Pt(12)
        p.font.color.rgb = SLATE_DARK
        p.space_before = Pt(8)

    c2 = s13.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.8))
    c2.fill.solid()
    c2.fill.fore_color.rgb = LIGHT_BG
    c2.line.color.rgb = CYAN
    c2.line.width = Pt(1.5)

    tb2 = s13.shapes.add_textbox(Inches(7.1), Inches(2.0), Inches(5.1), Inches(4.3))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    p = tf2.paragraphs[0]
    p.text = "Secondary AI Precision & Clustered Stats"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = CYAN

    items_s13_2 = [
        "AI Sample Size: 96 patients x 4 in-clinic visits yields >2,500 paired cardiac complexes.",
        "Precision Target: Sized for QTc MAE 95% CI margin of error <= +/- 3.5 ms.",
        "Clustering Adjustment: Multiple beats per patient violate standard i.i.d. assumptions.",
        "MMRM & GEE: Linear Mixed Models (unstructured covariance) and Generalized Estimating Equations for robust sandwich variance.",
        "Regulatory Defensibility: Primary sample size is powered on clinical events; secondary AI sample size exceeds precision requirements."
    ]
    for it in items_s13_2:
        p = tf2.add_paragraph()
        p.text = f"- {it}"
        p.font.size = Pt(11.5)
        p.font.color.rgb = SLATE_DARK
        p.space_before = Pt(6)

    set_notes(s13, "Chris and Alex, this slide demonstrates total alignment on sample size and statistics. The primary clinical trial sample size of 96 patients is calculated directly from expected AF recurrence rates—20% on patch vs 25% on smartwatch with an 80% power target. For the secondary AI aim, having 96 patients with multiple paired in-clinic visits provides over 2,500 paired cardiac complexes. Per Riley's 2024 BMJ guidelines, this is more than sufficient to bound QTc measurement error within a tight 3.5 ms margin of error. Furthermore, we pre-specify Linear Mixed Models (MMRM) and Generalized Estimating Equations (GEE) to adjust for intra-patient beat clustering.")

    # =========================================================================
    # Slide 14: Action Items & Alignment for Today's Meeting
    # =========================================================================
    s14 = prs.slides.add_slide(blank_layout)
    add_header(s14, "Action Items & Alignment for Today's Meeting", "4 Concrete Decisions to Move Forward")
    add_footer(s14, 14)

    c1 = s14.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.8), Inches(6.0), Inches(4.8))
    c1.fill.solid()
    c1.fill.fore_color.rgb = LIGHT_BG
    c1.line.color.rgb = NAVY
    c1.line.width = Pt(1.5)

    tb = s14.shapes.add_textbox(Inches(1.1), Inches(2.0), Inches(5.4), Inches(4.3))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "4 Concrete Decisions to Finalize Today"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = NAVY

    items_s14 = [
        "1. Two-Track Protocol Amendment: Approve formalizing Aim 1 (AF recurrence) as primary and Aim 2 (paired 12L reconstruction) as secondary.",
        "2. In-Clinic Paired Recording SOP: Approve the 5-minute simultaneous smartwatch + 12L recording protocol during routine clinic visits.",
        "3. Secondary Biomarker Selection: Confirm Chris priority secondary biomarker: QTc monitoring on sotalol vs. P-wave remodeling vs. conduction delay.",
        "4. Vanguard Cohort Launch: Authorize rolling out the 20-patient feasibility vanguard under Sunnybrook IRB."
    ]
    for it in items_s14:
        p = tf.add_paragraph()
        p.text = it
        p.font.size = Pt(12)
        p.font.color.rgb = SLATE_DARK
        p.space_before = Pt(10)

    c2 = s14.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.1), Inches(1.8), Inches(5.4), Inches(4.8))
    c2.fill.solid()
    c2.fill.fore_color.rgb = LIGHT_BG
    c2.line.color.rgb = CYAN
    c2.line.width = Pt(1.5)

    tb2 = s14.shapes.add_textbox(Inches(7.4), Inches(2.0), Inches(4.8), Inches(4.3))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    p = tf2.paragraphs[0]
    p.text = "Summary of Trial Benefits"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = CYAN

    items_ben = [
        "For Chris (Cardiology): High-impact clinical paper evaluating wearable AF surveillance against continuous patch monitoring.",
        "For Alex (Computer Science): World first prospectively paired wearable + 12L dataset; top-tier machine learning publications.",
        "For Industry / Funders: A clinically rigorous, dual-value proposition with zero trial design flaws.",
        "Next Step: Finalize protocol text and begin vanguard enrollment in the Arrhythmia Clinic."
    ]
    for it in items_ben:
        p = tf2.add_paragraph()
        p.text = f"- {it}"
        p.font.size = Pt(12)
        p.font.color.rgb = SLATE_DARK
        p.space_before = Pt(10)

    set_notes(s14, "To close our meeting, here are the 4 concrete decisions we need to agree on: First, confirm the two-track protocol amendment: Aim 1 primary AF recurrence, Aim 2 secondary 12-lead reconstruction. Second, approve the simple 5-minute simultaneous recording SOP during in-clinic visits. Third, decide which secondary clinical biomarker Chris wants to prioritize for the paired analysis—QTc interval tracking is our recommendation. Fourth, green-light the 20-patient vanguard cohort. Thank you Chris and Alex; let us open the floor for discussion.")

    output_path = "slides/presentation.pptx"
    prs.save(output_path)
    print(f"Presentation saved successfully to {output_path} (Total slides: 14)")

if __name__ == "__main__":
    create_deck()
'''

    with open("slides/generate_pptx.py", "w") as f:
        f.write(prefix + new_slides_code)

    print("Successfully updated generate_pptx.py to 14 slides.")

if __name__ == "__main__":
    main()
