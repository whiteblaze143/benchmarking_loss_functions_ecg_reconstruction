#!/usr/bin/env python3
"""
update_pptx_full.py
Updates generate_pptx.py to generate all 19 slides in full 1:1 alignment with slides/main.tex.
"""

def generate():
    with open("slides/generate_pptx.py", "r") as f:
        code = f.read()

    # Find the footer definition and update total_slides default to 19
    code = code.replace("def add_footer(slide, slide_num, total_slides=16):", "def add_footer(slide, slide_num, total_slides=19):")
    code = code.replace("(14 slides)", "(19 slides)")
    code = code.replace("(16 slides)", "(19 slides)")

    # Find where Slide 11 starts
    marker = "    # Slide 11: Empirical 12-Lead Reconstruction Gallery"
    if marker not in code:
        marker = "    # =========================================================================\n    # Slide 11:"
    
    split_pos = code.find(marker)
    if split_pos == -1:
        raise ValueError("Could not find Slide 11 marker in generate_pptx.py")
    
    prefix = code[:split_pos]

    new_slides_code = '''    # =========================================================================
    # Slide 11: Single-Lead Wearable Gallery: Smartwatch (Lead I) vs. Patch (Lead II)
    # =========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    add_header(s11, "Single-Lead Wearable Gallery: Smartwatch (Lead I) vs. Patch (Lead II)", "Unseen Benchmark Tracing: PTB-XL Record #10283 Across Single-Lead Sensors")
    add_footer(s11, 11)

    # Left Panel: Smartwatch Lead I
    c1_11 = s11.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.7), Inches(5.7), Inches(4.7))
    c1_11.fill.solid()
    c1_11.fill.fore_color.rgb = LIGHT_BG
    c1_11.line.color.rgb = NAVY
    c1_11.line.width = Pt(1.5)

    tb1_t = s11.shapes.add_textbox(Inches(1.0), Inches(1.75), Inches(5.3), Inches(0.35))
    tf1_t = tb1_t.text_frame
    p = tf1_t.paragraphs[0]
    p.text = "Smartwatch: Lead I (Transverse Vector)"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = NAVY

    fig_l1_path = "slides/figures/best_ecg_aim_lead1_reconstruction.png"
    if os.path.exists(fig_l1_path):
        s11.shapes.add_picture(fig_l1_path, Inches(0.95), Inches(2.15), Inches(5.4), Inches(2.8))

    tb1_b = s11.shapes.add_textbox(Inches(1.0), Inches(5.05), Inches(5.3), Inches(1.3))
    tf1_b = tb1_b.text_frame
    tf1_b.word_wrap = True
    items_l1_g = [
        "Input: Single wrist channel (Lead I smartwatch vector).",
        "Inferred: 11 missing leads (Record mean missing-lead r = 0.941).",
        "Precordials: V1 (r=0.995), V2 (r=0.989), V3 (r=0.988).",
        "Morphology: Outstanding anterior ventricular tracking."
    ]
    for idx, it in enumerate(items_l1_g):
        p = tf1_b.paragraphs[0] if idx == 0 else tf1_b.add_paragraph()
        p.text = f"- {it}"
        p.font.size = Pt(10)
        p.font.color.rgb = SLATE_DARK
        p.space_before = Pt(2)

    # Right Panel: Patch Lead II
    c2_11 = s11.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.7), Inches(5.7), Inches(4.7))
    c2_11.fill.solid()
    c2_11.fill.fore_color.rgb = LIGHT_BG
    c2_11.line.color.rgb = CYAN
    c2_11.line.width = Pt(1.5)

    tb2_t = s11.shapes.add_textbox(Inches(7.0), Inches(1.75), Inches(5.3), Inches(0.35))
    tf2_t = tb2_t.text_frame
    p = tf2_t.paragraphs[0]
    p.text = "Patch: Lead II (Inferior Axis Vector)"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = CYAN

    fig_l2_path = "slides/figures/best_ecg_aim_1lead_reconstruction.png"
    if os.path.exists(fig_l2_path):
        s11.shapes.add_picture(fig_l2_path, Inches(6.95), Inches(2.15), Inches(5.4), Inches(2.8))

    tb2_b = s11.shapes.add_textbox(Inches(7.0), Inches(5.05), Inches(5.3), Inches(1.3))
    tf2_b = tb2_b.text_frame
    tf2_b.word_wrap = True
    items_l2_g = [
        "Input: Single inferior channel (Lead II patch monitor).",
        "Inferred: 11 missing leads (Record mean missing-lead r = 0.919).",
        "Inferior Leads: aVF (r=0.992), III (r=0.963), V2 (r=0.966).",
        "Morphology: Preserves acute inferior axis and ST takeoff."
    ]
    for idx, it in enumerate(items_l2_g):
        p = tf2_b.paragraphs[0] if idx == 0 else tf2_b.add_paragraph()
        p.text = f"- {it}"
        p.font.size = Pt(10)
        p.font.color.rgb = SLATE_DARK
        p.space_before = Pt(2)

    # Bottom Banner
    card_bot11 = s11.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(6.5), Inches(11.7), Inches(0.42))
    card_bot11.fill.solid()
    card_bot11.fill.fore_color.rgb = LIGHT_BG
    card_bot11.line.color.rgb = NAVY
    card_bot11.line.width = Pt(1.5)

    tb_b11 = s11.shapes.add_textbox(Inches(1.0), Inches(6.52), Inches(11.3), Inches(0.38))
    tf_b11 = tb_b11.text_frame
    p = tf_b11.paragraphs[0]
    p.text = "Hardware Agility: Dynamic masking (M_obs) adapts to any wearable. Wrist Lead I captures the transverse dipole, yielding superior anterior precordial recovery (r > 0.98)."
    p.font.size = Pt(10.5)
    p.font.bold = True
    p.font.color.rgb = NAVY

    set_notes(s11, "Chris and Alex, here is the direct single-lead comparison between sensor modalities on unseen PTB-XL Record 10283. On the left, we input single-lead Lead I—the exact wrist-to-wrist vector acquired by our trial's Apple Watch and Fitbit devices. The model infers all other 11 leads with a record mean correlation of 0.941. Notice how cleanly it reconstructs the precordial chest leads: V1 is 0.995, V2 is 0.989, and V3 is 0.988! Because Lead I spans the transverse dipole plane, it captures anterior ventricular depolarization with remarkable fidelity. On the right, we input single-lead Lead II—the vertical vector typically captured by chest patches like CardioSTAT or Zio. Lead II achieves a record mean correlation of 0.919, with excellent recovery of inferior leads like aVF at 0.992 and III at 0.963, but slightly lower anterior resolution. This demonstrates that ECG-AIM is hardware-agnostic: whether a patient wears a smartwatch or an adhesive patch, the model accurately recovers the complementary 11 leads without retraining.")

    # =========================================================================
    # Slide 12: In-Clinic Escalation Gallery: 1-Lead Smartwatch vs. 3-Lead Diagnostic Triad
    # =========================================================================
    s12 = prs.slides.add_slide(blank_layout)
    add_header(s12, "In-Clinic Escalation Gallery: 1-Lead Smartwatch vs. 3-Lead Diagnostic Triad", "Unseen Benchmark Tracing: PTB-XL Record #10283 Across Clinical Modalities")
    add_footer(s12, 12)

    # Left Panel: 1-Lead Ambulatory
    c1_12 = s12.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.7), Inches(5.7), Inches(4.7))
    c1_12.fill.solid()
    c1_12.fill.fore_color.rgb = LIGHT_BG
    c1_12.line.color.rgb = NAVY
    c1_12.line.width = Pt(1.5)

    tb1_t = s12.shapes.add_textbox(Inches(1.0), Inches(1.75), Inches(5.3), Inches(0.35))
    tf1_t = tb1_t.text_frame
    p = tf1_t.paragraphs[0]
    p.text = "Ambulatory: Smartwatch Lead I"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = NAVY

    if os.path.exists(fig_l1_path):
        s12.shapes.add_picture(fig_l1_path, Inches(0.95), Inches(2.15), Inches(5.4), Inches(2.8))

    tb1_b = s12.shapes.add_textbox(Inches(1.0), Inches(5.05), Inches(5.3), Inches(1.3))
    tf1_b = tb1_b.text_frame
    tf1_b.word_wrap = True
    items_12_l = [
        "Input: Single physical smartwatch vector (Lead I).",
        "Inferred: 11 missing leads (Record mean r = 0.941).",
        "Clinical Role: Ambulatory home surveillance for AF recurrence.",
        "Limitation: Unresolvable planar dipole ambiguity on deep complexes."
    ]
    for idx, it in enumerate(items_12_l):
        p = tf1_b.paragraphs[0] if idx == 0 else tf1_b.add_paragraph()
        p.text = f"- {it}"
        p.font.size = Pt(10)
        p.font.color.rgb = SLATE_DARK
        p.space_before = Pt(2)

    # Right Panel: 3-Lead Diagnostic Triad
    c2_12 = s12.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.7), Inches(5.7), Inches(4.7))
    c2_12.fill.solid()
    c2_12.fill.fore_color.rgb = LIGHT_BG
    c2_12.line.color.rgb = CYAN
    c2_12.line.width = Pt(1.5)

    tb2_t = s12.shapes.add_textbox(Inches(7.0), Inches(1.75), Inches(5.3), Inches(0.35))
    tf2_t = tb2_t.text_frame
    p = tf2_t.paragraphs[0]
    p.text = "In-Clinic: Diagnostic Triad (I, II, V2)"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = CYAN

    fig_3l_ptb_path = "slides/figures/best_ecg_aim_12lead_reconstruction.png"
    if os.path.exists(fig_3l_ptb_path):
        s12.shapes.add_picture(fig_3l_ptb_path, Inches(6.95), Inches(2.15), Inches(5.4), Inches(2.8))

    tb2_b = s12.shapes.add_textbox(Inches(7.0), Inches(5.05), Inches(5.3), Inches(1.3))
    tf2_b = tb2_b.text_frame
    tf2_b.word_wrap = True
    items_12_r = [
        "Input: 3 physical stickers (Orthogonal Triad: I, II, V2).",
        "Inferred: 9 missing leads (r = 0.980 - 1.000 across all leads).",
        "Diagnostic Grade: Limb leads r = 1.000, QRS MAE = 8.12 ms.",
        "Clinical Role: Acute emergency triage & clinic follow-up."
    ]
    for idx, it in enumerate(items_12_r):
        p = tf2_b.paragraphs[0] if idx == 0 else tf2_b.add_paragraph()
        p.text = f"- {it}"
        p.font.size = Pt(10)
        p.font.color.rgb = SLATE_DARK
        p.space_before = Pt(2)

    # Bottom Banner
    card_bot12 = s12.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(6.5), Inches(11.7), Inches(0.42))
    card_bot12.fill.solid()
    card_bot12.fill.fore_color.rgb = LIGHT_BG
    card_bot12.line.color.rgb = NAVY
    card_bot12.line.width = Pt(1.5)

    tb_b12 = s12.shapes.add_textbox(Inches(1.0), Inches(6.52), Inches(11.3), Inches(0.38))
    tf_b12 = tb_b12.text_frame
    p = tf_b12.paragraphs[0]
    p.text = "Biophysical Triad Anchoring: Adding 2 clinical electrodes (V2 chest + II leg) locks both planes, transforming smartwatch screening into diagnostic 12-lead reconstruction."
    p.font.size = Pt(10.5)
    p.font.bold = True
    p.font.color.rgb = NAVY

    set_notes(s12, "Now look at how our model scales from at-home ambulatory screening to in-clinic diagnostic triage. On the left is the at-home smartwatch single-lead input with a mean correlation of 0.941. It is more than adequate for outpatient rhythm surveillance, but a single lead has an inherent mathematical limitation in resolving orthogonal planes simultaneously. On the right, when the patient arrives at Sunnybrook for routine follow-up, we place just two additional stickers—Lead II on the leg and V2 on the chest. Instantly, the model locks both the frontal and horizontal planes: Einthoven completion achieves a perfect 1.000 across all limb leads, and the precordial chest leads reach correlations between 0.980 and 0.999. Notice the terminal QRS notch and delicate J-point elevation in V1 and V3—they are reproduced with near-lossless diagnostic accuracy, reducing QRS duration error to 8.12 ms.")

    # =========================================================================
    # Slide 13: External Validation: Sunnybrook Physical 10-Wire XMLs
    # =========================================================================
    s13 = prs.slides.add_slide(blank_layout)
    add_header(s13, "External Validation: Sunnybrook Physical 10-Wire XMLs", "Zero-Shot Generalization on Real In-Clinic Data: Record ECG004 Across Modalities")
    add_footer(s13, 13)

    # Left Panel: Sunnybrook 1-Lead
    c1_13 = s13.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.7), Inches(5.7), Inches(4.7))
    c1_13.fill.solid()
    c1_13.fill.fore_color.rgb = LIGHT_BG
    c1_13.line.color.rgb = NAVY
    c1_13.line.width = Pt(1.5)

    tb1_t = s13.shapes.add_textbox(Inches(1.0), Inches(1.75), Inches(5.3), Inches(0.35))
    tf1_t = tb1_t.text_frame
    p = tf1_t.paragraphs[0]
    p.text = "Sunnybrook 1-Lead (Lead I)"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = NAVY

    fig_sb1_path = "slides/figures/sunnybrook_best_reconstruction_1lead.png"
    if os.path.exists(fig_sb1_path):
        s13.shapes.add_picture(fig_sb1_path, Inches(0.95), Inches(2.15), Inches(5.4), Inches(2.8))

    tb1_b = s13.shapes.add_textbox(Inches(1.0), Inches(5.05), Inches(5.3), Inches(1.3))
    tf1_b = tb1_b.text_frame
    tf1_b.word_wrap = True
    items_sb1 = [
        "Input: Single wrist channel (Einthoven Lead I).",
        "Inferred: 11 missing leads (Record mean r = 0.865).",
        "High-Fidelity Leads: aVR (r=0.966), V5 (r=0.934), V6 (r=0.929), II (r=0.925).",
        "Zero Recalibration: Tested zero-shot directly on Philips XMLs."
    ]
    for idx, it in enumerate(items_sb1):
        p = tf1_b.paragraphs[0] if idx == 0 else tf1_b.add_paragraph()
        p.text = f"- {it}"
        p.font.size = Pt(10)
        p.font.color.rgb = SLATE_DARK
        p.space_before = Pt(2)

    # Right Panel: Sunnybrook 3-Lead
    c2_13 = s13.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.7), Inches(5.7), Inches(4.7))
    c2_13.fill.solid()
    c2_13.fill.fore_color.rgb = LIGHT_BG
    c2_13.line.color.rgb = CYAN
    c2_13.line.width = Pt(1.5)

    tb2_t = s13.shapes.add_textbox(Inches(7.0), Inches(1.75), Inches(5.3), Inches(0.35))
    tf2_t = tb2_t.text_frame
    p = tf2_t.paragraphs[0]
    p.text = "Sunnybrook 3-Lead (I, II, V2)"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = CYAN

    fig_sb3_path = "slides/figures/sunnybrook_best_reconstruction_3lead.png"
    if os.path.exists(fig_sb3_path):
        s13.shapes.add_picture(fig_sb3_path, Inches(6.95), Inches(2.15), Inches(5.4), Inches(2.8))

    tb2_b = s13.shapes.add_textbox(Inches(7.0), Inches(5.05), Inches(5.3), Inches(1.3))
    tf2_b = tb2_b.text_frame
    tf2_b.word_wrap = True
    items_sb3 = [
        "Input: Physical triad (Leads I, II, V2).",
        "Inferred: 9 missing leads (Record mean r = 0.938).",
        "Limb Completion: r = 1.000; Precordials: V1 (r=0.973), V6 (r=0.957), V5 (r=0.942).",
        "Hardware Robustness: Preserves authentic 18 uV biological noise floor."
    ]
    for idx, it in enumerate(items_sb3):
        p = tf2_b.paragraphs[0] if idx == 0 else tf2_b.add_paragraph()
        p.text = f"- {it}"
        p.font.size = Pt(10)
        p.font.color.rgb = SLATE_DARK
        p.space_before = Pt(2)

    # Bottom Banner
    card_bot13 = s13.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(6.5), Inches(11.7), Inches(0.42))
    card_bot13.fill.solid()
    card_bot13.fill.fore_color.rgb = LIGHT_BG
    card_bot13.line.color.rgb = NAVY
    card_bot13.line.width = Pt(1.5)

    tb_b13 = s13.shapes.add_textbox(Inches(1.0), Inches(6.52), Inches(11.3), Inches(0.38))
    tf_b13 = tb_b13.text_frame
    p = tf_b13.paragraphs[0]
    p.text = "Hospital Hardware Transfer: Evaluated zero-shot on Dr. Cheung's own institution XMLs (ECG004: peak QRS > 2.1 mV in V2, > 1.4 mV in II). Preserves un-derived physical limb leads and authentic biological noise (~18 uV) with zero recalibration."
    p.font.size = Pt(10)
    p.font.bold = True
    p.font.color.rgb = NAVY

    set_notes(s13, "Chris, this slide is particularly meaningful because it is evaluated zero-shot on your own institution's physical ECG data—the Sunnybrook 10-wire XML recordings. We selected diagnostic record ECG004, which displays robust clinical voltages—over 1.4 mV in Lead II and 2.1 mV in V2. On the left, we provide only smartwatch Lead I to reconstruct the other 11 leads. Across real physical hospital hardware, with un-derived limb leads and natural biological noise, the model achieves a record mean correlation of 0.865, with limb Lead II at 0.925, aVR at 0.966, and lateral chest leads V5 and V6 reaching 0.934 and 0.929. On the right, look at the 3-lead physical triad. When we give Leads I, II, and V2, the mean correlation jumps to 0.938! All limb leads reach 1.000, anterior chest lead V1 reaches 0.973, and lateral leads reach 0.957. Crucially, unlike standard synthetic baselines that attenuate small-amplitude signals, ECG-AIM preserves authentic physical voltage scaling and sharp QRS deflections.")

    # =========================================================================
    # Slide 14: External Validation: Stanford EchoNext Structural Cohort
    # =========================================================================
    s14 = prs.slides.add_slide(blank_layout)
    add_header(s14, "External Validation: Stanford EchoNext Structural Cohort", "Zero-Shot Generalization on Paired Echo Waveforms: Record #10 Across Modalities")
    add_footer(s14, 14)

    # Left Panel: EchoNext 1-Lead
    c1_14 = s14.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.7), Inches(5.7), Inches(4.7))
    c1_14.fill.solid()
    c1_14.fill.fore_color.rgb = LIGHT_BG
    c1_14.line.color.rgb = NAVY
    c1_14.line.width = Pt(1.5)

    tb1_t = s14.shapes.add_textbox(Inches(1.0), Inches(1.75), Inches(5.3), Inches(0.35))
    tf1_t = tb1_t.text_frame
    p = tf1_t.paragraphs[0]
    p.text = "EchoNext 1-Lead (Lead I)"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = NAVY

    fig_en1_path = "slides/figures/echonext_best_reconstruction_1lead.png"
    if os.path.exists(fig_en1_path):
        s14.shapes.add_picture(fig_en1_path, Inches(0.95), Inches(2.15), Inches(5.4), Inches(2.8))

    tb1_b = s14.shapes.add_textbox(Inches(1.0), Inches(5.05), Inches(5.3), Inches(1.3))
    tf1_b = tb1_b.text_frame
    tf1_b.word_wrap = True
    items_en1 = [
        "Input: Single wrist channel (Lead I smartwatch vector).",
        "Inferred: 11 missing leads (Record mean r = 0.809).",
        "High-Fidelity Leads: aVR (r=0.967), aVL (r=0.946), V1 (r=0.942), V5 (r=0.915).",
        "Diagnostic Voltage: Normal cardiac voltages (0.9 - 1.4 mV) preserved."
    ]
    for idx, it in enumerate(items_en1):
        p = tf1_b.paragraphs[0] if idx == 0 else tf1_b.add_paragraph()
        p.text = f"- {it}"
        p.font.size = Pt(10)
        p.font.color.rgb = SLATE_DARK
        p.space_before = Pt(2)

    # Right Panel: EchoNext 3-Lead
    c2_14 = s14.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.7), Inches(5.7), Inches(4.7))
    c2_14.fill.solid()
    c2_14.fill.fore_color.rgb = LIGHT_BG
    c2_14.line.color.rgb = CYAN
    c2_14.line.width = Pt(1.5)

    tb2_t = s14.shapes.add_textbox(Inches(7.0), Inches(1.75), Inches(5.3), Inches(0.35))
    tf2_t = tb2_t.text_frame
    p = tf2_t.paragraphs[0]
    p.text = "EchoNext 3-Lead (I, II, V2)"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = CYAN

    fig_en3_path = "slides/figures/echonext_best_reconstruction_3lead.png"
    if os.path.exists(fig_en3_path):
        s14.shapes.add_picture(fig_en3_path, Inches(6.95), Inches(2.15), Inches(5.4), Inches(2.8))

    tb2_b = s14.shapes.add_textbox(Inches(7.0), Inches(5.05), Inches(5.3), Inches(1.3))
    tf2_b = tb2_b.text_frame
    tf2_b.word_wrap = True
    items_en3 = [
        "Input: Orthogonal physical triad (Leads I, II, V2).",
        "Inferred: 9 missing leads (Record mean r = 0.964).",
        "Limb Completion: r = 0.997 - 1.000; Precordials: V1 (r=0.987), V6 (r=0.965), V5 (r=0.962).",
        "Structural Concordance: 81.4% concordance on reduced LVEF <= 45%."
    ]
    for idx, it in enumerate(items_en3):
        p = tf2_b.paragraphs[0] if idx == 0 else tf2_b.add_paragraph()
        p.text = f"- {it}"
        p.font.size = Pt(10)
        p.font.color.rgb = SLATE_DARK
        p.space_before = Pt(2)

    # Bottom Banner
    card_bot14 = s14.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(6.5), Inches(11.7), Inches(0.42))
    card_bot14.fill.solid()
    card_bot14.fill.fore_color.rgb = LIGHT_BG
    card_bot14.line.color.rgb = NAVY
    card_bot14.line.width = Pt(1.5)

    tb_b14 = s14.shapes.add_textbox(Inches(1.0), Inches(6.52), Inches(11.3), Inches(0.38))
    tf_b14 = tb_b14.text_frame
    p = tf_b14.paragraphs[0]
    p.text = "Structural Phenotype Integrity: Preserves deep ventricular morphology across 100k echo-paired patients (Record #10: diagnostic voltages 0.9 - 1.4 mV; LVEF <= 45% concordance 81.4%), completely preventing Stanford 3DRECON's catastrophic collapse (33.3%)."
    p.font.size = Pt(10)
    p.font.bold = True
    p.font.color.rgb = NAVY

    set_notes(s14, "Alex and Chris, this slide shows our external validation on the massive Stanford EchoNext cohort—100,000 paired echocardiogram-ECG patient tracings. Here we examine diagnostic record #10, featuring standard normal cardiac voltages up to 1.40 mV in V4 and 1.15 mV in V2. On the left, 1-lead smartwatch reconstruction achieves a record mean correlation of 0.809, with aVR at 0.967 and chest leads V1 and V5 at 0.942 and 0.915. On the right, adding the 3-lead triad pushes the mean correlation to 0.964, with near-perfect limb tracking (aVR 1.000, aVF 0.997, III 0.986) and precordial tracking reaching 0.987 on V1 and 0.965 on V6. Why does this matter clinically? Because preserving the ST-T wave morphology and true QRS amplitude across these leads is exactly what allows downstream classifiers to predict reduced ejection fraction with 81.4% concordance. In contrast, Ansari's Stanford model suffered complete structural collapse down to 33.3% on this exact cohort because their autoencoder smoothed away the crucial high-frequency ventricular details.")

    # =========================================================================
    # Slide 15: Data Collection Methodology: Dual-Stream Protocol
    # =========================================================================
    s15 = prs.slides.add_slide(blank_layout)
    add_header(s15, "Data Collection Methodology: Dual-Stream Protocol", "Seamless Integration: Daily At-Home Surveillance + In-Clinic Paired Anchors")
    add_footer(s15, 15)

    c1 = s15.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.8), Inches(5.7), Inches(4.8))
    c1.fill.solid()
    c1.fill.fore_color.rgb = LIGHT_BG
    c1.line.color.rgb = NAVY
    c1.line.width = Pt(1.5)

    tb = s15.shapes.add_textbox(Inches(1.1), Inches(2.0), Inches(5.1), Inches(4.3))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Stream A: Ambulatory Home Surveillance"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = NAVY

    items_s15_1 = [
        "At-Home Daily ECG: Patient records daily 30s smartwatch ECG (Lead I) for 6 months post-ablation.",
        "Continuous Gold Standard: 14-day continuous CardioSTAT patch worn at 3 months and 6 months post-procedure.",
        "Primary Outcome Harvested: Exact time to recurrent AF, cumulative daily AF burden (% time in AF), and asymptomatic episode yield.",
        "Data Pipeline: Automated upload via SRI custom dashboard directly onto secure Sunnybrook servers behind the hospital firewall."
    ]
    for it in items_s15_1:
        p = tf.add_paragraph()
        p.text = f"- {it}"
        p.font.size = Pt(12)
        p.font.color.rgb = SLATE_DARK
        p.space_before = Pt(8)

    c2 = s15.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.8))
    c2.fill.solid()
    c2.fill.fore_color.rgb = LIGHT_BG
    c2.line.color.rgb = CYAN
    c2.line.width = Pt(1.5)

    tb2 = s15.shapes.add_textbox(Inches(7.1), Inches(2.0), Inches(5.1), Inches(4.3))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    p = tf2.paragraphs[0]
    p.text = "Stream B: In-Clinic Paired Anchor Protocol"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = CYAN

    items_s15_2 = [
        "Hospital Touchpoints: Routine visits already mandated in clinical care:",
        "  * Visit 1: Baseline pre-ablation / pre-cardioversion workup",
        "  * Visit 2: Index procedure day (post-ablation acute restoration)",
        "  * Visit 3: 3-month routine follow-up clinic visit",
        "  * Visit 4: 6-month study closeout clinic visit",
        "Synchronous Acquisition: Clinical 12-lead ECG is recorded simultaneously with smartwatch Lead I (delta t = 0 min).",
        "Patient Burden: Zero extra hospital visits, zero additional blood draws, adds <= 5 minutes per routine clinic encounter."
    ]
    for it in items_s15_2:
        p = tf2.add_paragraph()
        p.text = f"- {it}"
        p.font.size = Pt(11.5)
        p.font.color.rgb = SLATE_DARK
        p.space_before = Pt(6)

    set_notes(s15, "Chris and Alex, this is the exact data collection methodology. It operates as two synchronized streams: Stream A is the ambulatory home surveillance from Protocol v1.1. Stream B is the in-clinic paired protocol during routine hospital visits. Patients are already at Sunnybrook for four standard visits. When they get their standard 12-lead ECG, the research coordinator records a 30-second smartwatch tracing at the exact same moment. It adds less than 5 minutes to routine care, requires zero extra visits, and creates perfectly synchronized ground-truth pairs!")

    # =========================================================================
    # Slide 16: The Paired In-Clinic Evaluation Set: The Gold Standard
    # =========================================================================
    s16 = prs.slides.add_slide(blank_layout)
    add_header(s16, "The Paired In-Clinic Evaluation Set", "The Most Critical Dataset: Breaking the Synthetic Proxy Trap")
    add_footer(s16, 16)

    rows_eval = [
        ("Methodological Dimension", "Stanford 3DRECON-QT (Ansari 2026)", "Sunnybrook Wearable Trial (Our Evaluation Set)"),
        ("Hardware Evaluation Cohort", "Retrospective CareLink archive (N = 26 patients).", "Prospective cohort: 96 patients, ~384 paired recordings."),
        ("Temporal Synchronization", "+/- 7 days between ICM and 12L (|delta HR| <= 20 bpm).", "delta t = 0 synchronous: Captured in same clinical exam room."),
        ("Physiological Confounding", "Electrolytes, posture, autonomic tone vary over 7 days.", "Identical resting state: Supine, same autonomic state, same HR."),
        ("Clinical Transitions Captured", "Static cross-sectional outpatient snapshots.", "Longitudinal transitions: Baseline AF -> Acute ablation -> Healed."),
        ("Scientific Role", "Device proof-of-concept for publication.", "The Gold Standard Benchmark: Bridges synthetic AI to real wearables.")
    ]

    t_shape16 = s16.shapes.add_table(6, 3, Inches(0.8), Inches(1.8), Inches(11.7), Inches(3.8))
    table16 = t_shape16.table
    table16.columns[0].width = Inches(2.7)
    table16.columns[1].width = Inches(4.5)
    table16.columns[2].width = Inches(4.5)

    for r_idx, row in enumerate(rows_eval):
        for c_idx, val in enumerate(row):
            cell = table16.cell(r_idx, c_idx)
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

    card_eval = s16.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(5.8), Inches(11.7), Inches(1.0))
    card_eval.fill.solid()
    card_eval.fill.fore_color.rgb = LIGHT_BG
    card_eval.line.color.rgb = SUCCESS_GREEN
    card_eval.line.width = Pt(1.5)

    tb = s16.shapes.add_textbox(Inches(1.0), Inches(5.9), Inches(11.3), Inches(0.8))
    tf = tb.text_frame
    p = tf.paragraphs[0]
    p.text = "Why This Evaluation Set is the Crown Jewel: 99% of published ECG AI papers test only on synthetic derived leads from 12-lead files. That only tests math, not wearable physics (dry electrodes, wrist impedance, DSP filters). Our prospective paired dataset is the first true clinical-hardware benchmark for wearable 12-lead reconstruction."
    p.font.size = Pt(11.5)
    p.font.bold = True
    p.font.color.rgb = SUCCESS_GREEN

    set_notes(s16, "Alex, this slide explains why our computing science group needs this data so desperately: this evaluation set is the crown jewel of the entire project! Ansari and Rogers had to search retrospectively through old CareLink transmissions and only found 26 patients within +/- 7 days. In our trial, we are prospectively acquiring exact delta-t equals zero pairs across 96 patients at 4 distinct time points—yielding nearly 400 paired recordings! Furthermore, this evaluation set captures true longitudinal transitions: pre-ablation AF, acute sinus restoration, and 3-to-6-month healed rhythm. This completely breaks the synthetic proxy trap that invalidates other AI papers.")

    # =========================================================================
    # Slide 17: Figure 3: Integrated Clinical-AI Trial Pipeline
    # =========================================================================
    s17 = prs.slides.add_slide(blank_layout)
    add_header(s17, "Figure 3: Integrated Clinical-AI Trial Pipeline", "Tri-Modal Data Acquisition Engine, Domain Adaptation, and Dual Validation")
    add_footer(s17, 17)

    fig3_path = "slides/figures/paired_trial_model_integration.png"
    if os.path.exists(fig3_path):
        s17.shapes.add_picture(fig3_path, Inches(0.8), Inches(1.8), Inches(6.2), Inches(4.9))

    c_right = s17.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.3), Inches(1.8), Inches(5.2), Inches(4.9))
    c_right.fill.solid()
    c_right.fill.fore_color.rgb = LIGHT_BG
    c_right.line.color.rgb = BORDER_COLOR
    c_right.line.width = Pt(1)

    tb = s17.shapes.add_textbox(Inches(7.5), Inches(1.9), Inches(4.8), Inches(4.7))
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

    set_notes(s17, "Alex and Chris, this is Figure 3: the master architecture diagram showing how the clinical trial and AI engineering integrate seamlessly. Stage 1 is foundation models pretrained on big public data. Stage 2 is our tri-modal data acquisition engine. Stage 3 uses paired in-clinic data to calibrate the hardware transfer function and adapt the model. Stage 4 shows the clear separation of endpoints: primary clinical AF surveillance vs. secondary translational 12-lead reconstruction.")

    # =========================================================================
    # Slide 18: Sample Size, Statistical Power & Clustered Modeling
    # =========================================================================
    s18 = prs.slides.add_slide(blank_layout)
    add_header(s18, "Sample Size, Statistical Power & Clustered Modeling", "Precision-Driven Sizing for Clinical Trial and Clustered Secondary AI Aim")
    add_footer(s18, 18)

    c1 = s18.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.8), Inches(5.7), Inches(4.8))
    c1.fill.solid()
    c1.fill.fore_color.rgb = LIGHT_BG
    c1.line.color.rgb = NAVY
    c1.line.width = Pt(1.5)

    tb = s18.shapes.add_textbox(Inches(1.1), Inches(2.0), Inches(5.1), Inches(4.3))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Primary Clinical Power (Protocol v1.1)"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = NAVY

    items_s18_1 = [
        "Primary Outcome: Time to AF recurrence detection.",
        "Hypothesis: Smartwatch detects recurrent AF at higher or non-inferior rate vs. 14-day patch.",
        "Assumptions: AF recurrence rate 20% with patch vs. 25% with smartwatch.",
        "Target Power: 80% power, alpha = 0.05, 10% non-inferiority margin -> N = 96 patients.",
        "Vanguard Cohort: Initial N = 20 consecutive patients to verify protocol adherence and pipeline."
    ]
    for it in items_s18_1:
        p = tf.add_paragraph()
        p.text = f"- {it}"
        p.font.size = Pt(12)
        p.font.color.rgb = SLATE_DARK
        p.space_before = Pt(8)

    c2 = s18.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.8))
    c2.fill.solid()
    c2.fill.fore_color.rgb = LIGHT_BG
    c2.line.color.rgb = CYAN
    c2.line.width = Pt(1.5)

    tb2 = s18.shapes.add_textbox(Inches(7.1), Inches(2.0), Inches(5.1), Inches(4.3))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    p = tf2.paragraphs[0]
    p.text = "Secondary AI Precision & Clustered Stats"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = CYAN

    items_s18_2 = [
        "AI Sample Size: 96 patients x 4 in-clinic visits yields >2,500 paired cardiac complexes.",
        "Precision Target: Sized for QTc MAE 95% CI margin of error <= +/- 3.5 ms.",
        "Clustering Adjustment: Multiple beats per patient violate standard i.i.d. assumptions.",
        "MMRM & GEE: Linear Mixed Models (unstructured covariance) and Generalized Estimating Equations for robust sandwich variance.",
        "Regulatory Defensibility: Primary sample size is powered on clinical events; secondary AI sample size exceeds precision requirements."
    ]
    for it in items_s18_2:
        p = tf2.add_paragraph()
        p.text = f"- {it}"
        p.font.size = Pt(11.5)
        p.font.color.rgb = SLATE_DARK
        p.space_before = Pt(6)

    set_notes(s18, "Chris and Alex, this slide demonstrates total alignment on sample size and statistics. The primary clinical trial sample size of 96 patients is calculated directly from expected AF recurrence rates—20% on patch vs 25% on smartwatch with an 80% power target. For the secondary AI aim, having 96 patients with multiple paired in-clinic visits provides over 2,500 paired cardiac complexes. Per Riley's 2024 BMJ guidelines, this is more than sufficient to bound QTc measurement error within a tight 3.5 ms margin of error. Furthermore, we pre-specify Linear Mixed Models (MMRM) and Generalized Estimating Equations (GEE) to adjust for intra-patient beat clustering.")

    # =========================================================================
    # Slide 19: Action Items & Alignment for Today's Meeting
    # =========================================================================
    s19 = prs.slides.add_slide(blank_layout)
    add_header(s19, "Action Items & Alignment for Today's Meeting", "4 Concrete Decisions to Move Forward")
    add_footer(s19, 19)

    c1 = s19.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.8), Inches(6.0), Inches(4.8))
    c1.fill.solid()
    c1.fill.fore_color.rgb = LIGHT_BG
    c1.line.color.rgb = NAVY
    c1.line.width = Pt(1.5)

    tb = s19.shapes.add_textbox(Inches(1.1), Inches(2.0), Inches(5.4), Inches(4.3))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "4 Concrete Decisions to Finalize Today"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = NAVY

    items_s19 = [
        "1. Two-Track Protocol Amendment: Approve formalizing Aim 1 (AF recurrence) as primary and Aim 2 (paired 12L reconstruction) as secondary.",
        "2. In-Clinic Paired Recording SOP: Approve the 5-minute simultaneous smartwatch + 12L recording protocol during routine clinic visits.",
        "3. Secondary Biomarker Selection: Confirm Chris priority secondary biomarker: QTc monitoring on sotalol vs. P-wave remodeling vs. conduction delay.",
        "4. Vanguard Cohort Launch: Authorize rolling out the 20-patient feasibility vanguard under Sunnybrook IRB."
    ]
    for it in items_s19:
        p = tf.add_paragraph()
        p.text = it
        p.font.size = Pt(12)
        p.font.color.rgb = SLATE_DARK
        p.space_before = Pt(10)

    c2 = s19.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.1), Inches(1.8), Inches(5.4), Inches(4.8))
    c2.fill.solid()
    c2.fill.fore_color.rgb = LIGHT_BG
    c2.line.color.rgb = CYAN
    c2.line.width = Pt(1.5)

    tb2 = s19.shapes.add_textbox(Inches(7.4), Inches(2.0), Inches(4.8), Inches(4.3))
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

    set_notes(s19, "To close our meeting, here are the 4 concrete decisions we need to agree on: First, confirm the two-track protocol amendment: Aim 1 primary AF recurrence, Aim 2 secondary 12-lead reconstruction. Second, approve the simple 5-minute simultaneous recording SOP during in-clinic visits. Third, decide which secondary clinical biomarker Chris wants to prioritize for the paired analysis—QTc interval tracking is our recommendation. Fourth, green-light the 20-patient vanguard cohort. Thank you Chris and Alex; let us open the floor for discussion.")

    output_path = "slides/presentation.pptx"
    prs.save(output_path)
    print(f"Presentation saved successfully to {output_path} (Total slides: 19)")

if __name__ == "__main__":
    create_deck()
'''

    full_updated_code = prefix + new_slides_code
    with open("slides/generate_pptx.py", "w") as f:
        f.write(full_updated_code)
    print("Updated slides/generate_pptx.py successfully.")

if __name__ == "__main__":
    generate()
