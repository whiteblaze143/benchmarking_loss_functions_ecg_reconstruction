import os

code = r'''#!/usr/bin/env python3
"""
generate_pptx.py
Generates a polished 16:9 widescreen PowerPoint deck (12 slides) matching slides/main.tex
for the meeting with Dr. Christopher Cheung and Dr. Alex Mariakakis.
Protocol Reference: Evaluation of FitBit Monitoring to Detect Recurrent Atrial Fibrillation (v1.1)
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

def create_deck():
    prs = Presentation()
    # 16:9 widescreen dimensions
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6] # completely blank layout

    # Color definitions
    NAVY = RGBColor(0x1E, 0x3A, 0x8A)
    CYAN = RGBColor(0x02, 0x84, 0xC7)
    SLATE_DARK = RGBColor(0x0F, 0x17, 0x2A)
    SLATE_MUTED = RGBColor(0x64, 0x74, 0x8B)
    LIGHT_BG = RGBColor(0xF8, 0xFA, 0xFC)
    BORDER_COLOR = RGBColor(0xE2, 0xE8, 0xF0)
    ALERT_RED = RGBColor(0xB9, 0x1C, 0x1C)
    SUCCESS_GREEN = RGBColor(0x15, 0x80, 0x3D)
    WHITE = RGBColor(0xFF, 0xFF, 0xFF)

    def add_header(slide, title_text, subtitle_text):
        tb = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(1.1))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        
        p = tf.paragraphs[0]
        p.text = title_text
        p.font.size = Pt(24)
        p.font.bold = True
        p.font.color.rgb = NAVY
        p.font.name = "Arial"
        
        p2 = tf.add_paragraph()
        p2.text = subtitle_text
        p2.font.size = Pt(13)
        p2.font.color.rgb = SLATE_MUTED
        p2.font.name = "Arial"
        
        line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.5), Inches(11.7), Inches(0.02))
        line.fill.solid()
        line.fill.fore_color.rgb = BORDER_COLOR
        line.line.color.rgb = BORDER_COLOR

    def add_footer(slide, slide_num, total_slides=12):
        tb = slide.shapes.add_textbox(Inches(0.8), Inches(7.0), Inches(11.7), Inches(0.35))
        tf = tb.text_frame
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        p = tf.paragraphs[0]
        p.text = f"Sunnybrook Arrhythmia Trial Protocol  *  12-Lead AI Integration                                                                     {slide_num} / {total_slides}"
        p.font.size = Pt(10)
        p.font.color.rgb = SLATE_MUTED
        p.font.name = "Arial"

    def set_notes(slide, notes_text):
        notes_slide = slide.notes_slide
        tf = notes_slide.notes_text_frame
        tf.text = notes_text

    # =========================================================================
    # Slide 1: Title Slide
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    card = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.2), Inches(1.0), Inches(10.9), Inches(5.5))
    card.fill.solid()
    card.fill.fore_color.rgb = LIGHT_BG
    card.line.color.rgb = BORDER_COLOR
    card.line.width = Pt(1.5)

    tb = s1.shapes.add_textbox(Inches(1.8), Inches(1.5), Inches(9.7), Inches(4.5))
    tf = tb.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "Integrating 12-Lead AI Reconstruction into the Sunnybrook Wearable AF Trial"
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = NAVY
    p.font.name = "Arial"

    p2 = tf.add_paragraph()
    p2.text = "A Rigorous Two-Track Protocol Harmonizing Clinical Trial Efficacy and AI Discovery"
    p2.font.size = Pt(15)
    p2.font.color.rgb = CYAN
    p2.font.name = "Arial"
    p2.space_before = Pt(10)

    p3 = tf.add_paragraph()
    p3.text = "Sunnybrook Arrhythmia Clinic & University of Toronto Computer Science\nProtocol Reference: Evaluation of FitBit Monitoring to Detect Recurrent Atrial Fibrillation (v1.1)"
    p3.font.size = Pt(12)
    p3.font.color.rgb = SLATE_MUTED
    p3.font.name = "Arial"
    p3.space_before = Pt(24)

    p4 = tf.add_paragraph()
    p4.text = "Investigators: Christopher Cheung, MD * Alex Mariakakis, PhD * Mithun Manivannan, BSc * Robert Wu, MD * Alice Tu"
    p4.font.size = Pt(12)
    p4.font.bold = True
    p4.font.color.rgb = SLATE_DARK
    p4.font.name = "Arial"
    p4.space_before = Pt(20)

    add_footer(s1, 1)
    set_notes(s1, "Welcome Dr. Cheung and Dr. Mariakakis. Today we are presenting a unified research roadmap that bridges our Sunnybrook clinical trial protocol with state-of-the-art 12-lead reconstruction AI. Our core objective is to show how these two tracks strengthen each other while completely de-risking the clinical trial.")

    # =========================================================================
    # Slide 2: Purpose of Today's Meeting
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "Purpose of Today's Meeting", "Established Plan as Foundation: Cleanly Integrating 12-Lead Reconstruction")
    add_footer(s2, 2)

    cards_s2 = [
        ("1. The Established Plan", NAVY, [
            "We build upon Dr. Cheung's clinical protocol: Evaluation of FitBit Monitoring to Detect Recurrent Atrial Fibrillation (v1.1).",
            "Population: Persistent AF patients undergoing planned catheter ablation or cardioversion.",
            "Primary clinical endpoint: Time to recurrent AF detection using smartwatch vs. 14-day continuous CardioSTAT patch.",
            "This clinical foundation remains sovereign and fully preserved."
        ]),
        ("2. The Integration Question", CYAN, [
            "How do we cleanly integrate 12-lead AI reconstruction without risking the primary trial outcome?",
            "Reconstruction from single-lead wrist signals is an ill-posed mathematical inverse problem.",
            "Solution: Multi-task spatial regularization (inspired by Ansari et al., Circulation 2026).",
            "Secondary paired in-clinic evaluation creates the world's premier benchmark dataset."
        ]),
        ("3. Meeting Alignment Goal", SUCCESS_GREEN, [
            "Achieve total consensus across Cardiology and Computer Science on a unified two-track protocol.",
            "Approve the in-clinic paired acquisition SOP (adding <= 5 minutes to routine clinic visits).",
            "Confirm pre-specified secondary AI endpoints and sample size calculations.",
            "Green-light rolling out the 20-patient feasibility vanguard cohort."
        ])
    ]

    for i, (title, color, items) in enumerate(cards_s2):
        c = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8 + i*4.0), Inches(1.8), Inches(3.7), Inches(4.9))
        c.fill.solid()
        c.fill.fore_color.rgb = LIGHT_BG
        c.line.color.rgb = color
        c.line.width = Pt(1.5)

        tb = s2.shapes.add_textbox(Inches(0.95 + i*4.0), Inches(2.0), Inches(3.4), Inches(4.5))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(15)
        p.font.bold = True
        p.font.color.rgb = color

        for it in items:
            p = tf.add_paragraph()
            p.text = f"- {it}"
            p.font.size = Pt(11)
            p.font.color.rgb = SLATE_DARK
            p.space_before = Pt(8)

    set_notes(s2, "Let us establish our shared context right away. We are not proposing a brand new trial or asking anyone to pivot. We start directly from Protocol Version 1.1 written by Chris, Alex, and our team. Today's goal is to show how 12-lead reconstruction integrates cleanly as an auxiliary regularizer and secondary endpoint.")

    # =========================================================================
    # Slide 3: The Established Clinical Protocol (v1.1)
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "The Established Clinical Protocol (v1.1)", "Evaluation of Wearable Monitoring to Detect Recurrent Atrial Fibrillation")
    add_footer(s3, 3)

    c1 = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.8), Inches(5.7), Inches(4.8))
    c1.fill.solid()
    c1.fill.fore_color.rgb = LIGHT_BG
    c1.line.color.rgb = NAVY
    c1.line.width = Pt(1.5)

    tb = s3.shapes.add_textbox(Inches(1.1), Inches(2.0), Inches(5.1), Inches(4.3))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Trial Design & Patient Population"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = NAVY

    items_p1 = [
        "Design: Single-center prospective cohort at Sunnybrook Health Sciences Centre (Arrhythmia Clinic).",
        "Inclusion: Age >= 18 with persistent AF undergoing planned catheter ablation or cardioversion.",
        "Intervention: Smartwatch (FitBit Sense / wearable) daily 30s ECG + automated irregular rhythm notifications for 6 months.",
        "Comparator: Health Canada-approved 14-day continuous CardioSTAT patch at 3 and 6 months (analyzed by iCentia).",
        "Clinical Reality: This protocol is clinically validated, IRB-ready, and independently impactful."
    ]
    for it in items_p1:
        p = tf.add_paragraph()
        p.text = f"- {it}"
        p.font.size = Pt(12)
        p.font.color.rgb = SLATE_DARK
        p.space_before = Pt(8)

    c2 = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.8))
    c2.fill.solid()
    c2.fill.fore_color.rgb = LIGHT_BG
    c2.line.color.rgb = CYAN
    c2.line.width = Pt(1.5)

    tb2 = s3.shapes.add_textbox(Inches(7.1), Inches(2.0), Inches(5.1), Inches(4.3))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    p = tf2.paragraphs[0]
    p.text = "Primary Outcome & Statistical Sizing"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = CYAN

    items_p2 = [
        "Primary Outcome: Time to recurrent AF detection (smartwatch vs. conventional clinical monitoring).",
        "Secondary Clinical Endpoints: High/low HR alerts, resting HR, HRV, cumulative AF burden (% monitored time in AF).",
        "Sample Size: N = 96 patients (80% power, alpha = 0.05, 20% vs 25% detection rate, 10% non-inferiority margin).",
        "Vanguard Phase: Initial N = 20 consecutive patients to verify operational adherence and pipeline feasibility.",
        "Data Pipeline: Raw watch ECGs harvested via SRI custom dashboard directly behind Sunnybrook firewall."
    ]
    for it in items_p2:
        p = tf2.add_paragraph()
        p.text = f"- {it}"
        p.font.size = Pt(12)
        p.font.color.rgb = SLATE_DARK
        p.space_before = Pt(8)

    set_notes(s3, "Chris designed this study to evaluate whether daily smartwatch ECG recordings and photoplethysmography irregular rhythm notifications detect recurrent AF earlier than standard-of-care intermittent patch monitoring. Patients receive a 14-day CardioSTAT patch at 3 months and 6 months post-procedure. The sample size is 96 patients, starting with a 20-patient vanguard. This clinical core remains completely sovereign.")

    # =========================================================================
    # Slide 4: Why Reconstruction Cannot Be the Primary Endpoint
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "Why Reconstruction Cannot Be the Primary Endpoint", "Mathematical Ill-Posedness, Variance Compression, and Trial Risk")
    add_footer(s4, 4)

    rows = [
        ("Evaluation Dimension", "Primary Clinical Trial Standard", "Wearable 12-Lead Reconstruction Reality"),
        ("Mathematical Problem", "Well-defined time-to-event survival analysis.", "Highly underdetermined inverse problem (1 -> 12 projection)."),
        ("Empirical Performance", "Watch AF sensitivity ~ 94% (Abu-Alrub 2022).", "Precordial leads (V2-V4) suffer variance loss (r ~ 0.74, MAE ~ 0.13 mV)."),
        ("Literature Consensus", "Guideline-endorsed recurrence metrics (2024 EHRA/HRS).", "Presacan et al. (2025): Severe amplitude compression; pulls to mean."),
        ("Hardware Domain Gap", "Verified against 14-day CardioSTAT gold standard.", "Hu et al. (2026): Hospital Lead I != wrist watch (dry electrodes, DSP)."),
        ("Trial Execution Risk", "Low Risk: Clinical utility proven even if AI fails.", "Existential Risk: Staking primary gate on 12L fidelity guarantees failure.")
    ]

    t_shape = s4.shapes.add_table(6, 3, Inches(0.8), Inches(1.8), Inches(11.7), Inches(3.8))
    table = t_shape.table
    table.columns[0].width = Inches(2.7)
    table.columns[1].width = Inches(4.5)
    table.columns[2].width = Inches(4.5)

    for r_idx, row in enumerate(rows):
        for c_idx, val in enumerate(row):
            cell = table.cell(r_idx, c_idx)
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

    card_alert = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(5.8), Inches(11.7), Inches(1.0))
    card_alert.fill.solid()
    card_alert.fill.fore_color.rgb = LIGHT_BG
    card_alert.line.color.rgb = ALERT_RED
    card_alert.line.width = Pt(1.5)

    tb = s4.shapes.add_textbox(Inches(1.0), Inches(5.9), Inches(11.3), Inches(0.8))
    tf = tb.text_frame
    p = tf.paragraphs[0]
    p.text = "Methodological Takeaway: We cannot tell reviewers or Sunnybrook that wrist Lead I deterministically replaces an acute diagnostic 12-lead machine. Doing so would destroy trial credibility. Instead, we harvest prospective data for secondary translational discovery."
    p.font.size = Pt(11.5)
    p.font.bold = True
    p.font.color.rgb = ALERT_RED

    set_notes(s4, "Reviewers and clinical colleagues will rightly push back if anyone claims a smartwatch can replace a diagnostic 12-lead machine today. Electrophysiologically, reconstructing 12 leads from one lead is an underdetermined inverse problem with known variance compression. If a clinical trial staked its primary regulatory endpoint on 12-lead reconstruction accuracy, it would fail. However, that does not mean the data is useless—it means reconstruction belongs strictly as a secondary translational aim!")

    # =========================================================================
    # Slide 5: The Clean Integration: A Rigorous Two-Track Protocol
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "The Clean Integration: A Rigorous Two-Track Protocol", "Decoupling Clinical Trial Efficacy from Translational AI Discovery")
    add_footer(s5, 5)

    c1 = s5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.8), Inches(5.7), Inches(4.8))
    c1.fill.solid()
    c1.fill.fore_color.rgb = LIGHT_BG
    c1.line.color.rgb = NAVY
    c1.line.width = Pt(2)

    tb = s5.shapes.add_textbox(Inches(1.1), Inches(2.0), Inches(5.1), Inches(4.3))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Track 1: Primary Clinical Efficacy (Aim 1)"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = NAVY

    items_t1 = [
        "Research Question: Does longitudinal smartwatch surveillance improve yield and time-to-detection of post-ablation recurrent AF?",
        "Comparator: Standard-of-care 14-day continuous CardioSTAT patch at 3 and 6 months (analyzed by iCentia).",
        "Trial Gate: Powered on AF recurrence sensitivity (>= 90%) and non-inferiority (N = 96).",
        "Clinical Publication: Targets top cardiology journals (JACC, Circulation: Arrhythmia, Heart Rhythm).",
        "Status: Inviolable Primary Endpoint. Guarantees trial success and clinical impact."
    ]
    for it in items_t1:
        p = tf.add_paragraph()
        p.text = f"- {it}"
        p.font.size = Pt(12)
        p.font.color.rgb = SLATE_DARK
        p.space_before = Pt(8)

    c2 = s5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.8))
    c2.fill.solid()
    c2.fill.fore_color.rgb = LIGHT_BG
    c2.line.color.rgb = CYAN
    c2.line.width = Pt(2)

    tb2 = s5.shapes.add_textbox(Inches(7.1), Inches(2.0), Inches(5.1), Inches(4.3))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    p = tf2.paragraphs[0]
    p.text = "Track 2: Secondary Translational AI (Aims 2 & 3)"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = CYAN

    items_t2 = [
        "Aim 2 (Reconstruction Benchmarking): What 12-lead information is recoverable from wrist signals? Evaluated on simultaneous in-clinic paired recordings.",
        "Aim 3 (Longitudinal Prior Tracking): Does an in-clinic baseline 12L prior (Model C) enable tracking within-patient repolarization shifts (delta QTc)?",
        "Opportunity: Collects the world's premier prospectively paired wearable + 12L dataset.",
        "Safety: If precordial reconstruction has variance loss, the clinical trial remains completely unaffected.",
        "Status: Secondary / Exploratory. Advances computing science without risking trial endpoints."
    ]
    for it in items_t2:
        p = tf2.add_paragraph()
        p.text = f"- {it}"
        p.font.size = Pt(12)
        p.font.color.rgb = SLATE_DARK
        p.space_before = Pt(8)

    set_notes(s5, "This is the architecture that solves everything cleanly.\nTrack 1 is Chris's primary clinical trial: time to recurrent AF detection, backed by the 14-day CardioSTAT patch. This guarantees the study will produce high-impact clinical papers regardless of AI performance.\nTrack 2 is Alex's and our translational AI research: we pre-specify 12-lead reconstruction as a secondary translational aim. We use the in-clinic paired recordings to benchmark transfer functions, domain adaptation, and patient-specific priors.\nThis completely eliminates trial risk while maximizing scientific return.")

    # =========================================================================
    # Slide 6: The 3DRECON Methodology: Multitask Spatial Regularization
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "The 3DRECON Methodology (Ansari et al., Circulation 2026)", "How Auxiliary 12-Lead Reconstruction Powers Primary AF Detection")
    add_footer(s6, 6)

    c1 = s6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.8), Inches(5.7), Inches(4.8))
    c1.fill.solid()
    c1.fill.fore_color.rgb = LIGHT_BG
    c1.line.color.rgb = NAVY
    c1.line.width = Pt(1.5)

    tb = s6.shapes.add_textbox(Inches(1.1), Inches(2.0), Inches(5.1), Inches(4.3))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "The Stanford 3DRECON-QT Breakthrough"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = NAVY

    items_s6_1 = [
        "Reference: Ansari, Rogers et al., Circulation (Jan 2026 / Nov 2025).",
        "Multitask Learning Objective: Simultaneously reconstructed 12-lead ECG waveforms while predicting clinical biomarkers from single-lead input.",
        "Inductive Bias: Spatial lead-coordinate conditioning forced the shared encoder to learn 3D cardiac vector loops.",
        "Ablation Result: Removing spatial 12L reconstruction caused downstream clinical prediction to collapse (r = 0.73 -> 0.04)!"
    ]
    for it in items_s6_1:
        p = tf.add_paragraph()
        p.text = f"- {it}"
        p.font.size = Pt(12)
        p.font.color.rgb = SLATE_DARK
        p.space_before = Pt(8)

    c2 = s6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.8))
    c2.fill.solid()
    c2.fill.fore_color.rgb = LIGHT_BG
    c2.line.color.rgb = CYAN
    c2.line.width = Pt(1.5)

    tb2 = s6.shapes.add_textbox(Inches(7.1), Inches(2.0), Inches(5.1), Inches(4.3))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    p = tf2.paragraphs[0]
    p.text = "Application to Sunnybrook AF Trial (3DRECON-AF)"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = CYAN

    items_s6_2 = [
        "Primary Task (At Home): Single-lead smartwatch ECG detects AF recurrence & daily burden.",
        "The Single-Lead Blindness Problem: Lead I alone misses low-amplitude f-waves or ectopic triggers oriented along vertical/precordial axes.",
        "Auxiliary 12L Reconstruction Head: Forces shared encoder to represent full 3D atrial vectors (including V1 f-wave dynamics).",
        "Translational Win: Supercharges primary AF detection sensitivity while reconstructing 12 leads on the side!",
        "Key Mechanism: 12-lead reconstruction is an electrophysiologic regularizer preventing single-lead diagnostic failure."
    ]
    for it in items_s6_2:
        p = tf2.add_paragraph()
        p.text = f"- {it}"
        p.font.size = Pt(12)
        p.font.color.rgb = SLATE_DARK
        p.space_before = Pt(8)

    set_notes(s6, "Chris and Alex, this slide connects our project to the state-of-the-art literature published just two months ago in Circulation by Albert Rogers' group at Stanford (Ansari et al., 2026). Ansari developed 3DRECON-QT, showing that training an auxiliary 12-lead reconstruction head forces the single-lead encoder to learn 3D cardiac vector loops. In their ablation study, removing 12-lead spatial conditioning collapsed downstream clinical biomarker prediction from r=0.73 down to 0.04! We apply this exact paradigm: auxiliary 12L reconstruction regularizes the encoder so daily watch ECGs at home don't miss subtle AF recurrences!")

    # =========================================================================
    # Slide 7: Figure 1: The ECG-AIM Model Architecture
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "Figure 1: The ECG-AIM Model Architecture", "conv15e_A0_wave_noSSL_gated_add: 6 Core Engineering Components")
    add_footer(s7, 7)

    fig1_path = "slides/figures/conv15e_a0_wave_architecture.png"
    if os.path.exists(fig1_path):
        s7.shapes.add_picture(fig1_path, Inches(0.8), Inches(1.8), Inches(5.8), Inches(4.9))

    c_right = s7.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.9), Inches(1.8), Inches(5.6), Inches(4.9))
    c_right.fill.solid()
    c_right.fill.fore_color.rgb = LIGHT_BG
    c_right.line.color.rgb = BORDER_COLOR
    c_right.line.width = Pt(1)

    tb = s7.shapes.add_textbox(Inches(7.1), Inches(1.9), Inches(5.2), Inches(4.7))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "6 Core Architecture Modules"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = NAVY

    boxes = [
        ("1. Patch Tokenizer", "50ms non-overlapping temporal windows project single-lead wrist signals (1x5000) into tokens."),
        ("2. Continuous Morlet Wavelet Bank", "32 logarithmic scales (0.5-45 Hz) decompose raw signals into time-frequency scalograms."),
        ("3. TimeSformer + 1D-CNN", "Spatiotemporal self-attention across patches coupled with local convolutional receptive fields."),
        ("4. Gated-Add Fusion Module", "Sigmoid-gated adaptive combination: Z = Z_t + sigma(W[Z_t, Z_w]) * Z_w injects wavelets cleanly."),
        ("5. 8-Layer Transformer Encoder", "Captures long-range cardiac dependencies and rhythm transitions (d=768, 12 heads)."),
        ("6. Axial Transformer Decoder", "Decouples temporal and cross-lead spatial dimensions into continuous 12 leads and P-QRS-T delineation.")
    ]

    for title, desc in boxes:
        p = tf.add_paragraph()
        p.text = f"- {title}: {desc}"
        p.font.size = Pt(10.5)
        p.font.color.rgb = SLATE_DARK
        p.space_before = Pt(6)

    set_notes(s7, "Alex, this is the architectural engine that powers our system: ECG-AIM champion checkpoint conv15e_A0_wave_noSSL_gated_add. It combines continuous Morlet wavelets with a TimeSformer spatio-temporal encoder, gated-add fusion, and an axial decoder capable of predicting both 12-lead waveforms and explicit fiducials.")

    # =========================================================================
    # Slide 8: Data Collection Methodology: Dual-Stream Protocol
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "Data Collection Methodology: Dual-Stream Protocol", "Seamless Integration: Daily At-Home Surveillance + In-Clinic Paired Anchors")
    add_footer(s8, 8)

    c1 = s8.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.8), Inches(5.7), Inches(4.8))
    c1.fill.solid()
    c1.fill.fore_color.rgb = LIGHT_BG
    c1.line.color.rgb = NAVY
    c1.line.width = Pt(1.5)

    tb = s8.shapes.add_textbox(Inches(1.1), Inches(2.0), Inches(5.1), Inches(4.3))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Stream A: Ambulatory Home Surveillance"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = NAVY

    items_s8_1 = [
        "At-Home Daily ECG: Patient records daily 30s smartwatch ECG (Lead I) for 6 months post-ablation.",
        "Continuous Gold Standard: 14-day continuous CardioSTAT patch worn at 3 months and 6 months post-procedure.",
        "Primary Outcome Harvested: Exact time to recurrent AF, cumulative daily AF burden (% time in AF), and asymptomatic episode yield.",
        "Data Pipeline: Automated upload via SRI custom dashboard directly onto secure Sunnybrook servers behind the hospital firewall."
    ]
    for it in items_s8_1:
        p = tf.add_paragraph()
        p.text = f"- {it}"
        p.font.size = Pt(12)
        p.font.color.rgb = SLATE_DARK
        p.space_before = Pt(8)

    c2 = s8.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.8))
    c2.fill.solid()
    c2.fill.fore_color.rgb = LIGHT_BG
    c2.line.color.rgb = CYAN
    c2.line.width = Pt(1.5)

    tb2 = s8.shapes.add_textbox(Inches(7.1), Inches(2.0), Inches(5.1), Inches(4.3))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    p = tf2.paragraphs[0]
    p.text = "Stream B: In-Clinic Paired Anchor Protocol"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = CYAN

    items_s8_2 = [
        "Hospital Touchpoints: Routine visits already mandated in clinical care:",
        "  * Visit 1: Baseline pre-ablation / pre-cardioversion workup",
        "  * Visit 2: Index procedure day (post-ablation acute restoration)",
        "  * Visit 3: 3-month routine follow-up clinic visit",
        "  * Visit 4: 6-month study closeout clinic visit",
        "Synchronous Acquisition: Clinical 12-lead ECG is recorded simultaneously with smartwatch Lead I (delta t = 0 min).",
        "Patient Burden: Zero extra hospital visits, zero additional blood draws, adds <= 5 minutes per routine clinic encounter."
    ]
    for it in items_s8_2:
        p = tf2.add_paragraph()
        p.text = f"- {it}"
        p.font.size = Pt(11.5)
        p.font.color.rgb = SLATE_DARK
        p.space_before = Pt(6)

    set_notes(s8, "Chris and Alex, this is the exact data collection methodology. It operates as two synchronized streams: Stream A is the ambulatory home surveillance from Protocol v1.1. Stream B is the in-clinic paired protocol during routine hospital visits. Patients are already at Sunnybrook for four standard visits. When they get their standard 12-lead ECG, the research coordinator records a 30-second smartwatch tracing at the exact same moment. It adds less than 5 minutes to routine care, requires zero extra visits, and creates perfectly synchronized ground-truth pairs!")

    # =========================================================================
    # Slide 9: The Paired In-Clinic Evaluation Set: The Gold Standard
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "The Paired In-Clinic Evaluation Set", "The Most Critical Dataset: Breaking the Synthetic Proxy Trap")
    add_footer(s9, 9)

    rows_eval = [
        ("Methodological Dimension", "Stanford 3DRECON-QT (Ansari 2026)", "Sunnybrook Wearable Trial (Our Evaluation Set)"),
        ("Hardware Evaluation Cohort", "Retrospective CareLink archive (N = 26 patients).", "Prospective cohort: 96 patients, ~384 paired recordings."),
        ("Temporal Synchronization", "+/- 7 days between ICM and 12L (|delta HR| <= 20 bpm).", "delta t = 0 synchronous: Captured in same clinical exam room."),
        ("Physiological Confounding", "Electrolytes, posture, autonomic tone vary over 7 days.", "Identical resting state: Supine, same autonomic state, same HR."),
        ("Clinical Transitions Captured", "Static cross-sectional outpatient snapshots.", "Longitudinal transitions: Baseline AF -> Acute ablation -> Healed."),
        ("Scientific Role", "Device proof-of-concept for publication.", "The Gold Standard Benchmark: Bridges synthetic AI to real wearables.")
    ]

    t_shape = s9.shapes.add_table(6, 3, Inches(0.8), Inches(1.8), Inches(11.7), Inches(3.8))
    table = t_shape.table
    table.columns[0].width = Inches(2.7)
    table.columns[1].width = Inches(4.5)
    table.columns[2].width = Inches(4.5)

    for r_idx, row in enumerate(rows_eval):
        for c_idx, val in enumerate(row):
            cell = table.cell(r_idx, c_idx)
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

    card_eval = s9.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(5.8), Inches(11.7), Inches(1.0))
    card_eval.fill.solid()
    card_eval.fill.fore_color.rgb = LIGHT_BG
    card_eval.line.color.rgb = SUCCESS_GREEN
    card_eval.line.width = Pt(1.5)

    tb = s9.shapes.add_textbox(Inches(1.0), Inches(5.9), Inches(11.3), Inches(0.8))
    tf = tb.text_frame
    p = tf.paragraphs[0]
    p.text = "Why This Evaluation Set is the Crown Jewel: 99% of published ECG AI papers test only on synthetic derived leads from 12-lead files. That only tests math, not wearable physics (dry electrodes, wrist impedance, DSP filters). Our prospective paired dataset is the first true clinical-hardware benchmark for wearable 12-lead reconstruction."
    p.font.size = Pt(11.5)
    p.font.bold = True
    p.font.color.rgb = SUCCESS_GREEN

    set_notes(s9, "Alex, this slide explains why our computing science group needs this data so desperately: this evaluation set is the crown jewel of the entire project! Ansari and Rogers had to search retrospectively through old CareLink transmissions and only found 26 patients within +/- 7 days. In our trial, we are prospectively acquiring exact delta-t equals zero pairs across 96 patients at 4 distinct time points—yielding nearly 400 paired recordings! Furthermore, this evaluation set captures true longitudinal transitions: pre-ablation AF, acute sinus restoration, and 3-to-6-month healed rhythm. This completely breaks the synthetic proxy trap that invalidates other AI papers.")

    # =========================================================================
    # Slide 10: Figure 2: Integrated Clinical-AI Trial Pipeline
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    add_header(s10, "Figure 2: Integrated Clinical-AI Trial Pipeline", "Tri-Modal Data Acquisition Engine, Domain Adaptation, and Dual Validation")
    add_footer(s10, 10)

    fig2_path = "slides/figures/paired_trial_model_integration.png"
    if os.path.exists(fig2_path):
        s10.shapes.add_picture(fig2_path, Inches(0.8), Inches(1.8), Inches(6.2), Inches(4.9))

    c_right = s10.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.3), Inches(1.8), Inches(5.2), Inches(4.9))
    c_right.fill.solid()
    c_right.fill.fore_color.rgb = LIGHT_BG
    c_right.line.color.rgb = BORDER_COLOR
    c_right.line.width = Pt(1)

    tb = s10.shapes.add_textbox(Inches(7.5), Inches(1.9), Inches(4.8), Inches(4.7))
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

    set_notes(s10, "Alex and Chris, this is Figure 2: the master architecture diagram showing how the clinical trial and AI engineering integrate seamlessly. Stage 1 is foundation models pretrained on big public data. Stage 2 is our tri-modal data acquisition engine. Stage 3 uses paired in-clinic data to calibrate the hardware transfer function and adapt the model. Stage 4 shows the clear separation of endpoints: primary clinical AF surveillance vs. secondary translational 12-lead reconstruction.")

    # =========================================================================
    # Slide 11: Sample Size, Statistical Power & Clustered Modeling
    # =========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    add_header(s11, "Sample Size, Statistical Power & Clustered Modeling", "Precision-Driven Sizing for Clinical Trial and Clustered Secondary AI Aim")
    add_footer(s11, 11)

    c1 = s11.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.8), Inches(5.7), Inches(4.8))
    c1.fill.solid()
    c1.fill.fore_color.rgb = LIGHT_BG
    c1.line.color.rgb = NAVY
    c1.line.width = Pt(1.5)

    tb = s11.shapes.add_textbox(Inches(1.1), Inches(2.0), Inches(5.1), Inches(4.3))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Primary Clinical Power (Protocol v1.1)"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = NAVY

    items_s11_1 = [
        "Primary Outcome: Time to AF recurrence detection.",
        "Hypothesis: Smartwatch detects recurrent AF at higher or non-inferior rate vs. 14-day patch.",
        "Assumptions: AF recurrence rate 20% with patch vs. 25% with smartwatch.",
        "Target Power: 80% power, alpha = 0.05, 10% non-inferiority margin -> N = 96 patients.",
        "Vanguard Cohort: Initial N = 20 consecutive patients to verify protocol adherence and pipeline."
    ]
    for it in items_s11_1:
        p = tf.add_paragraph()
        p.text = f"- {it}"
        p.font.size = Pt(12)
        p.font.color.rgb = SLATE_DARK
        p.space_before = Pt(8)

    c2 = s11.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.8))
    c2.fill.solid()
    c2.fill.fore_color.rgb = LIGHT_BG
    c2.line.color.rgb = CYAN
    c2.line.width = Pt(1.5)

    tb2 = s11.shapes.add_textbox(Inches(7.1), Inches(2.0), Inches(5.1), Inches(4.3))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    p = tf2.paragraphs[0]
    p.text = "Secondary AI Precision & Clustered Stats"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = CYAN

    items_s11_2 = [
        "AI Sample Size: 96 patients x 4 in-clinic visits yields >2,500 paired cardiac complexes.",
        "Precision Target: Sized for QTc MAE 95% CI margin of error <= +/- 3.5 ms.",
        "Clustering Adjustment: Multiple beats per patient violate standard i.i.d. assumptions.",
        "MMRM & GEE: Linear Mixed Models (unstructured covariance) and Generalized Estimating Equations for robust sandwich variance.",
        "Regulatory Defensibility: Primary sample size is powered on clinical events; secondary AI sample size exceeds precision requirements."
    ]
    for it in items_s11_2:
        p = tf2.add_paragraph()
        p.text = f"- {it}"
        p.font.size = Pt(11.5)
        p.font.color.rgb = SLATE_DARK
        p.space_before = Pt(6)

    set_notes(s11, "Chris and Alex, this slide demonstrates total alignment on sample size and statistics. The primary clinical trial sample size of 96 patients is calculated directly from expected AF recurrence rates—20% on patch vs 25% on smartwatch with an 80% power target. For the secondary AI aim, having 96 patients with multiple paired in-clinic visits provides over 2,500 paired cardiac complexes. Per Riley's 2024 BMJ guidelines, this is more than sufficient to bound QTc measurement error within a tight 3.5 ms margin of error. Furthermore, we pre-specify Linear Mixed Models (MMRM) and Generalized Estimating Equations (GEE) to adjust for intra-patient beat clustering.")

    # =========================================================================
    # Slide 12: Action Items & Alignment for Today's Meeting
    # =========================================================================
    s12 = prs.slides.add_slide(blank_layout)
    add_header(s12, "Action Items & Alignment for Today's Meeting", "4 Concrete Decisions to Move Forward")
    add_footer(s12, 12)

    c1 = s12.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.8), Inches(6.0), Inches(4.8))
    c1.fill.solid()
    c1.fill.fore_color.rgb = LIGHT_BG
    c1.line.color.rgb = NAVY
    c1.line.width = Pt(1.5)

    tb = s12.shapes.add_textbox(Inches(1.1), Inches(2.0), Inches(5.4), Inches(4.3))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "4 Concrete Decisions to Finalize Today"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = NAVY

    items_s12 = [
        "1. Two-Track Protocol Amendment: Approve formalizing Aim 1 (AF recurrence) as primary and Aim 2 (paired 12L reconstruction) as secondary.",
        "2. In-Clinic Paired Recording SOP: Approve the 5-minute simultaneous smartwatch + 12L recording protocol during routine clinic visits.",
        "3. Secondary Biomarker Selection: Confirm Chris's priority secondary biomarker: QTc monitoring on sotalol vs. P-wave remodeling vs. conduction delay.",
        "4. Vanguard Cohort Launch: Authorize rolling out the 20-patient feasibility vanguard under Sunnybrook IRB."
    ]
    for it in items_s12:
        p = tf.add_paragraph()
        p.text = it
        p.font.size = Pt(12)
        p.font.color.rgb = SLATE_DARK
        p.space_before = Pt(10)

    c2 = s12.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.1), Inches(1.8), Inches(5.4), Inches(4.8))
    c2.fill.solid()
    c2.fill.fore_color.rgb = LIGHT_BG
    c2.line.color.rgb = CYAN
    c2.line.width = Pt(1.5)

    tb2 = s12.shapes.add_textbox(Inches(7.4), Inches(2.0), Inches(4.8), Inches(4.3))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    p = tf2.paragraphs[0]
    p.text = "Summary of Trial Benefits"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = CYAN

    items_ben = [
        "For Chris (Cardiology): High-impact clinical paper evaluating wearable AF surveillance against continuous patch monitoring.",
        "For Alex (Computer Science): World's first prospectively paired wearable + 12L dataset; top-tier machine learning publications.",
        "For Industry / Funders: A clinically rigorous, dual-value proposition with zero trial design flaws.",
        "Next Step: Finalize protocol text and begin vanguard enrollment in the Arrhythmia Clinic."
    ]
    for it in items_ben:
        p = tf2.add_paragraph()
        p.text = f"- {it}"
        p.font.size = Pt(12)
        p.font.color.rgb = SLATE_DARK
        p.space_before = Pt(10)

    set_notes(s12, "To close our meeting, here are the 4 concrete decisions we need to agree on: First, confirm the two-track protocol amendment: Aim 1 primary AF recurrence, Aim 2 secondary 12-lead reconstruction. Second, approve the simple 5-minute simultaneous recording SOP during in-clinic visits. Third, decide which secondary clinical biomarker Chris wants to prioritize for the paired analysis—QTc interval tracking is our recommendation. Fourth, green-light the 20-patient vanguard cohort. Thank you Chris and Alex; let's open the floor for discussion.")

    output_path = "slides/presentation.pptx"
    prs.save(output_path)
    print(f"Presentation saved successfully to {output_path} (Total slides: 12)")

if __name__ == "__main__":
    create_deck()
'''

with open("slides/generate_pptx.py", "w") as f:
    f.write(code)

print("Rewrote slides/generate_pptx.py successfully")
