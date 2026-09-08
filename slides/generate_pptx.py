#!/usr/bin/env python3
"""
generate_pptx.py
Generates a polished 16:9 widescreen PowerPoint deck (19 slides) matching slides/main.tex
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
    blank_layout = prs.slide_layouts[6]

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

    def add_footer(slide, slide_num, total_slides=19):
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

    tb = s1.shapes.add_textbox(Inches(1.8), Inches(1.6), Inches(9.7), Inches(4.3))
    tf = tb.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "Integrating 12-Lead AI Reconstruction into the Sunnybrook Wearable AF Trial"
    p.font.size = Pt(27)
    p.font.bold = True
    p.font.color.rgb = NAVY
    p.font.name = "Arial"

    p2 = tf.add_paragraph()
    p2.text = "Protocol Reference: Evaluation of Wearable/FitBit Monitoring to Detect Recurrent Atrial Fibrillation (v1.1)"
    p2.font.size = Pt(13)
    p2.font.color.rgb = SLATE_MUTED
    p2.font.name = "Arial"
    p2.space_before = Pt(16)

    p3 = tf.add_paragraph()
    p3.text = "Christopher Cheung, MD, MPH  *  Alex Mariakakis, PhD  *  Mithun Manivannan, BSc"
    p3.font.size = Pt(13)
    p3.font.bold = True
    p3.font.color.rgb = SLATE_DARK
    p3.font.name = "Arial"
    p3.space_before = Pt(20)

    p4 = tf.add_paragraph()
    p4.text = "Division of Cardiology, Sunnybrook Health Sciences Centre  *  Department of Computer Science, University of Toronto"
    p4.font.size = Pt(11.5)
    p4.font.color.rgb = SLATE_MUTED
    p4.font.name = "Arial"
    p4.space_before = Pt(6)

    add_footer(s1, 1)
    set_notes(s1, "Good morning Chris, good morning Alex. The purpose of today's meeting is straightforward: we already have an established clinical trial protocol—the Sunnybrook FitBit/Wearable ECG Study, authored by Chris, Alex, and our team. Our goal today is to establish how we can cleanly integrate our single-lead to 12-lead AI reconstruction research into this existing trial as a pre-specified secondary translational aim, without adding patient burden or endangering the primary clinical trial endpoints.")

    # =========================================================================
    # Slide 2: Purpose of Today's Meeting
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "Purpose of Today's Meeting", "Protocol alignment and study integration")
    add_footer(s2, 2)

    cards_s2 = [
        ("Clinical Trial Foundation", NAVY, [
            "Established protocol (Version 1.1, October 2025).",
            "Target cohort: persistent AF patients undergoing catheter ablation.",
            "Reference standard: 14-day continuous CardioSTAT patch at 3 and 6 months.",
            "Primary clinical endpoint: time to confirmed AF recurrence."
        ]),
        ("Reconstruction Constraints", CYAN, [
            "12-lead reconstruction from single-lead input is underdetermined.",
            "Precordial leads exhibit amplitude and morphology errors under direct projection.",
            "Diagnostic reconstruction cannot serve as a primary clinical endpoint.",
            "Separates clinical trial validity from exploratory AI evaluation."
        ]),
        ("Translational Integration", SUCCESS_GREEN, [
            "Routine outpatient visits already include scheduled 12-lead ECGs.",
            "Synchronous smartwatch and 12-lead acquisition adds <= 5 minutes.",
            "Evaluates reconstruction and biomarker recovery as a secondary aim.",
            "Generates a prospectively paired clinical and wearable dataset."
        ])
    ]

    for i, (title, color, items) in enumerate(cards_s2):
        c = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8 + i*4.0), Inches(1.8), Inches(3.7), Inches(4.5))
        c.fill.solid()
        c.fill.fore_color.rgb = LIGHT_BG
        c.line.color.rgb = color
        c.line.width = Pt(1.5)

        tb = s2.shapes.add_textbox(Inches(0.95 + i*4.0), Inches(2.0), Inches(3.4), Inches(4.1))
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

    tb_sub2 = s2.shapes.add_textbox(Inches(0.8), Inches(6.45), Inches(11.7), Inches(0.4))
    p_sub2 = tb_sub2.text_frame.paragraphs[0]
    p_sub2.text = "Objective: Align on the secondary data collection protocol and freeze translational evaluation metrics."
    p_sub2.alignment = PP_ALIGN.CENTER
    p_sub2.font.size = Pt(11.5)
    p_sub2.font.bold = True
    p_sub2.font.color.rgb = NAVY

    set_notes(s2, "Here is the layout for today. Box 1 is our foundation: Chris's clinical protocol is locked. It evaluates smartwatch monitoring versus continuous CardioSTAT patch for post-ablation AF recurrence. Box 2 is our methodological reality check: we know 12-lead reconstruction does not work with uniform diagnostic precision across all leads and patients; staking the trial on reconstruction would be scientifically unsound. Box 3 is our opportunity: during routine clinical visits, we can capture simultaneous paired watch and 12-lead recordings. This establishes a publishable secondary translational AI aim while protecting the primary trial.")

    # =========================================================================
    # Slide 3: The Established Clinical Protocol (v1.1)
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "The Established Clinical Protocol (v1.1)", "Protocol v1.1 design and endpoints")
    add_footer(s3, 3)

    c1 = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.8), Inches(5.7), Inches(4.5))
    c1.fill.solid()
    c1.fill.fore_color.rgb = LIGHT_BG
    c1.line.color.rgb = NAVY
    c1.line.width = Pt(1.5)

    tb = s3.shapes.add_textbox(Inches(1.1), Inches(2.0), Inches(5.1), Inches(4.1))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Trial Design & Patient Population"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = NAVY

    items_p1 = [
        "Single-center prospective cohort at Sunnybrook Arrhythmia Clinic.",
        "Patients aged >= 18 with persistent AF undergoing catheter ablation or cardioversion.",
        "Smartwatch monitoring: daily 30-second ECG and automated rhythm notifications for 6 months.",
        "Reference standard: 14-day continuous CardioSTAT patch at 3 and 6 months post-procedure."
    ]
    for it in items_p1:
        p = tf.add_paragraph()
        p.text = f"- {it}"
        p.font.size = Pt(12)
        p.font.color.rgb = SLATE_DARK
        p.space_before = Pt(8)

    c2 = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.5))
    c2.fill.solid()
    c2.fill.fore_color.rgb = LIGHT_BG
    c2.line.color.rgb = CYAN
    c2.line.width = Pt(1.5)

    tb2 = s3.shapes.add_textbox(Inches(7.1), Inches(2.0), Inches(5.1), Inches(4.1))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    p = tf2.paragraphs[0]
    p.text = "Endpoints & Sample Size"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = CYAN

    items_p2 = [
        "Primary outcome: time to recurrent AF detection (smartwatch vs. patch).",
        "Secondary clinical outcomes: heart rate trends, heart rate variability, and cumulative AF burden.",
        "Sample size: N = 96 patients (80% power, alpha = 0.05, 10% non-inferiority margin).",
        "Vanguard phase: initial 20 consecutive patients to assess protocol adherence and data pipeline."
    ]
    for it in items_p2:
        p = tf2.add_paragraph()
        p.text = f"- {it}"
        p.font.size = Pt(12)
        p.font.color.rgb = SLATE_DARK
        p.space_before = Pt(8)

    tb_sub3 = s3.shapes.add_textbox(Inches(0.8), Inches(6.45), Inches(11.7), Inches(0.4))
    p_sub3 = tb_sub3.text_frame.paragraphs[0]
    p_sub3.text = "Primary trial endpoints and clinical workflows remain unchanged under this proposal."
    p_sub3.alignment = PP_ALIGN.CENTER
    p_sub3.font.size = Pt(11.5)
    p_sub3.font.color.rgb = NAVY

    set_notes(s3, "Chris designed this study to evaluate whether daily smartwatch ECG recordings and photoplethysmography irregular rhythm notifications detect recurrent AF earlier than standard-of-care intermittent patch monitoring. Patients receive a 14-day CardioSTAT patch at 3 months and 6 months post-procedure. The sample size is 96 patients, starting with a 20-patient vanguard. This clinical core remains completely sovereign.")

    # =========================================================================
    # Slide 4: Why Reconstruction Cannot Be the Primary Endpoint
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "Why Reconstruction Cannot Be the Primary Endpoint", "Methodological constraints of single-lead reconstruction")
    add_footer(s4, 4)

    rows = [
        ("Evaluation Dimension", "Primary Clinical Trial Standard", "Single-Lead 12-Lead Reconstruction"),
        ("Mathematical Problem", "Time-to-event survival analysis", "Underdetermined inverse problem (1 -> 12 projection)"),
        ("Empirical Performance", "Wearable AF sensitivity ~ 94% (Abu-Alrub 2022)", "Precordial leads (V2-V4) exhibit variance loss (r ~ 0.74, MAE ~ 0.13 mV)"),
        ("Reported Literature", "Guideline-endorsed recurrence metrics (2024 EHRA/HRS)", "Presacan et al. (2025): amplitude compression and regression to mean"),
        ("Hardware Domain Gap", "Verified against 14-day CardioSTAT patch", "Hu et al. (2026): hospital limb leads differ from dry-electrode wrist signals"),
        ("Trial Execution Risk", "Low: clinical utility evaluable independently", "High: staking trial success on 12-lead fidelity introduces regulatory risk")
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
    p.text = "Methodological Summary: Single-lead reconstruction does not replace diagnostic 12-lead acquisition. Framing reconstruction as a primary endpoint introduces substantial clinical and regulatory risk, whereas a secondary translational aim leverages paired data without impacting trial governance."
    p.font.size = Pt(11.5)
    p.font.bold = True
    p.font.color.rgb = ALERT_RED

    set_notes(s4, "Reviewers and clinical colleagues will rightly push back if anyone claims a smartwatch can replace a diagnostic 12-lead machine today. Electrophysiologically, reconstructing 12 leads from one lead is an underdetermined inverse problem with known variance compression. If a clinical trial staked its primary regulatory endpoint on 12-lead reconstruction accuracy, it would introduce severe risk. However, that does not mean the data is useless—it means reconstruction belongs strictly as a secondary translational aim.")

    # =========================================================================
    # Slide 5: The Clean Integration: A Rigorous Two-Track Protocol
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "The Clean Integration: A Rigorous Two-Track Protocol", "Separation of primary clinical and secondary AI aims")
    add_footer(s5, 5)

    c1 = s5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.8), Inches(5.7), Inches(4.5))
    c1.fill.solid()
    c1.fill.fore_color.rgb = LIGHT_BG
    c1.line.color.rgb = NAVY
    c1.line.width = Pt(2)

    tb = s5.shapes.add_textbox(Inches(1.1), Inches(2.0), Inches(5.1), Inches(4.1))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Track 1: Primary Clinical Evaluation (Aim 1)"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = NAVY

    items_t1 = [
        "Evaluates whether smartwatch monitoring improves time to detection of recurrent AF.",
        "Reference standard: 14-day CardioSTAT patch at 3 and 6 months (iCentia analysis).",
        "Sized for AF recurrence detection sensitivity and non-inferiority (N = 96).",
        "Inviolable primary endpoint governing trial reporting."
    ]
    for it in items_t1:
        p = tf.add_paragraph()
        p.text = f"- {it}"
        p.font.size = Pt(12)
        p.font.color.rgb = SLATE_DARK
        p.space_before = Pt(8)

    c2 = s5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.5))
    c2.fill.solid()
    c2.fill.fore_color.rgb = LIGHT_BG
    c2.line.color.rgb = CYAN
    c2.line.width = Pt(2)

    tb2 = s5.shapes.add_textbox(Inches(7.1), Inches(2.0), Inches(5.1), Inches(4.1))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    p = tf2.paragraphs[0]
    p.text = "Track 2: Secondary Translational AI (Aims 2 & 3)"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = CYAN

    items_t2 = [
        "Aim 2: Benchmark 12-lead waveform recovery from wrist signals using paired clinic data.",
        "Aim 3: Evaluate within-patient repolarization tracking (delta QTc) using baseline priors.",
        "Secondary, exploratory endpoints with no effect on primary trial conclusions.",
        "Independent computational and methodological analyses."
    ]
    for it in items_t2:
        p = tf2.add_paragraph()
        p.text = f"- {it}"
        p.font.size = Pt(12)
        p.font.color.rgb = SLATE_DARK
        p.space_before = Pt(8)

    tb_sub5 = s5.shapes.add_textbox(Inches(0.8), Inches(6.45), Inches(11.7), Inches(0.4))
    p_sub5 = tb_sub5.text_frame.paragraphs[0]
    p_sub5.text = "Preserves primary clinical endpoints while establishing a prospectively paired validation dataset."
    p_sub5.alignment = PP_ALIGN.CENTER
    p_sub5.font.size = Pt(11.5)
    p_sub5.font.color.rgb = NAVY

    set_notes(s5, "This architecture cleanly separates the two efforts. Track 1 is Chris's primary clinical trial: time to recurrent AF detection, backed by the 14-day CardioSTAT patch. This guarantees the study will produce high-impact clinical papers regardless of AI performance. Track 2 is Alex's and our translational AI research: we pre-specify 12-lead reconstruction as a secondary translational aim. We use the in-clinic paired recordings to benchmark transfer functions, domain adaptation, and patient-specific priors. This protects trial integrity while maximizing scientific return.")

    # =========================================================================
    # Slide 6: The 3DRECON Methodology: Multitask Spatial Regularization
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "Multitask Auxiliary Supervision (Ansari et al., 2026)", "Spatial representation learning for single-lead models")
    add_footer(s6, 6)

    c1 = s6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.8), Inches(5.7), Inches(4.5))
    c1.fill.solid()
    c1.fill.fore_color.rgb = LIGHT_BG
    c1.line.color.rgb = NAVY
    c1.line.width = Pt(1.5)

    tb = s6.shapes.add_textbox(Inches(1.1), Inches(2.0), Inches(5.1), Inches(4.1))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Stanford 3DRECON-QT Framework"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = NAVY

    items_s6_1 = [
        "Reference: Ansari, Rogers et al., Circulation (January 2026).",
        "Multitask objective: simultaneously reconstruct 12 leads and estimate clinical biomarkers.",
        "Auxiliary 12-lead loss enforces 3D cardiac vector representations in the encoder.",
        "Removing spatial supervision reduced downstream biomarker correlation from r = 0.73 to 0.04."
    ]
    for it in items_s6_1:
        p = tf.add_paragraph()
        p.text = f"- {it}"
        p.font.size = Pt(12)
        p.font.color.rgb = SLATE_DARK
        p.space_before = Pt(8)

    c2 = s6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.5))
    c2.fill.solid()
    c2.fill.fore_color.rgb = LIGHT_BG
    c2.line.color.rgb = CYAN
    c2.line.width = Pt(1.5)

    tb2 = s6.shapes.add_textbox(Inches(7.1), Inches(2.0), Inches(5.1), Inches(4.1))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    p = tf2.paragraphs[0]
    p.text = "Application to Single-Lead AF Surveillance"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = CYAN

    items_s6_2 = [
        "Daily single-lead smartwatch ECG monitors for AF recurrence.",
        "Lead I captures frontal plane potentials, with limited sensitivity to orthogonal atrial vectors.",
        "Auxiliary 12-lead reconstruction head encourages representation of full 3D vector loops.",
        "Supports atrial feature extraction without altering primary single-lead inference inputs."
    ]
    for it in items_s6_2:
        p = tf2.add_paragraph()
        p.text = f"- {it}"
        p.font.size = Pt(12)
        p.font.color.rgb = SLATE_DARK
        p.space_before = Pt(8)

    tb_sub6 = s6.shapes.add_textbox(Inches(0.8), Inches(6.45), Inches(11.7), Inches(0.4))
    p_sub6 = tb_sub6.text_frame.paragraphs[0]
    p_sub6.text = "Auxiliary 12-lead reconstruction acts as an inductive regularizer during pretraining."
    p_sub6.alignment = PP_ALIGN.CENTER
    p_sub6.font.size = Pt(11.5)
    p_sub6.font.color.rgb = NAVY

    set_notes(s6, "This slide connects our project to recent literature published in Circulation by Albert Rogers' group at Stanford (Ansari et al., 2026). Ansari developed 3DRECON-QT, showing that training an auxiliary 12-lead reconstruction head forces the single-lead encoder to learn 3D cardiac vector loops. In their ablation study, removing the 12-lead spatial conditioning collapsed downstream clinical biomarker prediction from r=0.73 down to 0.04. We apply this paradigm to our trial: at home, patients record daily single-lead smartwatch ECGs for our primary endpoint: time to AF recurrence. Lead I alone often misses vector components oriented perpendicular to the frontal plane. Training an auxiliary 12-lead reconstruction head helps the encoder preserve 3D atrial vector dynamics.")

    # =========================================================================
    # Slide 7: Figure 1: The ECG-AIM Model Architecture
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "Figure 1: The ECG-AIM Model Architecture", "Network architecture and feature extraction")
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
    p.text = "Architecture Components"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = NAVY

    boxes = [
        "50 ms temporal patches project single-lead input (1x5000) into token embeddings.",
        "Continuous Morlet wavelet bank across 32 frequency scales (0.5-45 Hz).",
        "Combined TimeSformer spatiotemporal attention and 1D convolutions.",
        "Sigmoid-gated residual fusion: Z = Z_t + sigma(W[Z_t, Z_w]) * Z_w.",
        "8-layer transformer encoder (d=768, 12 attention heads).",
        "Axial transformer decoder resolving temporal and cross-lead spatial features."
    ]

    for desc in boxes:
        p = tf.add_paragraph()
        p.text = f"- {desc}"
        p.font.size = Pt(11)
        p.font.color.rgb = SLATE_DARK
        p.space_before = Pt(8)

    set_notes(s7, "This diagram shows the architecture of our champion model: conv15e_A0_wave_noSSL_gated_add. Notice the core components: 1. Patch tokenizer slicing the Lead I input into 50 ms tokens. 2. A continuous Morlet wavelet bank across 32 frequency scales from 0.5 to 45 Hz. 3. A hybrid TimeSformer and 1D-CNN backbone. 4. Our gated-add fusion module that dynamically modulates wavelet energy via a sigmoid gate. 5. An 8-layer transformer encoder. 6. An axial transformer decoder producing both continuous 12-lead waveforms and explicit fiducial delineations.")

    # =========================================================================
    # Slide 8: Figure 2: ECG-AIM Multi-Task Training Pipeline
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "Figure 2: ECG-AIM Multi-Task Training Pipeline", "Single-lead simulation and objective functions")
    add_footer(s8, 8)

    fig2_train_path = "slides/figures/ecg_aim_training_pipeline.png"
    if os.path.exists(fig2_train_path):
        s8.shapes.add_picture(fig2_train_path, Inches(2.05), Inches(1.7), Inches(9.23), Inches(5.15))

    set_notes(s8, "This figure illustrates how we trained the ECG-AIM model before testing on single-lead signals. First, on the left, we ingest full 12-lead hospital records from PTB-XL and MIMIC-IV. We simulate single-lead input by masking out the other 11 leads. Second, the single-lead signal passes through our dual-branch wavelet transformer backbone to predict continuous 12-lead waveforms and P-QRS-T boundary delineations. Third, looking at the loss formulation: we supervise waveform reconstruction with a composite loss (MSE, Pearson correlation, Wyatt ST-T phase loss, and spectral loss) plus Einthoven limb consistency, alongside multi-task cross-entropy and Dice loss on delineation. Importantly, during training, the single lead is a synthetic proxy derived from hospital ECGs on supine patients. It lacks dry-electrode contact noise and wearable filtering. This gap motivates the need for our Sunnybrook paired trial data to calibrate and validate the model on clinical hardware.")

    # =========================================================================
    # Slide 9: Empirical Validation of ECG-AIM: Benchmark Performance
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "Empirical Validation of ECG-AIM: Benchmark Performance", "Comparative benchmark against 3DRECON-QT")
    add_footer(s9, 9)

    rows_emp = [
        ("Clinical Evaluation Dimension", "Factorial Baseline", "Stanford 3DRECON-QT", "ECG-AIM (Ours)", "Statistical Contrast"),
        ("Precordial Chest Mean Pearson r", "0.570", "0.732", "0.776", "Delta = +0.206 (p < 10^-300), Ranked #3 of 55"),
        ("Anterior Lead V3 Pearson r", "0.382", "0.670", "0.739", "Delta = +0.357 (p < 10^-300), Anterior wall view"),
        ("QRS Duration Error (MAE / Median)", "13.85 ms", "10.26 ms", "10.05 / 8.0 ms", "Depolarization tracking"),
        ("Conduction Delay Concordance (>120ms)", "90.7%", "94.3%", "94.3%", "aOR = 1.84 (p < 10^-17), Specificity: 96.5%"),
        ("Sokolow-Lyon LVH Voltage MAE", "1.13 mV", "0.63 mV", "0.72 mV", "ICC = 0.714, Hypertrophy quantification"),
        ("EchoNext Structural Heart Disease", "68.8%", "46.5%", "71.6%", "Preserves chamber echocardiographic pathology")
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
                elif c_idx == 3:
                    p.font.bold = True
                    p.font.color.rgb = NAVY
                else:
                    p.font.color.rgb = SLATE_DARK

    # Left callout card
    c1_9 = s9.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(4.7), Inches(5.7), Inches(2.0))
    c1_9.fill.solid()
    c1_9.fill.fore_color.rgb = LIGHT_BG
    c1_9.line.color.rgb = NAVY
    c1_9.line.width = Pt(1.5)

    tb1_9 = s9.shapes.add_textbox(Inches(1.0), Inches(4.8), Inches(5.3), Inches(1.8))
    tf1_9 = tb1_9.text_frame
    tf1_9.word_wrap = True
    p = tf1_9.paragraphs[0]
    p.text = "Time-Frequency Feature Preservation"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = NAVY

    p_w1 = tf1_9.add_paragraph()
    p_w1.text = "Continuous Morlet wavelets preserve high-frequency QRS details and ST-T repolarization without baseline distortion, improving chest lead correlation over static CNN baselines (r = 0.776 vs. 0.732)."
    p_w1.font.size = Pt(10.5)
    p_w1.font.color.rgb = SLATE_DARK
    p_w1.space_before = Pt(4)

    # Right callout card
    c2_9 = s9.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(4.7), Inches(5.7), Inches(2.0))
    c2_9.fill.solid()
    c2_9.fill.fore_color.rgb = LIGHT_BG
    c2_9.line.color.rgb = NAVY
    c2_9.line.width = Pt(1.5)

    tb2_9 = s9.shapes.add_textbox(Inches(7.0), Inches(4.8), Inches(5.3), Inches(1.8))
    tf2_9 = tb2_9.text_frame
    tf2_9.word_wrap = True
    p = tf2_9.paragraphs[0]
    p.text = "Generalization to Structural Phenotypes"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = NAVY

    p_a1 = tf2_9.add_paragraph()
    p_a1.text = "On the EchoNext cohort, 3DRECON-QT demonstrated reduced structural concordance (46.5% overall, 33.3% on reduced LVEF). The gated wavelet transformer maintains 71.6% concordance across echocardiographic phenotypes."
    p_a1.font.size = Pt(10.5)
    p_a1.font.color.rgb = SLATE_DARK
    p_a1.space_before = Pt(4)

    set_notes(s9, "This slide presents the benchmark across models evaluated with patient-clustered MMRM and GEE. Compared to the Stanford 3DRECON-QT model, our model achieves higher precordial chest lead correlation—0.776 vs 0.732—and a Lead V3 correlation of 0.739 vs 0.670. On echocardiographic structural heart disease in EchoNext, Ansari's model dropped to 46.5% concordance because their scalar QT loss did not constrain chamber geometry. Our wavelet-gated transformer maintains 71.6% structural concordance and tracks QRS duration with a median error of 8.0 ms.")

    # =========================================================================
    # Slide 10: Zero-Shot Multi-Cohort Generalization & Flexible Lead Inputs
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    add_header(s10, "Zero-Shot Multi-Cohort Generalization & Flexible Lead Inputs", "Evaluation across variable lead configurations and external cohorts")
    add_footer(s10, 10)

    # Left Card
    c1_10 = s10.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.8), Inches(5.0), Inches(4.5))
    c1_10.fill.solid()
    c1_10.fill.fore_color.rgb = LIGHT_BG
    c1_10.line.color.rgb = NAVY
    c1_10.line.width = Pt(1.5)

    tb1_10 = s10.shapes.add_textbox(Inches(1.0), Inches(1.9), Inches(4.6), Inches(4.3))
    tf1_10 = tb1_10.text_frame
    tf1_10.word_wrap = True
    p = tf1_10.paragraphs[0]
    p.text = "Dynamic Lead Masking (M_obs)"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = NAVY

    items_s10_l = [
        "Stochastic lead masking during training enables arbitrary lead subsets at inference without retraining.",
        "Ambulatory screening (Lead I): PTB-XL AF AUROC: 0.952; Macro 150-task AUROC: 0.821.",
        "Clinical evaluation (Leads I, II, V2): Orthogonal vertical and horizontal leads resolve dipole ambiguity.",
        "Missing lead correlation increases to r = 0.892 - 0.945; QRS duration MAE reduced to 8.12 ms."
    ]
    for desc in items_s10_l:
        p = tf1_10.add_paragraph()
        p.text = f"- {desc}"
        p.font.size = Pt(11)
        p.font.color.rgb = SLATE_DARK
        p.space_before = Pt(8)

    # Right Table
    rows_s10 = [
        ("Cohort & Clinical Metric", "Stanford 3DRECON", "1-Lead ECG-AIM", "3-Lead ECG-AIM"),
        ("PTB-XL Foundation (N = 21,799)", "", "", ""),
        ("  ECGFounder 150-Task AUROC", "0.7720", "0.8208", "0.8409"),
        ("  AF Recurrence Detection AUROC", "0.9140", "0.9520", "0.9812"),
        ("  Ventricular Tachycardia AUROC", "0.8840", "0.9350", "0.9894"),
        ("  Conduction Delay (>120ms) Concord.", "94.3%", "94.6%", "96.8%"),
        ("  QRS Duration MAE", "10.26 ms", "10.05 ms", "8.12 ms"),
        ("EchoNext Structural (N = 100,000)", "", "", ""),
        ("  12 Deep Phenotypes Macro AUROC", "0.5890", "0.7374", "0.7542"),
        ("  Reduced LVEF (<=45%) Concord.", "33.3%", "77.7%", "81.4%"),
        ("  LV Hypertrophy (>=13mm) Concord.", "30.2%", "68.9%", "73.5%"),
        ("Sunnybrook 10-Wire XMLs (N = 20)", "", "", ""),
        ("  Physical Limb Leads", "Uncalibrated", "Preserved", "r = 0.938"),
        ("  Biological Noise Floor (~18 uV)", "Attenuated", "Preserved", "Preserved")
    ]

    t_shape10 = s10.shapes.add_table(14, 4, Inches(6.0), Inches(1.8), Inches(6.5), Inches(4.5))
    table10 = t_shape10.table
    table10.columns[0].width = Inches(2.7)
    table10.columns[1].width = Inches(1.25)
    table10.columns[2].width = Inches(1.25)
    table10.columns[3].width = Inches(1.3)

    for r_idx, row in enumerate(rows_s10):
        for c_idx, val in enumerate(row):
            cell = table10.cell(r_idx, c_idx)
            cell.text = val
            p = cell.text_frame.paragraphs[0]
            p.font.name = "Arial"
            if r_idx == 0:
                p.font.size = Pt(10)
                p.font.bold = True
                p.font.color.rgb = WHITE
                cell.fill.solid()
                cell.fill.fore_color.rgb = NAVY
            elif r_idx in (1, 7, 11):
                p.font.size = Pt(9.5)
                p.font.bold = True
                p.font.color.rgb = NAVY
                cell.fill.solid()
                cell.fill.fore_color.rgb = BORDER_COLOR
            else:
                p.font.size = Pt(9)
                cell.fill.solid()
                cell.fill.fore_color.rgb = LIGHT_BG if r_idx % 2 == 1 else WHITE
                if c_idx == 0:
                    p.font.bold = True
                    p.font.color.rgb = SLATE_DARK
                elif c_idx == 3:
                    p.font.bold = True
                    p.font.color.rgb = NAVY
                else:
                    p.font.color.rgb = SLATE_DARK

    tb_b10 = s10.shapes.add_textbox(Inches(0.8), Inches(6.45), Inches(11.7), Inches(0.4))
    p_b10 = tb_b10.text_frame.paragraphs[0]
    p_b10.text = "A single model adapts to available inputs: single-lead ambulatory screening or 3-lead clinical triage."
    p_b10.alignment = PP_ALIGN.CENTER
    p_b10.font.size = Pt(11.5)
    p_b10.font.color.rgb = NAVY

    set_notes(s10, "This slide highlights model evaluation under flexible lead inputs. Because we train with dynamic lead token masking, the exact same model weights accept variable combinations of leads. At home, single-lead Lead I achieves an ECGFounder Macro AUROC of 0.821 and detects AF with an AUROC of 0.952. When a patient attends clinic, adding Leads II and V2 resolves orthogonal planes without changing model weights: AF detection AUROC reaches 0.981, VT detection reaches 0.989, and QRS duration error drops to 8.12 ms. On EchoNext, the model maintains 75.4% AUROC and 81.4% LVEF concordance.")

    # =========================================================================
    # Slide 11: Single-Lead Wearable Gallery
    # =========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    add_header(s11, "Single-Lead Reconstruction: Smartwatch (Lead I) vs. Patch (Lead II)", "Reconstruction of PTB-XL Record #10283 from single-channel inputs")
    add_footer(s11, 11)

    # Left Panel
    c1_11 = s11.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.7), Inches(5.7), Inches(4.7))
    c1_11.fill.solid()
    c1_11.fill.fore_color.rgb = LIGHT_BG
    c1_11.line.color.rgb = NAVY
    c1_11.line.width = Pt(1.5)

    tb1_t = s11.shapes.add_textbox(Inches(1.0), Inches(1.75), Inches(5.3), Inches(0.35))
    tf1_t = tb1_t.text_frame
    p = tf1_t.paragraphs[0]
    p.text = "Lead I Input (Transverse Axis)"
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
        "Single-channel input: Lead I (wrist-to-wrist vector).",
        "11 reconstructed leads yield mean r = 0.941.",
        "Precordial correlation: V1 (r=0.995), V2 (r=0.989), V3 (r=0.988)."
    ]
    for idx, it in enumerate(items_l1_g):
        p = tf1_b.paragraphs[0] if idx == 0 else tf1_b.add_paragraph()
        p.text = f"- {it}"
        p.font.size = Pt(10)
        p.font.color.rgb = SLATE_DARK
        p.space_before = Pt(3)

    # Right Panel
    c2_11 = s11.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.7), Inches(5.7), Inches(4.7))
    c2_11.fill.solid()
    c2_11.fill.fore_color.rgb = LIGHT_BG
    c2_11.line.color.rgb = CYAN
    c2_11.line.width = Pt(1.5)

    tb2_t = s11.shapes.add_textbox(Inches(7.0), Inches(1.75), Inches(5.3), Inches(0.35))
    tf2_t = tb2_t.text_frame
    p = tf2_t.paragraphs[0]
    p.text = "Lead II Input (Vertical Axis)"
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
        "Single-channel input: Lead II (vertical torso vector).",
        "11 reconstructed leads yield mean r = 0.919.",
        "Inferior leads: aVF (r=0.992), III (r=0.963), V2 (r=0.966)."
    ]
    for idx, it in enumerate(items_l2_g):
        p = tf2_b.paragraphs[0] if idx == 0 else tf2_b.add_paragraph()
        p.text = f"- {it}"
        p.font.size = Pt(10)
        p.font.color.rgb = SLATE_DARK
        p.space_before = Pt(3)

    tb_b11 = s11.shapes.add_textbox(Inches(0.8), Inches(6.45), Inches(11.7), Inches(0.4))
    p_b11 = tb_b11.text_frame.paragraphs[0]
    p_b11.text = "Lead I input provides strong anterior precordial recovery (r > 0.98), whereas Lead II prioritizes inferior leads."
    p_b11.alignment = PP_ALIGN.CENTER
    p_b11.font.size = Pt(11.5)
    p_b11.font.color.rgb = NAVY

    set_notes(s11, "Here is the single-lead comparison between sensor modalities on unseen PTB-XL Record 10283. On the left, we input single-lead Lead I—the wrist-to-wrist vector acquired by smartwatch devices. The model infers the other 11 leads with a record mean correlation of 0.941. Notice the reconstruction of precordial chest leads: V1 is 0.995, V2 is 0.989, and V3 is 0.988. On the right, we input single-lead Lead II—the vertical vector typically captured by adhesive chest patches. Lead II achieves a record mean correlation of 0.919, with recovery of inferior leads like aVF at 0.992 and III at 0.963.")

    # =========================================================================
    # Slide 12: In-Clinic Escalation Gallery
    # =========================================================================
    s12 = prs.slides.add_slide(blank_layout)
    add_header(s12, "Impact of Input Dimension: 1-Lead vs. 3-Lead Input", "Reconstruction of PTB-XL Record #10283 under single-lead and 3-lead configurations")
    add_footer(s12, 12)

    c1_12 = s12.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.7), Inches(5.7), Inches(4.7))
    c1_12.fill.solid()
    c1_12.fill.fore_color.rgb = LIGHT_BG
    c1_12.line.color.rgb = NAVY
    c1_12.line.width = Pt(1.5)

    tb1_t = s12.shapes.add_textbox(Inches(1.0), Inches(1.75), Inches(5.3), Inches(0.35))
    tf1_t = tb1_t.text_frame
    p = tf1_t.paragraphs[0]
    p.text = "Single-Lead Input (Lead I)"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = NAVY

    if os.path.exists(fig_l1_path):
        s12.shapes.add_picture(fig_l1_path, Inches(0.95), Inches(2.15), Inches(5.4), Inches(2.8))

    tb1_b = s12.shapes.add_textbox(Inches(1.0), Inches(5.05), Inches(5.3), Inches(1.3))
    tf1_b = tb1_b.text_frame
    tf1_b.word_wrap = True
    items_12_l = [
        "Single-channel input: Lead I.",
        "11 reconstructed leads yield mean r = 0.941.",
        "Designed for continuous ambulatory surveillance."
    ]
    for idx, it in enumerate(items_12_l):
        p = tf1_b.paragraphs[0] if idx == 0 else tf1_b.add_paragraph()
        p.text = f"- {it}"
        p.font.size = Pt(10)
        p.font.color.rgb = SLATE_DARK
        p.space_before = Pt(3)

    c2_12 = s12.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.7), Inches(5.7), Inches(4.7))
    c2_12.fill.solid()
    c2_12.fill.fore_color.rgb = LIGHT_BG
    c2_12.line.color.rgb = CYAN
    c2_12.line.width = Pt(1.5)

    tb2_t = s12.shapes.add_textbox(Inches(7.0), Inches(1.75), Inches(5.3), Inches(0.35))
    tf2_t = tb2_t.text_frame
    p = tf2_t.paragraphs[0]
    p.text = "Three-Lead Input (I, II, V2)"
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
        "Three-channel input: orthogonal triad (I, II, V2).",
        "9 reconstructed leads yield r = 0.980 - 1.000.",
        "Limb completion: r = 1.000; QRS MAE = 8.12 ms."
    ]
    for idx, it in enumerate(items_12_r):
        p = tf2_b.paragraphs[0] if idx == 0 else tf2_b.add_paragraph()
        p.text = f"- {it}"
        p.font.size = Pt(10)
        p.font.color.rgb = SLATE_DARK
        p.space_before = Pt(3)

    tb_b12 = s12.shapes.add_textbox(Inches(0.8), Inches(6.45), Inches(11.7), Inches(0.4))
    p_b12 = tb_b12.text_frame.paragraphs[0]
    p_b12.text = "Adding two orthogonal leads (II and V2) constrains both planes, improving precordial and limb fidelity."
    p_b12.alignment = PP_ALIGN.CENTER
    p_b12.font.size = Pt(11.5)
    p_b12.font.color.rgb = NAVY

    set_notes(s12, "This comparison shows scaling from single-lead to 3-lead input. On the left is single-lead Lead I input with a mean correlation of 0.941. While suitable for outpatient rhythm surveillance, a single lead is mathematically limited in resolving orthogonal planes simultaneously. On the right, adding Lead II and V2 provides orthogonal axes. Einthoven completion yields 1.000 across all limb leads, and precordial chest leads reach correlations between 0.980 and 0.999, reducing QRS duration error to 8.12 ms.")

    # =========================================================================
    # Slide 13: External Validation: Sunnybrook Physical 10-Wire XMLs
    # =========================================================================
    s13 = prs.slides.add_slide(blank_layout)
    add_header(s13, "External Cohort Validation: Sunnybrook In-Clinic XMLs", "Zero-shot evaluation on Record ECG004 under 1-lead and 3-lead configurations")
    add_footer(s13, 13)

    c1_13 = s13.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.7), Inches(5.7), Inches(4.7))
    c1_13.fill.solid()
    c1_13.fill.fore_color.rgb = LIGHT_BG
    c1_13.line.color.rgb = NAVY
    c1_13.line.width = Pt(1.5)

    tb1_t = s13.shapes.add_textbox(Inches(1.0), Inches(1.75), Inches(5.3), Inches(0.35))
    tf1_t = tb1_t.text_frame
    p = tf1_t.paragraphs[0]
    p.text = "Single-Lead Input (Lead I)"
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
        "Single-channel input: Einthoven Lead I.",
        "11 reconstructed leads yield mean r = 0.865.",
        "Selected leads: aVR (r=0.966), V5 (r=0.934), V6 (r=0.929), II (r=0.925)."
    ]
    for idx, it in enumerate(items_sb1):
        p = tf1_b.paragraphs[0] if idx == 0 else tf1_b.add_paragraph()
        p.text = f"- {it}"
        p.font.size = Pt(10)
        p.font.color.rgb = SLATE_DARK
        p.space_before = Pt(3)

    c2_13 = s13.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.7), Inches(5.7), Inches(4.7))
    c2_13.fill.solid()
    c2_13.fill.fore_color.rgb = LIGHT_BG
    c2_13.line.color.rgb = CYAN
    c2_13.line.width = Pt(1.5)

    tb2_t = s13.shapes.add_textbox(Inches(7.0), Inches(1.75), Inches(5.3), Inches(0.35))
    tf2_t = tb2_t.text_frame
    p = tf2_t.paragraphs[0]
    p.text = "Three-Lead Input (I, II, V2)"
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
        "Three-channel input: orthogonal triad (Leads I, II, V2).",
        "9 reconstructed leads yield mean r = 0.938.",
        "Limb completion: r = 1.000; precordials: V1 (r=0.973), V6 (r=0.957)."
    ]
    for idx, it in enumerate(items_sb3):
        p = tf2_b.paragraphs[0] if idx == 0 else tf2_b.add_paragraph()
        p.text = f"- {it}"
        p.font.size = Pt(10)
        p.font.color.rgb = SLATE_DARK
        p.space_before = Pt(3)

    tb_b13 = s13.shapes.add_textbox(Inches(0.8), Inches(6.45), Inches(11.7), Inches(0.4))
    p_b13 = tb_b13.text_frame.paragraphs[0]
    p_b13.text = "Zero-shot evaluation on in-clinic XMLs preserves physiological amplitude (> 2.1 mV in V2) without recalibration."
    p_b13.alignment = PP_ALIGN.CENTER
    p_b13.font.size = Pt(11.5)
    p_b13.font.color.rgb = NAVY

    set_notes(s13, "This slide shows zero-shot evaluation on Sunnybrook hospital XML recordings. Record ECG004 displays clinical voltages—over 1.4 mV in Lead II and 2.1 mV in V2. On the left, using single-lead Lead I across physical hospital hardware with un-derived limb leads and natural noise, the model achieves a record mean correlation of 0.865 (Lead II at 0.925, aVR at 0.966, V5 at 0.934). On the right, with the 3-lead triad, mean correlation increases to 0.938. Limb leads reach 1.000, V1 reaches 0.973, and V6 reaches 0.957.")

    # =========================================================================
    # Slide 14: External Validation: Stanford EchoNext Structural Cohort
    # =========================================================================
    s14 = prs.slides.add_slide(blank_layout)
    add_header(s14, "External Cohort Validation: EchoNext Structural Cohort", "Zero-shot evaluation on Record #10 under 1-lead and 3-lead configurations")
    add_footer(s14, 14)

    c1_14 = s14.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.7), Inches(5.7), Inches(4.7))
    c1_14.fill.solid()
    c1_14.fill.fore_color.rgb = LIGHT_BG
    c1_14.line.color.rgb = NAVY
    c1_14.line.width = Pt(1.5)

    tb1_t = s14.shapes.add_textbox(Inches(1.0), Inches(1.75), Inches(5.3), Inches(0.35))
    tf1_t = tb1_t.text_frame
    p = tf1_t.paragraphs[0]
    p.text = "Single-Lead Input (Lead I)"
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
        "Single-channel input: Lead I.",
        "11 reconstructed leads yield mean r = 0.809.",
        "Selected leads: aVR (r=0.967), aVL (r=0.946), V1 (r=0.942), V5 (r=0.915)."
    ]
    for idx, it in enumerate(items_en1):
        p = tf1_b.paragraphs[0] if idx == 0 else tf1_b.add_paragraph()
        p.text = f"- {it}"
        p.font.size = Pt(10)
        p.font.color.rgb = SLATE_DARK
        p.space_before = Pt(3)

    c2_14 = s14.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.7), Inches(5.7), Inches(4.7))
    c2_14.fill.solid()
    c2_14.fill.fore_color.rgb = LIGHT_BG
    c2_14.line.color.rgb = CYAN
    c2_14.line.width = Pt(1.5)

    tb2_t = s14.shapes.add_textbox(Inches(7.0), Inches(1.75), Inches(5.3), Inches(0.35))
    tf2_t = tb2_t.text_frame
    p = tf2_t.paragraphs[0]
    p.text = "Three-Lead Input (I, II, V2)"
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
        "Three-channel input: orthogonal triad (Leads I, II, V2).",
        "9 reconstructed leads yield mean r = 0.964.",
        "Limb completion: r = 0.986 - 1.000; precordials: V1 (r=0.987), V6 (r=0.965)."
    ]
    for idx, it in enumerate(items_en3):
        p = tf2_b.paragraphs[0] if idx == 0 else tf2_b.add_paragraph()
        p.text = f"- {it}"
        p.font.size = Pt(10)
        p.font.color.rgb = SLATE_DARK
        p.space_before = Pt(3)

    tb_b14 = s14.shapes.add_textbox(Inches(0.8), Inches(6.45), Inches(11.7), Inches(0.4))
    p_b14 = tb_b14.text_frame.paragraphs[0]
    p_b14.text = "Preserves ventricular waveforms on echo-paired tracings (Record #10: peak voltages 0.9 - 1.4 mV; LVEF concordance 81.4%)."
    p_b14.alignment = PP_ALIGN.CENTER
    p_b14.font.size = Pt(11.5)
    p_b14.font.color.rgb = NAVY

    set_notes(s14, "This slide shows external validation on the Stanford EchoNext cohort of paired echocardiogram-ECG patient tracings. Examining Record 10, featuring normal voltages up to 1.40 mV in V4 and 1.15 mV in V2: On the left, 1-lead reconstruction achieves a record mean correlation of 0.809, with aVR at 0.967 and chest leads V1 and V5 at 0.942 and 0.915. On the right, adding the 3-lead triad increases mean correlation to 0.964, with limb tracking between 0.986 and 1.000, and precordial tracking reaching 0.987 on V1 and 0.965 on V6. Preserving ST-T morphology and QRS amplitude allows downstream models to achieve 81.4% concordance on reduced ejection fraction.")

    # =========================================================================
    # Slide 15: Data Collection Methodology: Dual-Stream Protocol
    # =========================================================================
    s15 = prs.slides.add_slide(blank_layout)
    add_header(s15, "Data Collection Methodology: Dual-Stream Protocol", "Ambulatory monitoring and in-clinic paired acquisition")
    add_footer(s15, 15)

    c1 = s15.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.8), Inches(5.7), Inches(4.5))
    c1.fill.solid()
    c1.fill.fore_color.rgb = LIGHT_BG
    c1.line.color.rgb = NAVY
    c1.line.width = Pt(1.5)

    tb = s15.shapes.add_textbox(Inches(1.1), Inches(2.0), Inches(5.1), Inches(4.1))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Stream A: Ambulatory Surveillance"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = NAVY

    items_s15_1 = [
        "Daily 30-second smartwatch ECG (Lead I) for 6 months post-ablation.",
        "14-day continuous CardioSTAT patch worn at 3 and 6 months.",
        "Primary trial endpoints: time to AF recurrence, AF burden, and symptom correlation.",
        "Automated data transmission to Sunnybrook secure study server."
    ]
    for it in items_s15_1:
        p = tf.add_paragraph()
        p.text = f"- {it}"
        p.font.size = Pt(12)
        p.font.color.rgb = SLATE_DARK
        p.space_before = Pt(8)

    c2 = s15.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.5))
    c2.fill.solid()
    c2.fill.fore_color.rgb = LIGHT_BG
    c2.line.color.rgb = CYAN
    c2.line.width = Pt(1.5)

    tb2 = s15.shapes.add_textbox(Inches(7.1), Inches(2.0), Inches(5.1), Inches(4.1))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    p = tf2.paragraphs[0]
    p.text = "Stream B: In-Clinic Paired Protocol"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = CYAN

    items_s15_2 = [
        "Four scheduled clinical visits: baseline, post-procedure, 3 months, and 6 months.",
        "Simultaneous recording of 12-lead ECG and smartwatch Lead I (delta t = 0).",
        "Standardized acquisition adds <= 5 minutes to routine visit.",
        "Requires no additional clinic visits or invasive procedures."
    ]
    for it in items_s15_2:
        p = tf2.add_paragraph()
        p.text = f"- {it}"
        p.font.size = Pt(12)
        p.font.color.rgb = SLATE_DARK
        p.space_before = Pt(8)

    tb_sub15 = s15.shapes.add_textbox(Inches(0.8), Inches(6.45), Inches(11.7), Inches(0.4))
    p_sub15 = tb_sub15.text_frame.paragraphs[0]
    p_sub15.text = "Stream A fulfills primary trial objectives; Stream B provides synchronous paired data for secondary analysis."
    p_sub15.alignment = PP_ALIGN.CENTER
    p_sub15.font.size = Pt(11.5)
    p_sub15.font.color.rgb = NAVY

    set_notes(s15, "This slide outlines the data collection workflow. It operates as two synchronized streams: Stream A is the ambulatory home surveillance from Protocol v1.1. Patients wear the watch daily for 6 months and wear the 14-day continuous CardioSTAT patch at 3 and 6 months. This feeds directly into our primary clinical efficacy endpoint. Stream B is the in-clinic paired protocol. Patients are already at Sunnybrook for four standard clinical visits: baseline pre-ablation, procedure day, 3 months, and 6 months. During the routine clinical 12-lead ECG, the research coordinator records a 30-second smartwatch tracing simultaneously. It adds less than 5 minutes to routine care and creates synchronized ground-truth pairs.")

    # =========================================================================
    # Slide 16: The Paired In-Clinic Evaluation Set
    # =========================================================================
    s16 = prs.slides.add_slide(blank_layout)
    add_header(s16, "Paired In-Clinic Dataset Design", "Comparison with prior retrospective evaluations")
    add_footer(s16, 16)

    rows_eval = [
        ("Methodological Dimension", "Ansari et al. (2026)", "Sunnybrook Prospective Protocol"),
        ("Evaluation Cohort", "Retrospective CareLink archive (N = 26)", "Prospective cohort (N = 96 patients, ~384 paired visits)"),
        ("Temporal Offset", "+/- 7 days between device and 12-lead ECG", "Synchronous acquisition (delta t = 0, simultaneous)"),
        ("Physiological State", "Uncontrolled variations in hydration and medication", "Identical resting state (supine, matched heart rate)"),
        ("Rhythm States", "Uncontrolled outpatient recordings", "Defined clinical milestones (baseline AF, post-procedure, follow-up)"),
        ("Methodological Role", "Device-specific feasibility check", "Prospective benchmark for domain adaptation and prior models")
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
    p.text = "Significance of Prospective Paired Data: Most published ECG reconstruction studies evaluate models on synthetically masked 12-lead recordings. Synthetic masking does not account for dry-electrode interfaces, skin impedance, or wearable filtering. Synchronous in-clinic acquisition provides an empirical standard for evaluating wearable reconstruction models."
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = SUCCESS_GREEN

    set_notes(s16, "This slide contrasts our prospective paired acquisition with the retrospective analysis in the Stanford paper. Ansari and Rogers searched through Medtronic CareLink transmissions, identifying 26 patients who had a clinical 12-lead within plus-or-minus 7 days. Over a week, autonomic tone, electrolytes, and medications can vary. In our trial, we acquire simultaneous pairs across 96 patients at 4 defined clinical visits. Furthermore, this cohort captures longitudinal transitions: pre-ablation AF, acute post-procedure rhythm, and 3-to-6-month follow-up. This directly addresses the gap between synthetic lead masking and physical wearable recording.")

    # =========================================================================
    # Slide 17: Figure 3: Integrated Clinical-AI Trial Pipeline
    # =========================================================================
    s17 = prs.slides.add_slide(blank_layout)
    add_header(s17, "Figure 3: Integrated Clinical-AI Trial Pipeline", "Trial workflow and computational integration")
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
    p.text = "Integration Stages"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = NAVY

    stages = [
        "Pretraining on public cohorts (PTB-XL, MIMIC-IV, Icentia11k) to establish baseline representations.",
        "Data acquisition across three streams: smartwatch daily rhythm and in-clinic paired Lead I; 14-day continuous CardioSTAT patch; hospital 12-lead machine.",
        "Hardware transfer function estimation and adapter calibration on paired recordings.",
        "Evaluation on distinct endpoints: primary AF recurrence and burden vs. CardioSTAT patch; secondary 12-lead reconstruction and repolarization tracking."
    ]

    for desc in stages:
        p = tf.add_paragraph()
        p.text = f"- {desc}"
        p.font.size = Pt(11)
        p.font.color.rgb = SLATE_DARK
        p.space_before = Pt(8)

    set_notes(s17, "Figure 3 shows how the clinical trial and AI engineering integrate. Stage 1 represents foundation models pretrained on public datasets. Stage 2 represents data acquisition: smartwatch, 14-day continuous CardioSTAT patch, and hospital 12-lead machine. Stage 3 uses the paired in-clinic data to calibrate the hardware transfer function and adapt the model. Stage 4 shows the separation of endpoints: primary clinical AF surveillance vs. secondary translational 12-lead reconstruction.")

    # =========================================================================
    # Slide 18: Sample Size, Statistical Power & Clustered Modeling
    # =========================================================================
    s18 = prs.slides.add_slide(blank_layout)
    add_header(s18, "Sample Size, Statistical Power & Clustered Modeling", "Sample size justification and statistical analysis plan")
    add_footer(s18, 18)

    c1 = s18.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.8), Inches(5.7), Inches(4.5))
    c1.fill.solid()
    c1.fill.fore_color.rgb = LIGHT_BG
    c1.line.color.rgb = NAVY
    c1.line.width = Pt(1.5)

    tb = s18.shapes.add_textbox(Inches(1.1), Inches(2.0), Inches(5.1), Inches(4.1))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Primary Clinical Power (Protocol v1.1)"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = NAVY

    items_s18_1 = [
        "Primary outcome: time to confirmed AF recurrence.",
        "Hypothesis: smartwatch surveillance is non-inferior to 14-day patch monitoring.",
        "Event rate assumptions: 20% recurrence with patch, 25% with smartwatch.",
        "Target power: 80% power, alpha = 0.05, 10% non-inferiority margin (N = 96 patients).",
        "Vanguard phase: initial 20 consecutive patients to evaluate workflow feasibility."
    ]
    for it in items_s18_1:
        p = tf.add_paragraph()
        p.text = f"- {it}"
        p.font.size = Pt(11.5)
        p.font.color.rgb = SLATE_DARK
        p.space_before = Pt(6)

    c2 = s18.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.5))
    c2.fill.solid()
    c2.fill.fore_color.rgb = LIGHT_BG
    c2.line.color.rgb = CYAN
    c2.line.width = Pt(1.5)

    tb2 = s18.shapes.add_textbox(Inches(7.1), Inches(2.0), Inches(5.1), Inches(4.1))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    p = tf2.paragraphs[0]
    p.text = "Secondary AI Precision & Analysis Plan"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = CYAN

    items_s18_2 = [
        "Sample yield: 96 patients across 4 clinic visits provides >2,500 paired complexes.",
        "Precision target: bounds QTc estimation error within 95% CI margin of +/- 3.5 ms.",
        "Within-patient beat clustering violates independent observation assumptions.",
        "Linear mixed-effects models (unstructured covariance) and generalized estimating equations (GEE)."
    ]
    for it in items_s18_2:
        p = tf2.add_paragraph()
        p.text = f"- {it}"
        p.font.size = Pt(11.5)
        p.font.color.rgb = SLATE_DARK
        p.space_before = Pt(6)

    tb_sub18 = s18.shapes.add_textbox(Inches(0.8), Inches(6.45), Inches(11.7), Inches(0.4))
    p_sub18 = tb_sub18.text_frame.paragraphs[0]
    p_sub18.text = "Primary sample size is powered on clinical event rates; secondary analysis accounts for repeated measures."
    p_sub18.alignment = PP_ALIGN.CENTER
    p_sub18.font.size = Pt(11.5)
    p_sub18.font.color.rgb = NAVY

    set_notes(s18, "This slide outlines sample size and statistical considerations. The primary clinical trial sample size of 96 patients is calculated directly from expected AF recurrence rates—20% on patch vs 25% on smartwatch with an 80% power target. For the secondary AI aim, 96 patients across multiple clinic visits provides over 2,500 paired cardiac complexes. Per Riley's 2024 BMJ guidelines, this is sufficient to bound QTc measurement error within a 3.5 ms margin of error. We specify linear mixed-effects models and generalized estimating equations to adjust for intra-patient beat clustering.")

    # =========================================================================
    # Slide 19: Action Items & Alignment for Today's Meeting
    # =========================================================================
    s19 = prs.slides.add_slide(blank_layout)
    add_header(s19, "Action Items & Alignment for Today's Meeting", "Decisions for protocol finalization")
    add_footer(s19, 19)

    c1 = s19.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.8), Inches(6.0), Inches(4.5))
    c1.fill.solid()
    c1.fill.fore_color.rgb = LIGHT_BG
    c1.line.color.rgb = NAVY
    c1.line.width = Pt(1.5)

    tb = s19.shapes.add_textbox(Inches(1.1), Inches(2.0), Inches(5.4), Inches(4.1))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Decisions for Today's Meeting"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = NAVY

    items_s19 = [
        "1. Protocol structure: Approve formal separation of Aim 1 (clinical trial) and Aim 2 (translational AI).",
        "2. Paired recording protocol: Approve 5-minute simultaneous in-clinic acquisition procedure.",
        "3. Secondary biomarker selection: Finalize priority clinical biomarker (QTc monitoring vs. conduction intervals).",
        "4. Vanguard enrollment: Authorize start of the 20-patient feasibility vanguard."
    ]
    for it in items_s19:
        p = tf.add_paragraph()
        p.text = it
        p.font.size = Pt(11.5)
        p.font.color.rgb = SLATE_DARK
        p.space_before = Pt(8)

    c2 = s19.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.1), Inches(1.8), Inches(5.4), Inches(4.5))
    c2.fill.solid()
    c2.fill.fore_color.rgb = LIGHT_BG
    c2.line.color.rgb = CYAN
    c2.line.width = Pt(1.5)

    tb2 = s19.shapes.add_textbox(Inches(7.4), Inches(2.0), Inches(4.8), Inches(4.1))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    p = tf2.paragraphs[0]
    p.text = "Expected Outcomes"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = CYAN

    items_ben = [
        "Clinical outcomes: Prospective evaluation of smartwatch surveillance for post-ablation AF recurrence.",
        "Methodological outcomes: Benchmark dataset and algorithms for wearable 12-lead reconstruction.",
        "Trial integrity: Clear division between primary regulatory endpoints and exploratory AI objectives."
    ]
    for it in items_ben:
        p = tf2.add_paragraph()
        p.text = f"- {it}"
        p.font.size = Pt(11.5)
        p.font.color.rgb = SLATE_DARK
        p.space_before = Pt(8)

    tb_sub19 = s19.shapes.add_textbox(Inches(0.8), Inches(6.45), Inches(11.7), Inches(0.4))
    p_sub19 = tb_sub19.text_frame.paragraphs[0]
    p_sub19.text = "Next Step: Finalize protocol amendment text and initiate vanguard enrollment."
    p_sub19.alignment = PP_ALIGN.CENTER
    p_sub19.font.size = Pt(11.5)
    p_sub19.font.color.rgb = NAVY

    set_notes(s19, "To close our meeting, here are the 4 decisions we need to agree on: First, confirm the two-track protocol amendment: Aim 1 primary AF recurrence, Aim 2 secondary 12-lead reconstruction. Second, approve the 5-minute simultaneous recording procedure during in-clinic visits. Third, decide which secondary clinical biomarker Chris wants to prioritize for the paired analysis—QTc interval tracking is our recommendation. Fourth, approve the 20-patient vanguard cohort. Thank you Chris and Alex; let us open the floor for discussion.")

    output_path = "slides/presentation.pptx"
    prs.save(output_path)
    print(f"Presentation saved successfully to {output_path} (Total slides: 19)")

if __name__ == "__main__":
    create_deck()
