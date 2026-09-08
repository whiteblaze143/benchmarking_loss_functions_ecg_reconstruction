#!/usr/bin/env python3
"""
Generate an 8-slide, highly polished 16:9 widescreen PowerPoint deck (.pptx)
for the 5-Minute CMU Forum on Biomedical Engineering presentation:
"Can We Reconstruct a Diagnostic 12-Lead ECG from Only a Few Measured Leads?"
Authors: Mithun Manivannan, Alex Mariakakis, Christopher Cheung
Affiliations: University of Toronto, Sunnybrook Health Sciences Centre
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

def create_deck(output_path="slides/cmu_presentation_5min.pptx"):
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # Professional Palette
    NAVY = RGBColor(15, 32, 66)          # #0F2042 Primary headers & accents
    NAVY_LIGHT = RGBColor(30, 58, 138)    # #1E3A8A Secondary text & subheadings
    CYAN = RGBColor(2, 132, 199)          # #0284C7 Accent highlights & tags
    SLATE_DARK = RGBColor(30, 41, 59)     # #1E293B High-contrast body text
    SLATE_MUTED = RGBColor(100, 116, 139) # #64748B Subtitles & references
    LIGHT_BG = RGBColor(248, 250, 252)    # #F8FAFC Card container backgrounds
    BORDER_COLOR = RGBColor(203, 213, 225)# #CBD5E1 Clean card borders
    ALERT_RED = RGBColor(220, 38, 38)     # #DC2626 Warning / baseline collapse
    SUCCESS_GREEN = RGBColor(5, 150, 105) # #059669 Positive transfer & pass gates

    TOTAL_SLIDES = 8

    def add_header(slide, eyebrow, title_text, subtitle_text=None):
        tb = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.733), Inches(1.1))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        
        # Eyebrow tag
        p_eye = tf.paragraphs[0]
        p_eye.text = eyebrow.upper()
        p_eye.font.size = Pt(9.5)
        p_eye.font.bold = True
        p_eye.font.color.rgb = CYAN
        p_eye.font.name = "Arial"
        
        # Main Title
        p_title = tf.add_paragraph()
        p_title.text = title_text
        p_title.font.size = Pt(20)
        p_title.font.bold = True
        p_title.font.color.rgb = NAVY
        p_title.font.name = "Arial"
        
        # Subtitle
        if subtitle_text:
            p_sub = tf.add_paragraph()
            p_sub.text = subtitle_text
            p_sub.font.size = Pt(11.5)
            p_sub.font.color.rgb = SLATE_MUTED
            p_sub.font.name = "Arial"

    def add_footer(slide, slide_num, ref_text=None):
        if ref_text:
            tb_ref = slide.shapes.add_textbox(Inches(0.8), Inches(6.72), Inches(11.733), Inches(0.25))
            tf_ref = tb_ref.text_frame
            tf_ref.margin_left = tf_ref.margin_top = tf_ref.margin_right = tf_ref.margin_bottom = 0
            p_ref = tf_ref.paragraphs[0]
            p_ref.text = f"Ref: {ref_text}"
            p_ref.font.size = Pt(8)
            p_ref.font.color.rgb = SLATE_MUTED
            p_ref.font.name = "Arial"

        line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(7.0), Inches(11.733), Inches(0.015))
        line.fill.solid()
        line.fill.fore_color.rgb = BORDER_COLOR
        line.line.fill.background()

        # Left: Forum & Abstract metadata
        tb_left = slide.shapes.add_textbox(Inches(0.8), Inches(7.05), Inches(9.5), Inches(0.35))
        tf_left = tb_left.text_frame
        tf_left.margin_left = tf_left.margin_top = tf_left.margin_right = tf_left.margin_bottom = 0
        p = tf_left.paragraphs[0]
        p.text = "Carnegie Mellon Forum on Biomedical Engineering 2026  •  Abstract #194  •  Discrete Latent ECG Reconstruction"
        p.font.size = Pt(9)
        p.font.color.rgb = SLATE_MUTED
        p.font.name = "Arial"
        
        # Right: Slide number
        tb_right = slide.shapes.add_textbox(Inches(10.5), Inches(7.05), Inches(2.033), Inches(0.35))
        tf_right = tb_right.text_frame
        tf_right.margin_left = tf_right.margin_top = tf_right.margin_right = tf_right.margin_bottom = 0
        p_num = tf_right.paragraphs[0]
        p_num.text = f"Slide {slide_num} of {TOTAL_SLIDES}"
        p_num.font.size = Pt(9)
        p_num.font.bold = True
        p_num.font.color.rgb = NAVY_LIGHT
        p_num.font.name = "Arial"
        p_num.alignment = PP_ALIGN.RIGHT

    def set_notes(slide, notes_text):
        notes_slide = slide.notes_slide
        tf = notes_slide.notes_text_frame
        tf.text = notes_text

    # =========================================================================
    # Slide 1: Title Slide & Core Question (10–15 sec)
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    
    uoft_logo_path = "slides/figures/logos/uoft_cs_logo.png"
    sb_logo_path = "slides/figures/logos/sunnybrook_logo.png"
    if os.path.exists(uoft_logo_path):
        s1.shapes.add_picture(uoft_logo_path, Inches(0.8), Inches(0.5), Inches(2.35), Inches(0.55))
    if os.path.exists(sb_logo_path):
        s1.shapes.add_picture(sb_logo_path, Inches(10.183), Inches(0.5), Inches(2.35), Inches(0.55))

    tb1 = s1.shapes.add_textbox(Inches(1.0), Inches(1.5), Inches(11.333), Inches(2.6))
    tf1 = tb1.text_frame
    tf1.word_wrap = True
    
    p = tf1.paragraphs[0]
    p.text = "2026 CARNEGIE MELLON FORUM ON BIOMEDICAL ENGINEERING  •  ABSTRACT #194"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = CYAN
    p.font.name = "Arial"
    
    p2 = tf1.add_paragraph()
    p2.text = "Can We Reconstruct a Diagnostic 12-Lead ECG from Only a Few Measured Leads?"
    p2.font.size = Pt(26)
    p2.font.bold = True
    p2.font.color.rgb = NAVY
    p2.font.name = "Arial"
    p2.space_before = Pt(8)
    
    p3 = tf1.add_paragraph()
    p3.text = "Discrete Latent Sequence Generation for Limited-Lead ECG Reconstruction"
    p3.font.size = Pt(15)
    p3.font.color.rgb = SLATE_MUTED
    p3.font.name = "Arial"
    p3.space_before = Pt(6)

    card_authors = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(4.3), Inches(11.333), Inches(1.6))
    card_authors.fill.solid()
    card_authors.fill.fore_color.rgb = LIGHT_BG
    card_authors.line.color.rgb = BORDER_COLOR
    card_authors.line.width = Pt(1.0)
    
    tb_auth = s1.shapes.add_textbox(Inches(1.2), Inches(4.45), Inches(10.9), Inches(1.3))
    tf_auth = tb_auth.text_frame
    tf_auth.word_wrap = True
    
    pa1 = tf_auth.paragraphs[0]
    pa1.text = "Mithun Manivannan, BSc¹'²  •  Alex Mariakakis, PhD¹  •  Christopher Cheung, MD²"
    pa1.font.size = Pt(14)
    pa1.font.bold = True
    pa1.font.color.rgb = NAVY
    pa1.alignment = PP_ALIGN.CENTER
    
    pa2 = tf_auth.add_paragraph()
    pa2.text = "¹Department of Computer Science, University of Toronto    |    ²Division of Cardiology, Sunnybrook Health Sciences Centre"
    pa2.font.size = Pt(11.5)
    pa2.font.color.rgb = SLATE_DARK
    pa2.space_before = Pt(4)
    pa2.alignment = PP_ALIGN.CENTER

    pa3 = tf_auth.add_paragraph()
    pa3.text = "Contact: mithun.manivannan@sri.utoronto.ca"
    pa3.font.size = Pt(10.5)
    pa3.font.color.rgb = CYAN
    pa3.space_before = Pt(4)
    pa3.alignment = PP_ALIGN.CENTER

    set_notes(s1, "Hello everyone. I’m Mithun Manivannan from the University of Toronto and Sunnybrook Health Sciences Centre. Today I am presenting our work asking a fundamental question: can we reconstruct a full diagnostic twelve-lead electrocardiogram from only a few measured wearable leads?")

    # =========================================================================
    # Slide 2: 12 Leads vs. One Wearable Lead (30 sec)
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "Clinical Context", "12 Leads vs. One Wearable Lead", "Comparing full hospital diagnostic fields against consumer wearable monitoring")
    add_footer(s2, 2, "Perez et al., Apple Heart Study, N Engl J Med 2019; Cheung et al., Heart Rhythm 2023.")

    # Left: ECG Delineation Image + Fiducials Card
    delineation_img = "slides/figures/ecg_delineation_components.png"
    if os.path.exists(delineation_img):
        s2.shapes.add_picture(delineation_img, Inches(0.8), Inches(1.65), Inches(6.0), Inches(3.1))

    c2_fid = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(4.9), Inches(6.0), Inches(1.65))
    c2_fid.fill.solid()
    c2_fid.fill.fore_color.rgb = LIGHT_BG
    c2_fid.line.color.rgb = BORDER_COLOR
    c2_fid.line.width = Pt(1.0)
    tb2_fid = s2.shapes.add_textbox(Inches(1.0), Inches(5.05), Inches(5.6), Inches(1.35))
    tf2_fid = tb2_fid.text_frame
    tf2_fid.word_wrap = True
    p = tf2_fid.paragraphs[0]
    p.text = "Electrophysiological Fiducials"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = NAVY
    p.alignment = PP_ALIGN.CENTER

    p_fid_sub = tf2_fid.add_paragraph()
    p_fid_sub.text = "P wave (atria)  →  QRS complex (ventricles)  →  ST–T (repolarization)"
    p_fid_sub.font.size = Pt(11.5)
    p_fid_sub.font.color.rgb = SLATE_DARK
    p_fid_sub.space_before = Pt(6)
    p_fid_sub.alignment = PP_ALIGN.CENTER

    # Right Top Card: Clinical 12-Lead
    c2_1 = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.1), Inches(1.65), Inches(5.433), Inches(2.25))
    c2_1.fill.solid()
    c2_1.fill.fore_color.rgb = LIGHT_BG
    c2_1.line.color.rgb = NAVY
    c2_1.line.width = Pt(1.5)
    tb2_1 = s2.shapes.add_textbox(Inches(7.3), Inches(1.78), Inches(5.033), Inches(2.0))
    tf2_1 = tb2_1.text_frame
    tf2_1.word_wrap = True
    p = tf2_1.paragraphs[0]
    p.text = "Clinical 12-Lead ECG"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = NAVY
    p.alignment = PP_ALIGN.CENTER
    
    p_sub = tf2_1.add_paragraph()
    p_sub.text = "12 Spatial Views"
    p_sub.font.size = Pt(13)
    p_sub.font.bold = True
    p_sub.font.color.rgb = NAVY_LIGHT
    p_sub.space_before = Pt(2)
    p_sub.alignment = PP_ALIGN.CENTER

    p_body = tf2_1.add_paragraph()
    p_body.text = "• Frontal (I–aVF) + Precordial (V1–V6) planes\n• Comprehensive 3D cardiac vector field"
    p_body.font.size = Pt(11.5)
    p_body.font.color.rgb = SLATE_DARK
    p_body.space_before = Pt(6)

    # Right Bottom Card: Smartwatch ECG
    c2_2 = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.1), Inches(4.1), Inches(5.433), Inches(2.45))
    c2_2.fill.solid()
    c2_2.fill.fore_color.rgb = LIGHT_BG
    c2_2.line.color.rgb = ALERT_RED
    c2_2.line.width = Pt(1.5)
    tb2_2 = s2.shapes.add_textbox(Inches(7.3), Inches(4.23), Inches(5.033), Inches(2.2))
    tf2_2 = tb2_2.text_frame
    tf2_2.word_wrap = True
    p = tf2_2.paragraphs[0]
    p.text = "Smartwatch ECG"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = ALERT_RED
    p.alignment = PP_ALIGN.CENTER

    p_sub = tf2_2.add_paragraph()
    p_sub.text = "1 Measured Projection"
    p_sub.font.size = Pt(13)
    p_sub.font.bold = True
    p_sub.font.color.rgb = ALERT_RED
    p_sub.space_before = Pt(2)
    p_sub.alignment = PP_ALIGN.CENTER

    p_body = tf2_2.add_paragraph()
    p_body.text = "• Single wrist transverse vector (Lead I)\n• Completely blind to vertical & chest planes"
    p_body.font.size = Pt(11.5)
    p_body.font.color.rgb = SLATE_DARK
    p_body.space_before = Pt(6)

    p_bot = tf2_2.add_paragraph()
    p_bot.text = "Most spatial information is unobserved."
    p_bot.font.size = Pt(11.5)
    p_bot.font.bold = True
    p_bot.font.color.rgb = ALERT_RED
    p_bot.space_before = Pt(6)
    p_bot.alignment = PP_ALIGN.CENTER

    set_notes(s2, "Standard clinical care relies on a twelve-lead ECG, using ten electrodes to capture twelve spatial projections across frontal and horizontal planes. Consumer smartwatches, by contrast, measure only a single projection—Lead I across the wrists. While great for rhythm screening, most spatial information remains completely unobserved.")

    # =========================================================================
    # Slide 3: The Reconstruction Challenge (30 sec)
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "Mathematical Formulation", "The Reconstruction Challenge", "Inferring unobserved 3D fields without statistical mean collapse")
    add_footer(s3, 3, "Frank, Circulation 1956; Atoui et al., IEEE TBME 2010.")

    # Top Schematic Card
    c3_top = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.65), Inches(11.733), Inches(1.15))
    c3_top.fill.solid()
    c3_top.fill.fore_color.rgb = LIGHT_BG
    c3_top.line.color.rgb = NAVY
    c3_top.line.width = Pt(1.5)
    tb3_top = s3.shapes.add_textbox(Inches(1.0), Inches(1.75), Inches(11.333), Inches(0.95))
    tf3_top = tb3_top.text_frame
    p = tf3_top.paragraphs[0]
    p.text = "One Observed Lead    ⟶    [ ? ]    ⟶    12-Lead ECG"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = NAVY
    p.alignment = PP_ALIGN.CENTER

    # Dual Concept Cards
    c3_1 = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(3.05), Inches(5.7), Inches(3.4))
    c3_1.fill.solid()
    c3_1.fill.fore_color.rgb = LIGHT_BG
    c3_1.line.color.rgb = NAVY_LIGHT
    c3_1.line.width = Pt(1.5)
    tb3_1 = s3.shapes.add_textbox(Inches(1.0), Inches(3.25), Inches(5.3), Inches(3.0))
    tf3_1 = tb3_1.text_frame
    tf3_1.word_wrap = True
    p = tf3_1.paragraphs[0]
    p.text = "Spatial Ambiguity"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = NAVY_LIGHT
    p.alignment = PP_ALIGN.CENTER
    p2 = tf3_1.add_paragraph()
    p2.text = "Different cardiac electrical states can produce similar single-lead measurements."
    p2.font.size = Pt(13)
    p2.font.color.rgb = SLATE_DARK
    p2.space_before = Pt(14)
    p2.alignment = PP_ALIGN.CENTER

    c3_2 = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.833), Inches(3.05), Inches(5.7), Inches(3.4))
    c3_2.fill.solid()
    c3_2.fill.fore_color.rgb = LIGHT_BG
    c3_2.line.color.rgb = SUCCESS_GREEN
    c3_2.line.width = Pt(1.5)
    tb3_2 = s3.shapes.add_textbox(Inches(7.033), Inches(3.25), Inches(5.3), Inches(3.0))
    tf3_2 = tb3_2.text_frame
    tf3_2.word_wrap = True
    p = tf3_2.paragraphs[0]
    p.text = "Generative Inference"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = SUCCESS_GREEN
    p.alignment = PP_ALIGN.CENTER
    p2 = tf3_2.add_paragraph()
    p2.text = "The model therefore has to infer plausible missing morphology without simply smoothing it away."
    p2.font.size = Pt(13)
    p2.font.color.rgb = SLATE_DARK
    p2.space_before = Pt(14)
    p2.alignment = PP_ALIGN.CENTER

    set_notes(s3, "This creates a fundamental reconstruction challenge: because multiple distinct cardiac electrical states project onto very similar single-lead signals, the problem is underdetermined. Rather than performing simple pointwise regression that averages away critical spikes, the model must infer plausible missing morphology while preserving sharp physiological features.")

    # =========================================================================
    # Slide 4: ECG-AIM Model Architecture (30 sec)
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "Methodological Contribution", "ECG-AIM Model Architecture", "Disentangled time-frequency conditioning with axial transformer cross-attention")
    add_footer(s4, 4, "van den Oord et al., VQ-VAE, NeurIPS 2017; Ho et al., Axial Attention, ECCV 2020.")

    arch_path = "slides/figures/conv15e_a0_wave_architecture.png"
    if os.path.exists(arch_path):
        s4.shapes.add_picture(arch_path, Inches(0.8), Inches(1.55), Inches(11.733), Inches(5.05))

    set_notes(s4, "To address this, we developed ECG-AIM. ECG-AIM combines the raw temporal signal with a multiscale wavelet representation, maps these features into a compact latent representation, and jointly reconstructs the missing leads.")

    # =========================================================================
    # Slide 5: Experimental Setup (30 sec)
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "Benchmark Setup", "Experimental Setup", "Rigorous multi-cohort evaluation across diverse clinical populations")
    add_footer(s5, 5, "Wagner et al., PTB-XL, Sci Data 2020; Gow et al., MIMIC-IV-ECG, PhysioNet 2023.")

    w_box = Inches(5.7)
    h_box = Inches(2.2)
    
    # 2x2 Grid
    boxes = [
        ("~175k ECGs", "Trained and benchmarked across diverse cohorts from PTB-XL and MIMIC-IV-ECG", NAVY, Inches(0.8), Inches(1.7)),
        ("Competitive Baselines", "Direct head-to-head comparison against continuous 1D U-Net and MultiScale-VAE", NAVY_LIGHT, Inches(6.833), Inches(1.7)),
        ("1 to 3 Observed Leads", "Evaluated from single wearable Lead I up to 3 complementary leads (I, II, V2)", CYAN, Inches(0.8), Inches(4.2)),
        ("Comprehensive Evaluation", "Evaluated on waveform correlation, QRS timing, and downstream clinical preservation", SUCCESS_GREEN, Inches(6.833), Inches(4.2))
    ]

    for title, desc, col, bx, by in boxes:
        c_box = s5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, bx, by, w_box, h_box)
        c_box.fill.solid()
        c_box.fill.fore_color.rgb = LIGHT_BG
        c_box.line.color.rgb = col
        c_box.line.width = Pt(1.5)
        tb_box = s5.shapes.add_textbox(bx + Inches(0.2), by + Inches(0.2), w_box - Inches(0.4), h_box - Inches(0.4))
        tf_box = tb_box.text_frame
        tf_box.word_wrap = True
        p = tf_box.paragraphs[0]
        p.text = title
        p.font.size = Pt(16)
        p.font.bold = True
        p.font.color.rgb = col
        p.alignment = PP_ALIGN.CENTER
        p2 = tf_box.add_paragraph()
        p2.text = desc
        p2.font.size = Pt(12)
        p2.font.color.rgb = SLATE_DARK
        p2.space_before = Pt(8)
        p2.alignment = PP_ALIGN.CENTER

    set_notes(s5, "We evaluated our framework across 175,000 hospital ECGs from PTB-XL and MIMIC-IV. We tested configurations ranging from one to three observed leads, compared directly against standard continuous 1D U-Net and MultiScale-VAE baselines, and evaluated both raw waveform fidelity and downstream clinical preservation.")

    # =========================================================================
    # Slide 6: Spatial Information Determines Reconstruction Quality (45 sec)
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "Reconstruction Benchmark", "Spatial Information Determines Reconstruction Quality", "Reconstruction quality scales sharply with complementary spatial measurements")
    add_footer(s6, 6, "Frank, Circulation 1956; Atoui et al., IEEE TBME 2010.")

    c6_w = Inches(5.7)
    recon12_path = "slides/figures/best_ecg_aim_12lead_reconstruction.png"
    lead1_path = "slides/figures/best_ecg_aim_lead1_reconstruction.png"

    # Left: 3-lead
    c6_1 = s6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.6), c6_w, Inches(3.9))
    c6_1.fill.solid()
    c6_1.fill.fore_color.rgb = LIGHT_BG
    c6_1.line.color.rgb = SUCCESS_GREEN
    c6_1.line.width = Pt(1.5)
    tb6_1 = s6.shapes.add_textbox(Inches(0.9), Inches(1.65), c6_w - Inches(0.2), Inches(0.4))
    tf6_1 = tb6_1.text_frame
    p = tf6_1.paragraphs[0]
    p.text = "3 Leads Acquired (I, II, V2):  r > 0.98"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = SUCCESS_GREEN
    p.alignment = PP_ALIGN.CENTER
    if os.path.exists(recon12_path):
        s6.shapes.add_picture(recon12_path, Inches(0.95), Inches(2.1), c6_w - Inches(0.3), Inches(3.3))

    # Right: 1-lead
    c6_2 = s6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.833), Inches(1.6), c6_w, Inches(3.9))
    c6_2.fill.solid()
    c6_2.fill.fore_color.rgb = LIGHT_BG
    c6_2.line.color.rgb = NAVY_LIGHT
    c6_2.line.width = Pt(1.5)
    tb6_2 = s6.shapes.add_textbox(Inches(6.933), Inches(1.65), c6_w - Inches(0.2), Inches(0.4))
    tf6_2 = tb6_2.text_frame
    p = tf6_2.paragraphs[0]
    p.text = "1 Lead Acquired (Smartwatch Lead I):  r = 0.776"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = NAVY_LIGHT
    p.alignment = PP_ALIGN.CENTER
    if os.path.exists(lead1_path):
        s6.shapes.add_picture(lead1_path, Inches(6.983), Inches(2.1), c6_w - Inches(0.3), Inches(3.3))

    # Bottom Headline
    c6_bot = s6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(5.65), Inches(11.733), Inches(0.9))
    c6_bot.fill.solid()
    c6_bot.fill.fore_color.rgb = LIGHT_BG
    c6_bot.line.color.rgb = SUCCESS_GREEN
    c6_bot.line.width = Pt(1.5)
    tb6_bot = s6.shapes.add_textbox(Inches(0.95), Inches(5.72), Inches(11.433), Inches(0.75))
    tf6_bot = tb6_bot.text_frame
    tf6_bot.word_wrap = True
    p = tf6_bot.paragraphs[0]
    p.text = "Adding complementary spatial measurements sharply improves reconstruction."
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = SUCCESS_GREEN
    p.alignment = PP_ALIGN.CENTER
    p2 = tf6_bot.add_paragraph()
    p2.text = "Spatially complementary leads provide substantially more information for reconstruction."
    p2.font.size = Pt(11)
    p2.font.color.rgb = SLATE_DARK
    p2.alignment = PP_ALIGN.CENTER

    set_notes(s6, "With three complementary leads, reconstruction approaches the measured signal. With a single wearable-like Lead I, substantial morphology remains recoverable, but performance falls because the missing spatial information cannot be directly observed.")

    # =========================================================================
    # Slide 7: Preservation of Clinically Relevant Structure (45 sec)
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "Clinical Utility", "Preservation of Clinically Relevant Structure", "Validating beyond Euclidean waveform similarity")
    add_footer(s7, 7, "Holste et al., EchoNext Structural Prediction, JACC Adv 2024; Surawicz et al., JACC 2009.")

    c7_w = Inches(3.644)
    h7_col = Inches(3.8)
    
    cols_data = [
        ("Morphology", "0.776", "Precordial Lead r", "(vs. 0.571 U-Net, 0.684 MS-VAE)", NAVY, Inches(0.8)),
        ("Timing", "8.0 ms", "QRS Timing Error", "(Meets < 10.0 ms Clinical Gate)", SUCCESS_GREEN, Inches(4.844)),
        ("Downstream Task", "0.775", "EchoNext Structural AUROC", "(Ground Truth = 0.803)", CYAN, Inches(8.888))
    ]

    for title, val, sub1, sub2, col, left_x in cols_data:
        c_shape = s7.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left_x, Inches(1.65), c7_w, h7_col)
        c_shape.fill.solid()
        c_shape.fill.fore_color.rgb = LIGHT_BG
        c_shape.line.color.rgb = col
        c_shape.line.width = Pt(1.5)
        
        tb = s7.shapes.add_textbox(left_x + Inches(0.15), Inches(1.85), c7_w - Inches(0.3), h7_col - Inches(0.4))
        tf = tb.text_frame
        tf.word_wrap = True
        
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(16)
        p.font.bold = True
        p.font.color.rgb = col
        p.alignment = PP_ALIGN.CENTER
        
        p_val = tf.add_paragraph()
        p_val.text = val
        p_val.font.size = Pt(30)
        p_val.font.bold = True
        p_val.font.color.rgb = col
        p_val.space_before = Pt(14)
        p_val.alignment = PP_ALIGN.CENTER
        
        p_sub1 = tf.add_paragraph()
        p_sub1.text = sub1
        p_sub1.font.size = Pt(12)
        p_sub1.font.bold = True
        p_sub1.font.color.rgb = SLATE_DARK
        p_sub1.space_before = Pt(10)
        p_sub1.alignment = PP_ALIGN.CENTER
        
        p_sub2 = tf.add_paragraph()
        p_sub2.text = sub2
        p_sub2.font.size = Pt(10.5)
        p_sub2.font.color.rgb = SLATE_MUTED
        p_sub2.space_before = Pt(4)
        p_sub2.alignment = PP_ALIGN.CENTER

    # Bottom Headline
    c7_bot = s7.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(5.65), Inches(11.733), Inches(0.9))
    c7_bot.fill.solid()
    c7_bot.fill.fore_color.rgb = LIGHT_BG
    c7_bot.line.color.rgb = SUCCESS_GREEN
    c7_bot.line.width = Pt(1.5)
    tb7_bot = s7.shapes.add_textbox(Inches(0.95), Inches(5.75), Inches(11.433), Inches(0.7))
    tf7_bot = tb7_bot.text_frame
    p = tf7_bot.paragraphs[0]
    p.text = "The reconstruction preserved more than waveform similarity."
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = SUCCESS_GREEN
    p.alignment = PP_ALIGN.CENTER

    set_notes(s7, "Crucially, the reconstruction preserved more than waveform similarity. On precordial morphology, ECG-AIM reaches a correlation of 0.776. For clinical timing, it achieves a QRS duration error of 8.0 milliseconds, meeting the clinical gate. And on downstream tasks, EchoNext structural AUROC reaches 0.775, closely tracking the 0.803 ground truth.")

    # =========================================================================
    # Slide 8: Conclusions and Translational Next Steps (35 sec)
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "Conclusions & Future Work", "Conclusions and Translational Next Steps", "Core insights and prospective clinical validation roadmap")
    add_footer(s8, 8)

    takeaways = [
        ("1. Reconstruction is limited by the spatial information observed.",
         "Single wearable Lead I recovers useful morphology (r = 0.78); 3 complementary leads unlock near-lossless recovery (r > 0.98).",
         NAVY),
        ("2. ECG-AIM preserves substantial morphology and timing from limited leads.",
         "Reconstructed signals meet clinical timing gates (8.0 ms QRS error) and retain downstream diagnostic utility (0.775 EchoNext AUROC).",
         NAVY_LIGHT),
        ("3. Prospective paired smartwatch–12-lead validation is the next test.",
         "Translating from retrospective archives to live clinical deployment requires validating across live hardware factors.",
         SUCCESS_GREEN)
    ]

    t_y = Inches(1.6)
    t_h = Inches(1.15)
    t_gap = Inches(0.2)
    for idx, (title, body, col) in enumerate(takeaways):
        box_y = t_y + idx * (t_h + t_gap)
        c_take = s8.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), box_y, Inches(11.733), t_h)
        c_take.fill.solid()
        c_take.fill.fore_color.rgb = LIGHT_BG
        c_take.line.color.rgb = col
        c_take.line.width = Pt(1.5)
        
        tb = s8.shapes.add_textbox(Inches(1.0), box_y + Inches(0.12), Inches(11.333), t_h - Inches(0.24))
        tf = tb.text_frame
        tf.word_wrap = True
        
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = col
        
        p_body = tf.add_paragraph()
        p_body.text = body
        p_body.font.size = Pt(11)
        p_body.font.color.rgb = SLATE_DARK
        p_body.space_before = Pt(3)

    # Footer metadata card
    tb_meta = s8.shapes.add_textbox(Inches(0.8), Inches(5.8), Inches(11.733), Inches(0.8))
    tf_meta = tb_meta.text_frame
    p_prot = tf_meta.paragraphs[0]
    p_prot.text = "Sunnybrook paired validation protocol in preparation  •  PI: Dr. Christopher Cheung"
    p_prot.font.size = Pt(11)
    p_prot.font.bold = True
    p_prot.font.color.rgb = CYAN
    p_prot.alignment = PP_ALIGN.CENTER
    
    p_cont = tf_meta.add_paragraph()
    p_cont.text = "Mithun Manivannan  •  mithun.manivannan@sri.utoronto.ca  •  Abstract #194"
    p_cont.font.size = Pt(10)
    p_cont.font.color.rgb = SLATE_MUTED
    p_cont.space_before = Pt(3)
    p_cont.alignment = PP_ALIGN.CENTER

    set_notes(s8, "In conclusion: first, reconstruction is limited by the spatial information observed. Second, ECG-AIM preserves substantial morphology and timing from limited leads. Third, prospective paired smartwatch to twelve-lead validation is our next test. Thank you very much for your attention, and I look forward to your questions.")

    # Save presentation
    prs.save(output_path)
    print(f"Presentation successfully saved to {output_path} ({len(prs.slides)} slides)")

if __name__ == "__main__":
    create_deck()
