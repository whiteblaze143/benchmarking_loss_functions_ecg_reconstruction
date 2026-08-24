#!/usr/bin/env python3
"""
Generate publication-quality FigureSpec JSON and SVG for the 3-Architecture Factorial ECG Reconstruction System.
"""

import json
import subprocess
from pathlib import Path

spec = {
    "title": "3-Architecture Factorial 12-Lead ECG Reconstruction & Clinical Audit Benchmark",
    "canvas": {"width": 1420, "height": 820},
    "style": {
        "font_family": "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
        "font_size": 13,
        "bg_color": "#F8FAFC",
        "palette": [
            "#3B82F6",  # Blue
            "#8B5CF6",  # Purple
            "#10B981",  # Emerald
            "#F59E0B",  # Amber
            "#EF4444",  # Red
            "#0EA5E9",  # Sky
            "#6366F1"   # Indigo
        ]
    },
    "nodes": [
        # --- Column 1: Input Space (x: 140) ---
        {
            "id": "input_acquisition",
            "label": "3D Orthogonal Lead\nInput (I, II, V2)",
            "sublabel": "X (I), Y (II), Z (V2) Dipoles",
            "x": 140, "y": 240,
            "width": 180, "height": 70,
            "shape": "rounded",
            "fill": "#EFF6FF", "stroke": "#3B82F6", "text_color": "#1E3A8A"
        },
        {
            "id": "input_canvas",
            "label": "12-Lead Sparse Canvas\n[B, 12, 5000]",
            "sublabel": "9 Unobserved Leads Masked",
            "x": 140, "y": 420,
            "width": 180, "height": 70,
            "shape": "rounded",
            "fill": "#F1F5F9", "stroke": "#64748B", "text_color": "#0F172A"
        },
        {
            "id": "einthoven_prior",
            "label": "Einthoven Physics Prior\n(III, aVR, aVL, aVF)",
            "sublabel": "Deterministic Kirchhoff Loops",
            "x": 140, "y": 600,
            "width": 180, "height": 70,
            "shape": "rounded",
            "fill": "#FEF3C7", "stroke": "#F59E0B", "text_color": "#78350F"
        },

        # --- Column 2: Architectures (x: 480) ---
        {
            "id": "arch_unet",
            "label": "Deterministic U-Net\n(1D Conv MCMA)",
            "sublabel": "Skip-Connections | Severe R2M",
            "x": 480, "y": 200,
            "width": 220, "height": 80,
            "shape": "rounded",
            "fill": "#FEE2E2", "stroke": "#EF4444", "text_color": "#991B1B"
        },
        {
            "id": "arch_msvae",
            "label": "MultiScale-VAE (WearECG)\n(Hierarchical ResNet + Attention)",
            "sublabel": "q(z|x) ∈ ℝ^{4×625} | 41.8% Var Ret",
            "x": 480, "y": 420,
            "width": 230, "height": 85,
            "shape": "rounded",
            "fill": "#EDE9FE", "stroke": "#8B5CF6", "text_color": "#4C1D95"
        },
        {
            "id": "arch_ecgaim",
            "label": "ECG-AIM\n(Axial Spatio-Temporal MAE)",
            "sublabel": "Space×Time Attention (12×200)",
            "x": 480, "y": 640,
            "width": 220, "height": 80,
            "shape": "rounded",
            "fill": "#DCFCE7", "stroke": "#10B981", "text_color": "#065F46"
        },

        # --- Column 3: Factorial Loss Suite (x: 840) ---
        {
            "id": "loss_time",
            "label": "Waveform & Derivative Losses\n(LMSE + LCorr + LDeriv)",
            "sublabel": "Phase & High-Freq QRS Alignment",
            "x": 840, "y": 220,
            "width": 230, "height": 75,
            "shape": "rounded",
            "fill": "#E0F2FE", "stroke": "#0284C7", "text_color": "#0369A1"
        },
        {
            "id": "loss_vcg",
            "label": "Kors 3D VCG Dipole Loss\n(K_Kors ∈ ℝ^{3×8})",
            "sublabel": "Spatial Cosine Loop Orientation",
            "x": 840, "y": 420,
            "width": 230, "height": 75,
            "shape": "rounded",
            "fill": "#FDF4FF", "stroke": "#C026D3", "text_color": "#86198F"
        },
        {
            "id": "loss_mmd",
            "label": "Multiscale MMD Regularizer\n(IMQ / Temporal RKHS Kernels)",
            "sublabel": "Non-Parametric Energy Matching",
            "x": 840, "y": 620,
            "width": 230, "height": 75,
            "shape": "rounded",
            "fill": "#ECFDF5", "stroke": "#059669", "text_color": "#065F46"
        },

        # --- Column 4: 212-Endpoint Clinical Verification (x: 1200) ---
        {
            "id": "eval_macro",
            "label": "Macro Diagnostic AUROC\n(PTB-XL + EchoNext)",
            "sublabel": ">96% Global Retention (AFib, LBBB)",
            "x": 1200, "y": 200,
            "width": 210, "height": 75,
            "shape": "rounded",
            "fill": "#F0FDF4", "stroke": "#22C55E", "text_color": "#15803D"
        },
        {
            "id": "eval_calipers",
            "label": "Continuous Calipers & LVH\n(Sunnybrook, LUDB, ISP)",
            "sublabel": "QRS MAE (14ms) | Sokolow LVH aOR",
            "x": 1200, "y": 420,
            "width": 210, "height": 75,
            "shape": "rounded",
            "fill": "#FFFBEB", "stroke": "#F59E0B", "text_color": "#B45309"
        },
        {
            "id": "eval_presacan",
            "label": "Presacan Bland-Altman Audit\n(R² & Variance Collapse)",
            "sublabel": "U-Net R²=0.94 vs MS-VAE R²=0.58",
            "x": 1200, "y": 640,
            "width": 210, "height": 75,
            "shape": "rounded",
            "fill": "#FEF2F2", "stroke": "#EF4444", "text_color": "#B91C1C"
        }
    ],
    "edges": [
        # Input -> Canvas
        {"from": "input_acquisition", "to": "input_canvas", "label": "3-Lead Anchor", "color": "#3B82F6", "thickness": 2},
        {"from": "input_acquisition", "to": "einthoven_prior", "label": "Limb Geometry", "color": "#F59E0B", "style": "dashed", "thickness": 2},

        # Canvas -> Architectures
        {"from": "input_canvas", "to": "arch_unet", "label": "Zero-Filled", "color": "#64748B", "thickness": 2},
        {"from": "input_canvas", "to": "arch_msvae", "label": "Sparse Waveform", "color": "#8B5CF6", "thickness": 2},
        {"from": "input_canvas", "to": "arch_ecgaim", "label": "2D Patches", "color": "#10B981", "thickness": 2},
        {"from": "einthoven_prior", "to": "arch_ecgaim", "label": "Biophysical Bias", "color": "#F59E0B", "thickness": 2},

        # Architectures -> Loss Suite
        {"from": "arch_unet", "to": "loss_time", "color": "#EF4444", "thickness": 2},
        {"from": "arch_msvae", "to": "loss_vcg", "color": "#8B5CF6", "thickness": 2},
        {"from": "arch_msvae", "to": "loss_mmd", "color": "#8B5CF6", "thickness": 2},
        {"from": "arch_ecgaim", "to": "loss_time", "color": "#10B981", "thickness": 2},
        {"from": "arch_ecgaim", "to": "loss_mmd", "color": "#10B981", "thickness": 2},

        # Loss Suite -> Clinical Evaluation Cascade
        {"from": "loss_time", "to": "eval_macro", "label": "Rhythm Alignment", "color": "#0284C7", "thickness": 2},
        {"from": "loss_vcg", "to": "eval_calipers", "label": "Spatial Axis", "color": "#C026D3", "thickness": 2},
        {"from": "loss_mmd", "to": "eval_presacan", "label": "Distribution Fit", "color": "#059669", "thickness": 2}
    ],
    "groups": [
        {
            "id": "grp_input",
            "label": "1. 3D Physical Input Space",
            "node_ids": ["input_acquisition", "input_canvas", "einthoven_prior"],
            "fill": "#F8FAFC", "stroke": "#CBD5E1", "padding": 25
        },
        {
            "id": "grp_arch",
            "label": "2. Tri-Architecture Generative Framework (480 Models)",
            "node_ids": ["arch_unet", "arch_msvae", "arch_ecgaim"],
            "fill": "#F8FAFC", "stroke": "#CBD5E1", "padding": 25
        },
        {
            "id": "grp_loss",
            "label": "3. 160-Configuration Combinatorial Factorial Loss Suite",
            "node_ids": ["loss_time", "loss_vcg", "loss_mmd"],
            "fill": "#F8FAFC", "stroke": "#CBD5E1", "padding": 25
        },
        {
            "id": "grp_eval",
            "label": "4. Zero-Context Clinical Verification Cascade (212 Endpoints, 6 Datasets)",
            "node_ids": ["eval_macro", "eval_calipers", "eval_presacan"],
            "fill": "#F8FAFC", "stroke": "#CBD5E1", "padding": 25
        }
    ]
}

# Write FigureSpec JSON
json_path = Path("results/figures/architecture_spec.json")
json_path.parent.mkdir(parents=True, exist_ok=True)
with open(json_path, "w") as f:
    json.dump(spec, f, indent=2)
print(f"Saved FigureSpec JSON to {json_path}")

# Render to SVG
svg_path = Path("results/figures/architecture_diagram.svg")
cmd = [
    "python3",
    ".agents/skills/figure-spec/scripts/figure_renderer.py",
    "render",
    str(json_path),
    "--output",
    str(svg_path)
]
subprocess.run(cmd, check=True)
print(f"Successfully rendered publication-quality architecture SVG to {svg_path}!")

# Also copy to artifacts directory for instant embedding
artifact_svg = Path("/home/mithunmanivannan/.gemini/antigravity-ide/brain/df14c00e-f738-4b5c-866b-9f8e43bebaa5/architecture_diagram.svg")
with open(svg_path, "r") as src, open(artifact_svg, "w") as dst:
    dst.write(src.read())
print(f"Copied SVG to artifact path: {artifact_svg}")
