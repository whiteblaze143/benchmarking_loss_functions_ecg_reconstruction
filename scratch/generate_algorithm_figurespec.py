#!/usr/bin/env python3
"""
Generate publication-quality FigureSpec JSON and SVG for the ECG-AIM Training & Inference Algorithm.
"""

import json
import subprocess
from pathlib import Path

spec = {
    "title": "Algorithm Flow: ECG-AIM Forward Training & Biophysical Imputation",
    "canvas": {"width": 1400, "height": 950},
    "style": {
        "font_family": "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
        "font_size": 12,
        "bg_color": "#ffffff",
        "palette": [
            "#3B82F6",  # Blue
            "#8B5CF6",  # Purple
            "#10B981",  # Emerald
            "#F59E0B",  # Amber
            "#EF4444",  # Red
            "#0EA5E9",  # Sky
            "#6366F1",  # Indigo
            "#EC4899"   # Pink
        ]
    },
    "nodes": [
        # Phase 1: Inputs & Pre-processing (x: 180)
        {
            "id": "alg_input",
            "label": "1. Input Initialization",
            "sublabel": "Raw 12-Lead Y [B, 12, 5000]\nSparse Masking -> X_masked [B, 12, 5000]",
            "x": 180, "y": 180,
            "width": 240, "height": 80,
            "shape": "rounded",
            "fill": "#EFF6FF", "stroke": "#3B82F6", "text_color": "#1E3A8A"
        },
        {
            "id": "alg_mask",
            "label": "2. Stochastic Dynamic Masking",
            "sublabel": "Inherited Mask + Strategy ~ {Patch, Span, Lead}\n-> available_mask [B, 12, 200]",
            "x": 180, "y": 320,
            "width": 240, "height": 85,
            "shape": "rounded",
            "fill": "#FDF2F8", "stroke": "#DB2777", "text_color": "#9D174D"
        },
        {
            "id": "alg_norm",
            "label": "3. Unit-RMS Energy Scaling",
            "sublabel": "sigma_RMS = sqrt(mean(X_obs^2)) + 1e-3\nX_norm = X_masked / sigma_RMS",
            "x": 180, "y": 460,
            "width": 240, "height": 80,
            "shape": "rounded",
            "fill": "#F0FDF4", "stroke": "#16A34A", "text_color": "#166534"
        },

        # Phase 2: Tokenization & Encoding (x: 520)
        {
            "id": "alg_patch",
            "label": "4. 1D Patch Tokenization",
            "sublabel": "Patchify(Lp=25, P=200) -> [B, 12, 200, 25]\nLinear + LayerNorm -> H0 [B, 12, 200, 384]",
            "x": 520, "y": 180,
            "width": 250, "height": 80,
            "shape": "rounded",
            "fill": "#E0F2FE", "stroke": "#0284C7", "text_color": "#0369A1"
        },
        {
            "id": "alg_embed",
            "label": "5. Spatiotemporal Embeddings",
            "sublabel": "Z_in = H0 + E_lead + E_time + E_type[0]\nExtract Visible: S = Z_in[available]",
            "x": 520, "y": 320,
            "width": 250, "height": 80,
            "shape": "rounded",
            "fill": "#EDE9FE", "stroke": "#8B5CF6", "text_color": "#4C1D95"
        },
        {
            "id": "alg_encoder",
            "label": "6. Asymmetric Encoder",
            "sublabel": "Pad visible sequences to N_max <= 600\nMemory = TransformerEncoder(Padded, 8 L)",
            "x": 520, "y": 460,
            "width": 250, "height": 80,
            "shape": "rounded",
            "fill": "#F5F3FF", "stroke": "#7C3AED", "text_color": "#5B21B6"
        },

        # Phase 3: Grid Assembly & Biophysical Seeding (x: 870)
        {
            "id": "alg_grid",
            "label": "7. Latent Grid Reassembly",
            "sublabel": "Init Grid = [MASK] + E_lead + E_time\nInsert Memory[available] -> [B, 12, 200, 384]",
            "x": 870, "y": 180,
            "width": 260, "height": 80,
            "shape": "rounded",
            "fill": "#FEF3C7", "stroke": "#D97706", "text_color": "#92400E"
        },
        {
            "id": "alg_einthoven",
            "label": "8. Einthoven Biophysical Seeding",
            "sublabel": "Compute Baseline B: III = II - I, aVR/L/F\nGrid = Grid + Linear_baseline(B)",
            "x": 870, "y": 320,
            "width": 260, "height": 80,
            "shape": "rounded",
            "fill": "#ECFDF5", "stroke": "#059669", "text_color": "#065F46"
        },
        {
            "id": "alg_decoder",
            "label": "9. Axial Transformer Decoding",
            "sublabel": "For block = 1 to 4:\n  SpatialAttn(12) -> TimeAttn(200) -> FFN",
            "x": 870, "y": 460,
            "width": 260, "height": 80,
            "shape": "rounded",
            "fill": "#FFFBEB", "stroke": "#F59E0B", "text_color": "#78350F"
        },

        # Phase 4: Output Synthesis & Loss Engine (x: 1210)
        {
            "id": "alg_synth",
            "label": "10. Waveform Synthesis",
            "sublabel": "Residual = Linear_head(LN(Grid))\nY_pred = (B + gamma * Res) * sigma_RMS",
            "x": 1210, "y": 180,
            "width": 240, "height": 80,
            "shape": "rounded",
            "fill": "#DCFCE7", "stroke": "#10B981", "text_color": "#065F46"
        },
        {
            "id": "alg_loss",
            "label": "11. 7-Factorial Loss Engine",
            "sublabel": "L_masked + L_cons +\nComposite(MSE, Corr, Deriv, VCG, MMD)",
            "x": 1210, "y": 320,
            "width": 240, "height": 80,
            "shape": "rounded",
            "fill": "#FEF2F2", "stroke": "#EF4444", "text_color": "#991B1B"
        },
        {
            "id": "alg_backward",
            "label": "12. Optimization Step",
            "sublabel": "Scaler.backward() -> Grad Clip (1.0)\nAdamW Step (lr=3e-4, wd=1e-4)",
            "x": 1210, "y": 460,
            "width": 240, "height": 80,
            "shape": "rounded",
            "fill": "#F1F5F9", "stroke": "#475569", "text_color": "#0F172A"
        }
    ],
    "edges": [
        # Col 1 -> Col 2
        {"from": "alg_input", "to": "alg_mask", "color": "#3B82F6", "thickness": 2},
        {"from": "alg_mask", "to": "alg_norm", "color": "#DB2777", "thickness": 2},
        {"from": "alg_norm", "to": "alg_patch", "color": "#16A34A", "thickness": 2, "curve": True},

        # Col 2 -> Col 3
        {"from": "alg_patch", "to": "alg_embed", "color": "#0284C7", "thickness": 2},
        {"from": "alg_embed", "to": "alg_encoder", "color": "#8B5CF6", "thickness": 2},
        {"from": "alg_encoder", "to": "alg_grid", "color": "#7C3AED", "thickness": 2, "curve": True},

        # Col 3 -> Col 4
        {"from": "alg_grid", "to": "alg_einthoven", "color": "#D97706", "thickness": 2},
        {"from": "alg_einthoven", "to": "alg_decoder", "color": "#059669", "thickness": 2},
        {"from": "alg_decoder", "to": "alg_synth", "color": "#F59E0B", "thickness": 2, "curve": True},

        # Col 4 Finish
        {"from": "alg_synth", "to": "alg_loss", "color": "#10B981", "thickness": 2},
        {"from": "alg_loss", "to": "alg_backward", "color": "#EF4444", "thickness": 2}
    ],
    "groups": [
        {
            "id": "grp_p1",
            "label": "Stage I: Ingestion & Masking",
            "node_ids": ["alg_input", "alg_mask", "alg_norm"],
            "fill": "#FAFAFA", "stroke": "#E2E8F0", "padding": 25
        },
        {
            "id": "grp_p2",
            "label": "Stage II: Tokenization & Encoding",
            "node_ids": ["alg_patch", "alg_embed", "alg_encoder"],
            "fill": "#FAFAFA", "stroke": "#E2E8F0", "padding": 25
        },
        {
            "id": "grp_p3",
            "label": "Stage III: Biophysical Grid & Axial Decoding",
            "node_ids": ["alg_grid", "alg_einthoven", "alg_decoder"],
            "fill": "#FAFAFA", "stroke": "#E2E8F0", "padding": 25
        },
        {
            "id": "grp_p4",
            "label": "Stage IV: Synthesis & Loss Supervision",
            "node_ids": ["alg_synth", "alg_loss", "alg_backward"],
            "fill": "#FAFAFA", "stroke": "#E2E8F0", "padding": 25
        }
    ]
}

# Write FigureSpec JSON
json_path = Path("results/figures/ecg_aim_algorithm_spec.json")
json_path.parent.mkdir(parents=True, exist_ok=True)
with open(json_path, "w") as f:
    json.dump(spec, f, indent=2)
print(f"Saved FigureSpec JSON to {json_path}")

# Render to SVG
svg_path = Path("results/figures/ecg_aim_algorithm_diagram.svg")
cmd = [
    "python3",
    ".agents/skills/figure-spec/scripts/figure_renderer.py",
    "render",
    str(json_path),
    "--output",
    str(svg_path)
]
subprocess.run(cmd, check=True)
print(f"Successfully rendered publication-quality ECG-AIM Algorithm SVG to {svg_path}!")

# Also copy to artifacts directory for instant embedding
artifact_svg = Path("/home/mithunmanivannan/.gemini/antigravity-ide/brain/df14c00e-f738-4b5c-866b-9f8e43bebaa5/ecg_aim_algorithm_diagram.svg")
with open(svg_path, "r") as src, open(artifact_svg, "w") as dst:
    dst.write(src.read())
print(f"Copied SVG to artifact path: {artifact_svg}")
