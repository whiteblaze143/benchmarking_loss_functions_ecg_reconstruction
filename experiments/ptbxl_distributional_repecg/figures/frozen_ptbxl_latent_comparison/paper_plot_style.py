from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

FIG_DIR = Path(__file__).resolve().parent
COLORS = {
    "NORM": "#0072B2",
    "MI": "#D55E00",
    "STTC": "#009E73",
    "CD": "#CC79A7",
    "HYP": "#E69F00",
}

mpl.rcParams.update({
    "font.size": 9,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def save(fig, stem: str) -> None:
    path = FIG_DIR / f"{stem}.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"saved {path}")
