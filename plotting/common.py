import matplotlib.pyplot as plt
from pathlib import Path

# Tool name constant
TOOL = "ProToken"

# Model name mappings (keys match sanitized exp_key form, e.g. underscores not slashes)
MODEL_NAMES = {
    "google_gemma-3-270m-it": "Gemma",
    "HuggingFaceTB_SmolLM2-360M-Instruct": "SmolLM",
    "meta-llama_Llama-3.2-1B-Instruct": "Llama",
    "Qwen_Qwen2.5-0.5B-Instruct": "Qwen",
}

# Domain name mappings
DOMAIN_NAMES = {
    "medical": "Medical",
    "math": "Math",
    "finance": "Finance",
    "coding": "Coding",
}

# Color palette for plots
COLORS = {
    "attribution": "Blue",
    "token_acc": "#E63946",
    "malicious": "#DC2F02",
    "benign": "#0077B6"
}

# Output directory
OUTPUT_DIR = Path("paper/graphs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def setup_plot_style(font_size: int = 14):
    plt.rcParams.update({
        "font.size": font_size,
        "axes.titlesize": font_size + 4 if font_size <= 14 else font_size,
        "axes.labelsize": font_size + 2 if font_size <= 14 else font_size,
        "xtick.labelsize": font_size - 2 if font_size > 14 else font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": font_size,
        "figure.dpi": 300,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
    })


def save_figure(fig, filename: str, output_dir: Path = OUTPUT_DIR):
  
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_pdf = output_dir / f"{filename}.pdf"
    output_png = output_dir / f"{filename}.png"
    
    fig.savefig(output_pdf, bbox_inches='tight', dpi=300)
    fig.savefig(output_png, bbox_inches='tight', dpi=300)
    
    print(f"Saved {filename} to {output_pdf} and {output_png}")
    plt.close(fig)


def apply_axis_aesthetics(ax, xlabel: str = "", ylabel: str = "", 
                         row: int = 0, col: int = 0, 
                         nrows: int = 2, ncols: int = 4):
   
    # Only show x-label on bottom row
    if row == nrows - 1 and xlabel:
        ax.set_xlabel(xlabel, fontweight='bold')
    elif xlabel:
        ax.set_xlabel("")
    
    # Only show y-label on leftmost column
    if col == 0 and ylabel:
        ax.set_ylabel(ylabel, fontweight='bold')
    elif ylabel:
        ax.set_ylabel("")
    
    # Add minor ticks with consistent styling
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', direction='in', 
                  length=8, width=2, top=True, right=True)
    ax.tick_params(axis='both', which='minor', direction='in', 
                  length=4, width=1.5, top=True, right=True)



    