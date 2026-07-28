import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import numpy as np

# Define tasks: (Task Name, Lead/Type, Start Date YYYY-MM-DD, End Date YYYY-MM-DD, Phase)
tasks_data = [
    ("Project Setup & CI/CD", "Joint", "2026-07-28", "2026-07-30", "Week 1: Baselines"),
    ("Dataset Loader & ROI Crop", "Kushagra", "2026-07-28", "2026-07-31", "Week 1: Baselines"),
    ("MLEP Preprocessing Pipeline", "Kushagra", "2026-07-30", "2026-08-03", "Week 1: Baselines"),
    ("LOTA Preprocessing Pipeline", "Aishwarya", "2026-07-30", "2026-08-03", "Week 1: Baselines"),
    ("Milestone 1: Baseline Review & Merge", "Milestone", "2026-08-03", "2026-08-04", "Week 1: Baselines"),

    ("ResNet Backbones & Stems", "Kushagra", "2026-08-04", "2026-08-07", "Week 2: DL Architectures"),
    ("Arch I: Freq Filter & SupCon", "Kushagra", "2026-08-06", "2026-08-10", "Week 2: DL Architectures"),
    ("Arch II: MGA-Net Attention", "Aishwarya", "2026-08-05", "2026-08-10", "Week 2: DL Architectures"),
    ("Milestone 2: Joint Architecture Merge", "Milestone", "2026-08-10", "2026-08-11", "Week 2: DL Architectures"),

    ("Arch III: MoE & DANN", "Kushagra", "2026-08-11", "2026-08-15", "Week 3: Generalization & Eval"),
    ("Evaluation Engine & Metrics", "Aishwarya", "2026-08-11", "2026-08-15", "Week 3: Generalization & Eval"),
    ("Grad-CAM & Figure Scripts", "Aishwarya", "2026-08-12", "2026-08-15", "Week 3: Generalization & Eval"),

    ("End-to-End Model Assembly", "Joint", "2026-08-15", "2026-08-16", "Final Integration Sprint"),
    ("Full Training & Debugging", "Joint", "2026-08-16", "2026-08-17", "Final Integration Sprint"),
    ("Benchmarking & Doc Polish", "Joint", "2026-08-17", "2026-08-18", "Final Integration Sprint"),
    ("Final Sign-Off & Delivery", "Milestone", "2026-08-18", "2026-08-18.5", "Final Integration Sprint")
]

# Color Palette
colors = {
    "Kushagra": "#3fb950",   # Green (Lenovo RTX 4050 CUDA / Intel i5)
    "Aishwarya": "#a371f7",  # Purple (MacBook Air M4 Unified Memory)
    "Joint": "#58a6ff",      # Blue
    "Milestone": "#f85149"   # Red/Orange
}

# Setup Figure in Modern Dark Theme
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(16, 10), facecolor='#0d1117')
ax.set_facecolor('#161b22')

# Process tasks in reverse order for top-to-bottom display
y_pos = np.arange(len(tasks_data))
y_labels = []

for i, (name, lead, start_str, end_str, phase) in enumerate(reversed(tasks_data)):
    start_date = datetime.strptime(start_str, "%Y-%m-%d")
    if end_str.endswith(".5"):
        end_date = datetime.strptime(end_str[:-2], "%Y-%m-%d") + timedelta(hours=12)
    else:
        end_date = datetime.strptime(end_str, "%Y-%m-%d")
    
    duration = (end_date - start_date).total_seconds() / (24 * 3600)
    if duration == 0:
        duration = 0.5
        
    color = colors[lead]
    
    # Plot horizontal bar
    bar = ax.barh(i, duration, left=start_date, height=0.65, color=color, edgecolor='#30363d', linewidth=1.2, alpha=0.9)
    
    # Label inside or next to bar
    label_text = f"  {name} ({lead})"
    text_color = '#ffffff' if lead != "Joint" else '#ffffff'
    
    # Position text inside bar if wide enough, else to the right
    if duration >= 2.0:
        ax.text(start_date + timedelta(days=duration/2), i, f"{name}", 
                va='center', ha='center', color=text_color, fontweight='bold', fontsize=10)
    else:
        ax.text(end_date + timedelta(days=0.2), i, f"{name} [{lead}]", 
                va='center', ha='left', color=colors[lead], fontweight='bold', fontsize=10)
        
    y_labels.append(f"[{phase.split(':')[0]}] {name}")

# Formatting X and Y axes
ax.set_yticks(y_pos)
ax.set_yticklabels(y_labels, fontsize=11, color='#c9d1d9', fontweight='medium')

# Format X axis dates
ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %02d'))
ax.tick_params(axis='x', rotation=0, labelsize=11, colors='#8b949e')
ax.tick_params(axis='y', colors='#c9d1d9')

# Add vertical grid lines and milestone boundary lines
ax.grid(axis='x', color='#30363d', linestyle='--', alpha=0.7)
ax.set_axisbelow(True)

# Add vertical lines for weeks
week_boundaries = ["2026-08-03", "2026-08-10", "2026-08-15", "2026-08-18"]
for wb in week_boundaries:
    dt = datetime.strptime(wb, "%Y-%m-%d")
    ax.axvline(x=dt, color='#8b949e', linestyle=':', linewidth=1.5, alpha=0.8)
    ax.text(dt, len(tasks_data) - 0.3, f"  {wb}", color='#8b949e', fontsize=9, rotation=90, va='top')

# Create Custom Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=colors["Kushagra"], edgecolor='#30363d', label='Kushagra (Lenovo RTX 4050 CUDA / Intel i5)'),
    Patch(facecolor=colors["Aishwarya"], edgecolor='#30363d', label='Aishwarya (MacBook Air M4 Unified Memory)'),
    Patch(facecolor=colors["Joint"], edgecolor='#30363d', label='Joint / Synchronous Collaboration'),
    Patch(facecolor=colors["Milestone"], edgecolor='#30363d', label='Critical Milestone Sign-Off')
]
ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.08), 
          ncol=2, frameon=True, facecolor='#161b22', edgecolor='#30363d', fontsize=11)

# Title and subtitle
plt.title("MLEP & LOTA Fusion: 3-Week Engineering Sprint Schedule\n(28 July 2026 – 18 August 2026)", 
          fontsize=18, fontweight='bold', color='#58a6ff', pad=45)

# Add borders
for spine in ax.spines.values():
    spine.set_color('#30363d')

plt.tight_layout()

# Save image at 300 DPI
output_path = "/Volumes/Seagate/JIO TERM/JIO-TERM 3/DL AND CV PROJECT/MLEP_LOTA_Gantt_Chart.png"
plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
print(f"Successfully generated Gantt chart image at: {output_path}")
