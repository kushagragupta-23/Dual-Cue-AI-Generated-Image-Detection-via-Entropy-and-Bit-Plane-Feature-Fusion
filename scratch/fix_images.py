import re

with open('d:/MAIN PROJECT CV AND DL/HydraFusion/scripts/generate_html_report.py', 'r', encoding='utf-8') as f:
    text = f.read()

replacement = '''        "calibration_curve": img_to_base64(batch_vis_dir / "calibration_curve.png"),
        "performance_summary": img_to_base64(batch_vis_dir / "performance_summary.png"),
        "gating_weights": img_to_base64(batch_vis_dir / "gating_weights.png"),'''
text = re.sub(r'"calibration_curve":.*?,', replacement, text)

# Also fix the `metrics` reference inside TabLOTA and TabComparative
text = text.replace("{metrics.get('figures', {}).get('lota_acc', '')}", "{images.get('performance_summary', '')}")
text = text.replace("{metrics.get('figures', {}).get('lota_loss', '')}", "{images.get('gating_weights', '')}")
text = text.replace('alt="LOTA Accuracy Curve"', 'alt="Performance Summary"').replace("LOTA Standalone Training & Validation Accuracy", "Performance Summary Comparison")
text = text.replace('alt="LOTA Loss Curve"', 'alt="Gating Weights"').replace("LOTA Standalone Training & Validation Loss", "Gating Weights Distribution")

with open('d:/MAIN PROJECT CV AND DL/HydraFusion/scripts/generate_html_report.py', 'w', encoding='utf-8') as f:
    f.write(text)
print('Done!')
