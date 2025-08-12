from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import math


def write_metrics_report(metrics: Dict[str, Any], output_dir: Path) -> str:
    """Save a simple markdown metrics report for dissertation demo."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = output_dir / f"model_metrics_{ts}.md"
    lines = ["# Model Evaluation Metrics", ""]
    for k, v in metrics.items():
        lines.append(f"- **{k}**: {v}")
    report_file.write_text("\n".join(lines), encoding="utf-8")
    return str(report_file)


def write_comparative_metrics_report(
    metrics: Dict[str, Any],
    output_dir: Path,
) -> str:
    """Write a richer markdown report including baseline comparison and significance stats.

    Expected keys in metrics (best-effort; handle missing):
      - dataset, size, accuracy, precision, recall, f1, model_source
      - baseline_* (optional)
      - mcnemar_b, mcnemar_c, mcnemar_chi2, mcnemar_p (optional)
      - shap_samples_json / shap_samples_markdown (optional paths)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = output_dir / f"model_comparative_metrics_{ts}.md"

    lines: List[str] = []
    lines.append("# Model Evaluation (Research Report)")
    lines.append("")
    lines.append(f"- **Generated**: {datetime.now().isoformat()}")
    if "dataset" in metrics:
        lines.append(f"- **Dataset**: {metrics['dataset']}")
    if "size" in metrics:
        lines.append(f"- **Evaluated Samples**: {metrics['size']}")
    if "model_source" in metrics:
        lines.append(f"- **Model Source**: `{metrics['model_source']}`")
    lines.append("")

    # Primary metrics
    lines.append("## Primary Metrics")
    for key in ("accuracy", "precision", "recall", "f1"):
        if key in metrics:
            lines.append(f"- **{key.title()}**: {metrics[key]}")
    lines.append("")

    # Baseline comparison
    has_baseline = any(k.startswith("baseline_") for k in metrics.keys())
    if has_baseline:
        lines.append("## Baseline Comparison (Heuristic)")
        for key in ("baseline_accuracy", "baseline_precision", "baseline_recall", "baseline_f1"):
            if key in metrics:
                lines.append(f"- **{key.replace('baseline_', 'Baseline ').title()}**: {metrics[key]}")
        lines.append("")

    # Statistical significance
    if all(k in metrics for k in ("mcnemar_b", "mcnemar_c", "mcnemar_chi2", "mcnemar_p")):
        b = metrics.get("mcnemar_b", 0)
        c = metrics.get("mcnemar_c", 0)
        chi2 = metrics.get("mcnemar_chi2", 0.0)
        pval = metrics.get("mcnemar_p", 1.0)
        lines.append("## Statistical Significance (McNemar's Test)")
        lines.append("- Compares model vs baseline correctness on the same samples.")
        lines.append("- Contingency (disagreements only):")
        lines.append(f"  - b = model correct, baseline wrong: {b}")
        lines.append(f"  - c = model wrong, baseline correct: {c}")
        lines.append(f"- Chi-square (with continuity correction): {chi2:.4f}")
        lines.append(f"- p-value (df=1): {pval:.6f}")
        if pval < 0.05:
            lines.append("- Conclusion: Difference is statistically significant at alpha=0.05.")
        else:
            lines.append("- Conclusion: No statistically significant difference at alpha=0.05.")
        lines.append("")

    # Calibration & discrimination
    if any(k in metrics for k in ("brier_score", "ece", "roc_auc", "pr_auc")):
        lines.append("## Calibration and Discrimination")
        if "brier_score" in metrics:
            lines.append(f"- Brier Score: {metrics['brier_score']}")
        if "ece" in metrics:
            lines.append(f"- Expected Calibration Error (ECE): {metrics['ece']}")
        if "roc_auc" in metrics:
            lines.append(f"- ROC-AUC: {metrics['roc_auc']}")
        if "pr_auc" in metrics:
            lines.append(f"- PR-AUC (Average Precision): {metrics['pr_auc']}")
        # Reliability table
        if isinstance(metrics.get("reliability_bins"), list) and metrics["reliability_bins"]:
            lines.append("")
            lines.append("### Reliability (Confidence vs Accuracy)")
            lines.append("| Bin | Range | Count | Avg Confidence | Accuracy |")
            lines.append("|-----|-------|-------|----------------|----------|")
            for b in metrics["reliability_bins"]:
                rng = f"[{b['low']:.1f}, {b['high']:.1f}]"
                count = b.get("count", 0)
                avg_conf = ("-" if b.get("avg_conf") is None else f"{b['avg_conf']:.3f}")
                acc = ("-" if b.get("accuracy") is None else f"{b['accuracy']:.3f}")
                lines.append(f"| {b['bin']} | {rng} | {count} | {avg_conf} | {acc} |")
        lines.append("")

    # Robustness
    if any(k in metrics for k in ("robustness_accuracy", "prediction_stability", "robustness_delta")):
        lines.append("## Robustness (Text Perturbations)")
        if "robustness_accuracy" in metrics:
            lines.append(f"- Accuracy under perturbations: {metrics['robustness_accuracy']}")
        if "robustness_delta" in metrics:
            lines.append(f"- Accuracy delta (original - perturbed): {metrics['robustness_delta']}")
        if "prediction_stability" in metrics:
            lines.append(f"- Prediction stability (unchanged fraction): {metrics['prediction_stability']}")
        lines.append("")

    # Coverage-Accuracy
    if isinstance(metrics.get("coverage_accuracy"), list) and metrics["coverage_accuracy"]:
        lines.append("## Coverage vs Accuracy (Abstention using confidence margin)")
        lines.append("| Coverage | Accuracy | N |")
        lines.append("|----------|----------|---|")
        for row in metrics["coverage_accuracy"]:
            lines.append(f"| {row['coverage']:.2f} | {row['accuracy']:.4f} | {row['n']} |")
        lines.append("")

    # MC Dropout Uncertainty summary
    if "mutual_information_mean" in metrics or "mc_dropout_samples" in metrics:
        lines.append("## Epistemic Uncertainty (MC Dropout)")
        if "mc_dropout_samples" in metrics:
            lines.append(f"- MC samples: {metrics['mc_dropout_samples']}")
        if "mutual_information_mean" in metrics:
            lines.append(f"- Mutual Information (mean): {metrics['mutual_information_mean']}")
        if "mutual_information_std" in metrics:
            lines.append(f"- Mutual Information (std): {metrics['mutual_information_std']}")
        lines.append("")

    # Explainability Quality (Faithfulness via Deletion)
    if "explainability_quality_topk_delta_mean" in metrics:
        lines.append("## Explainability Quality (Deletion Faithfulness)")
        lines.append(f"- Top-k removal delta (mean): {metrics['explainability_quality_topk_delta_mean']}")
        if "explainability_quality_random_delta_mean" in metrics:
            lines.append(f"- Random removal delta (mean): {metrics['explainability_quality_random_delta_mean']}")
        if "explainability_quality_effect" in metrics:
            lines.append(f"- Effect (Top-k - Random): {metrics['explainability_quality_effect']}")
        if "explainability_quality_samples" in metrics:
            lines.append(f"- Samples: {metrics['explainability_quality_samples']}")
        lines.append("")

    # SHAP artifacts if present
    if "shap_samples_markdown" in metrics or "shap_samples_json" in metrics:
        lines.append("## Explainability Artifacts")
        if metrics.get("shap_samples_markdown"):
            lines.append(f"- SHAP Markdown Preview: `{metrics['shap_samples_markdown']}`")
        if metrics.get("shap_samples_json"):
            lines.append(f"- SHAP JSON Samples: `{metrics['shap_samples_json']}`")
        lines.append("")

    # Fairness across groups
    if isinstance(metrics.get("fairness_groups"), dict) and metrics["fairness_groups"]:
        lines.append("## Fairness Across Subgroups")
        lines.append("| Group | Accuracy | N |")
        lines.append("|-------|----------|---|")
        for g, vals in metrics["fairness_groups"].items():
            lines.append(f"| {g} | {vals.get('accuracy','-')} | {vals.get('n','-')} |")
        if "fairness_accuracy_gap" in metrics:
            lines.append("")
            lines.append(f"- Accuracy gap (max - min): {metrics['fairness_accuracy_gap']}")
        lines.append("")

    # Temporal stability
    if isinstance(metrics.get("temporal_periods"), dict) and metrics["temporal_periods"]:
        lines.append("## Temporal Stability")
        lines.append("| Period | Accuracy | N |")
        lines.append("|--------|----------|---|")
        for k, vals in metrics["temporal_periods"].items():
            lines.append(f"| {k} | {vals.get('accuracy','-')} | {vals.get('n','-')} |")
        if "temporal_accuracy_std" in metrics:
            lines.append("")
            lines.append(f"- Accuracy std across periods: {metrics['temporal_accuracy_std']}")
        lines.append("")

    # Attention analysis summary
    if "attention_topk_mean" in metrics:
        lines.append("## Attention Analysis")
        lines.append(f"- Mean of top-k attention weights over sample: {metrics['attention_topk_mean']}")
        if "attention_samples" in metrics:
            lines.append(f"- Samples: {metrics['attention_samples']}")
        lines.append("")

    report_file.write_text("\n".join(lines), encoding="utf-8")
    return str(report_file)


def write_cross_domain_report(
    results: Dict[str, Any],
    output_dir: Path,
) -> str:
    """Write a cross-domain evaluation summary report.

    results: {
      'evaluations': [ { 'dataset': name, 'size': n, 'accuracy': ..., 'f1': ..., 'baseline_accuracy': ...?, ... }, ... ]
    }
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = output_dir / f"cross_domain_evaluation_{ts}.md"

    lines: List[str] = []
    lines.append("# Cross-Domain Evaluation Summary")
    lines.append("")
    lines.append(f"- **Generated**: {datetime.now().isoformat()}")
    lines.append("")

    evaluations: List[Dict[str, Any]] = results.get("evaluations", [])
    for ev in evaluations:
        lines.append(f"## Dataset: {ev.get('dataset', 'unknown')}")
        if "size" in ev:
            lines.append(f"- Samples: {ev['size']}")
        for key in ("accuracy", "precision", "recall", "f1"):
            if key in ev:
                lines.append(f"- {key.title()}: {ev[key]}")
        if any(k.startswith("baseline_") for k in ev.keys()):
            lines.append("- Baseline:")
            for key in ("baseline_accuracy", "baseline_precision", "baseline_recall", "baseline_f1"):
                if key in ev:
                    lines.append(f"  - {key.replace('baseline_', '').title()}: {ev[key]}")
        lines.append("")

    report_file.write_text("\n".join(lines), encoding="utf-8")
    return str(report_file)

