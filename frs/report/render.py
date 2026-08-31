"""Self-contained training report in HTML and Markdown.

Every figure is base64-embedded, so ``report.html`` is a single file with no
external assets -- it can be emailed, attached to a ticket, or opened from a USB
stick and still render. The Markdown mirror is for committing to git, where a
diff of the numbers is more useful than a picture.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from .plots import build_training_figures, roc_plot

logger = logging.getLogger(__name__)


_CSS = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body { margin: 0; background: #f6f7f9; color: #1a1d21;
       font: 15px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
.wrap { max-width: 980px; margin: 0 auto; padding: 40px 24px 80px; }
h1 { font-size: 28px; margin: 0 0 4px; letter-spacing: -0.02em; }
h2 { font-size: 19px; margin: 40px 0 14px; padding-bottom: 8px;
     border-bottom: 1px solid #e3e6ea; letter-spacing: -0.01em; }
.sub { color: #6b7280; font-size: 14px; margin-bottom: 8px; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
         gap: 12px; margin: 20px 0; }
.card { background: #fff; border: 1px solid #e3e6ea; border-radius: 10px; padding: 14px 16px; }
.card .k { font-size: 11px; text-transform: uppercase; letter-spacing: .06em;
           color: #6b7280; margin-bottom: 6px; }
.card .v { font-size: 22px; font-weight: 650; letter-spacing: -0.02em; }
.card .v small { font-size: 13px; font-weight: 500; color: #6b7280; }
table { width: 100%; border-collapse: collapse; background: #fff;
        border: 1px solid #e3e6ea; border-radius: 10px; overflow: hidden; font-size: 14px; }
th, td { padding: 9px 14px; text-align: left; border-bottom: 1px solid #eef0f3; }
th { background: #fafbfc; font-weight: 600; font-size: 12px;
     text-transform: uppercase; letter-spacing: .04em; color: #4b5563; }
tr:last-child td { border-bottom: none; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
figure { margin: 18px 0; background: #fff; border: 1px solid #e3e6ea;
         border-radius: 10px; padding: 14px; }
figure img { width: 100%; display: block; }
.note { background: #fffbeb; border: 1px solid #fde68a; border-radius: 8px;
        padding: 12px 16px; font-size: 14px; margin: 18px 0; }
.scroll { overflow-x: auto; }
pre { background: #fff; border: 1px solid #e3e6ea; border-radius: 10px;
      padding: 14px; overflow-x: auto; font-size: 12.5px; line-height: 1.5; }
"""


def _card(key: str, value: str, unit: str = "") -> str:
    suffix = f" <small>{unit}</small>" if unit else ""
    return f'<div class="card"><div class="k">{key}</div><div class="v">{value}{suffix}</div></div>'


def _figure(title: str, b64: str) -> str:
    return (
        f'<figure><img alt="{title}" src="data:image/png;base64,{b64}">'
        f"</figure>"
    )


def _fmt(value: Any, spec: str = ".4f") -> str:
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return str(value) if value is not None else "-"


def _build_sections(
    cfg: Any,
    history: Any,
    dataset_stats: dict[str, Any],
    best_metrics: dict[str, Any],
    eval_results: dict[str, Any] | None,
) -> dict[str, Any]:
    """Collect everything both renderers need."""
    records = history.to_list()
    last = records[-1] if records else {}
    total_time = sum(float(r.get("epoch_time_sec", 0) or 0) for r in records)
    primary = cfg.get("eval", {}).get("primary")

    return {
        "name": cfg.get_path("experiment.name", "unnamed"),
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "arch": cfg.get_path("model.backbone.arch", "?"),
        "head": cfg.get_path("model.head.type", "?"),
        "epochs_done": len(records),
        "epochs_planned": cfg.get_path("train.epochs", 0),
        "batch": cfg.get_path("data.batch_size", 0),
        "accum": cfg.get_path("train.grad_accum_steps", 1),
        "lr": cfg.get_path("optim.lr", 0),
        "optimizer": cfg.get_path("optim.type", "?"),
        "scheduler": cfg.get_path("scheduler.type", "?"),
        "amp": cfg.get_path("train.amp", False),
        "dataset": dataset_stats,
        "final_loss": last.get("train_loss"),
        "final_acc": last.get("train_acc1"),
        "total_hours": total_time / 3600.0,
        "throughput": last.get("images_per_sec"),
        "primary_metric": primary,
        "best_value": best_metrics.get(primary) if primary else None,
        "best_epoch": best_metrics.get("epoch"),
        "records": records,
        "eval_results": eval_results or {},
    }


def render_html(ctx: dict[str, Any], figures: dict[str, str]) -> str:
    d = ctx["dataset"]
    eff_batch = ctx["batch"] * ctx["accum"]

    cards = [
        _card("Backbone", str(ctx["arch"])),
        _card("Loss head", str(ctx["head"])),
        _card("Epochs", f'{ctx["epochs_done"]}', f'of {ctx["epochs_planned"]}'),
        _card("Effective batch", str(eff_batch)),
    ]
    if ctx["best_value"] is not None:
        cards.insert(0, _card("Best accuracy", f'{float(ctx["best_value"]) * 100:.2f}', "%"))
    if ctx["final_loss"] is not None:
        cards.append(_card("Final loss", _fmt(ctx["final_loss"])))
    if ctx["total_hours"]:
        cards.append(_card("Wall clock", f'{ctx["total_hours"]:.2f}', "h"))
    if ctx["throughput"]:
        cards.append(_card("Throughput", f'{float(ctx["throughput"]):.0f}', "img/s"))

    parts = [
        "<!-- ESSI-FRS training report -->",
        f"<style>{_CSS}</style>",
        '<div class="wrap">',
        f'<h1>{ctx["name"]}</h1>',
        f'<div class="sub">Training report &middot; generated {ctx["generated"]}</div>',
        f'<div class="cards">{"".join(cards)}</div>',
    ]

    # --- dataset ---
    parts.append("<h2>Dataset</h2>")
    parts.append(
        '<div class="scroll"><table>'
        "<tr><th>Images</th><th>Identities</th><th class='num'>Min/id</th>"
        "<th class='num'>Max/id</th><th class='num'>Mean/id</th></tr>"
        f'<tr><td>{d.get("num_samples", 0):,}</td>'
        f'<td>{d.get("num_identities", 0):,}</td>'
        f'<td class="num">{d.get("min_per_identity", 0)}</td>'
        f'<td class="num">{d.get("max_per_identity", 0)}</td>'
        f'<td class="num">{_fmt(d.get("mean_per_identity"), ".1f")}</td></tr>'
        "</table></div>"
    )
    if "identity_distribution" in figures:
        parts.append(_figure("Images per identity", figures["identity_distribution"]))

    # --- evaluation ---
    if ctx["eval_results"]:
        parts.append("<h2>Verification results</h2>")
        rows = [
            "<tr><th>Benchmark</th><th class='num'>Accuracy</th><th class='num'>Std</th>"
            "<th class='num'>AUC</th><th class='num'>Threshold</th>"
            "<th class='num'>TAR@FAR=1e-3</th><th class='num'>Pairs</th></tr>"
        ]
        for name, res in ctx["eval_results"].items():
            rows.append(
                f"<tr><td>{name}</td>"
                f'<td class="num"><strong>{res.get("accuracy", 0) * 100:.3f}%</strong></td>'
                f'<td class="num">{res.get("accuracy_std", 0) * 100:.3f}</td>'
                f'<td class="num">{_fmt(res.get("auc"))}</td>'
                f'<td class="num">{_fmt(res.get("threshold"))}</td>'
                f'<td class="num">{res.get("tar@far=0.001", 0) * 100:.2f}%</td>'
                f'<td class="num">{res.get("num_pairs", 0):,}</td></tr>'
            )
        parts.append(f'<div class="scroll"><table>{"".join(rows)}</table></div>')

        curves = {
            name: {**res["roc"], "auc": res.get("auc")}
            for name, res in ctx["eval_results"].items()
            if res.get("roc")
        }
        roc = roc_plot(curves)
        if roc:
            parts.append(_figure("ROC", roc))

        parts.append(
            '<div class="note"><strong>Reading these numbers.</strong> Accuracy is the '
            "mean over 10 folds, with the threshold chosen on 9 folds and applied to the "
            "held-out one. Validation identities were excluded from training entirely, so "
            "this measures verification of unseen people rather than memorisation. "
            "TAR@FAR is the more operationally meaningful figure for access control.</div>"
        )

    # --- curves ---
    parts.append("<h2>Training curves</h2>")
    for key, title in (
        ("loss", "Loss"),
        ("eval_accuracy", "Verification accuracy"),
        ("lr", "Learning rate"),
        ("train_acc", "Training accuracy"),
        ("feature_norm", "Feature norm"),
        ("grad_norm", "Gradient norm"),
        ("throughput", "Throughput"),
    ):
        if key in figures:
            parts.append(_figure(title, figures[key]))

    # --- history ---
    if ctx["records"]:
        parts.append("<h2>Per-epoch history</h2>")
        keys = [k for k in ctx["records"][-1] if k != "epoch"]
        head_row = "<tr><th class='num'>Epoch</th>" + "".join(
            f"<th class='num'>{k}</th>" for k in keys
        ) + "</tr>"
        body = []
        for rec in ctx["records"]:
            cells = "".join(
                f'<td class="num">{_fmt(rec.get(k), ".4f")}</td>' for k in keys
            )
            body.append(f'<tr><td class="num">{rec["epoch"]}</td>{cells}</tr>')
        parts.append(f'<div class="scroll"><table>{head_row}{"".join(body)}</table></div>')

    # --- config ---
    parts.append("<h2>Configuration</h2>")
    try:
        import yaml

        dumped = yaml.safe_dump(
            cfg_to_dict(ctx.get("config")), sort_keys=False, default_flow_style=False
        )
    except Exception:
        dumped = "(config unavailable)"
    parts.append(f"<pre>{_escape(dumped)}</pre>")
    parts.append("</div>")
    return "\n".join(parts)


def cfg_to_dict(cfg: Any) -> Any:
    return cfg.to_dict() if hasattr(cfg, "to_dict") else (cfg or {})


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )


def render_markdown(ctx: dict[str, Any]) -> str:
    d = ctx["dataset"]
    lines = [
        f'# {ctx["name"]} - training report',
        "",
        f'Generated {ctx["generated"]}',
        "",
        "## Summary",
        "",
        "| | |",
        "|---|---|",
        f'| Backbone | `{ctx["arch"]}` |',
        f'| Loss head | `{ctx["head"]}` |',
        f'| Epochs | {ctx["epochs_done"]} of {ctx["epochs_planned"]} |',
        f'| Batch | {ctx["batch"]} x {ctx["accum"]} accum = {ctx["batch"] * ctx["accum"]} effective |',
        f'| Optimizer | {ctx["optimizer"]}, lr={ctx["lr"]} |',
        f'| Scheduler | {ctx["scheduler"]} |',
        f'| AMP | {ctx["amp"]} |',
        f'| Wall clock | {ctx["total_hours"]:.2f} h |',
    ]
    if ctx["best_value"] is not None:
        lines.append(
            f'| **Best {ctx["primary_metric"]}** | '
            f'**{float(ctx["best_value"]) * 100:.3f}%** (epoch {ctx["best_epoch"]}) |'
        )

    lines += [
        "",
        "## Dataset",
        "",
        f'- {d.get("num_samples", 0):,} images across {d.get("num_identities", 0):,} identities',
        f'- {d.get("min_per_identity", 0)}-{d.get("max_per_identity", 0)} images per identity '
        f'(mean {_fmt(d.get("mean_per_identity"), ".1f")})',
    ]

    if ctx["eval_results"]:
        lines += [
            "",
            "## Verification results",
            "",
            "| Benchmark | Accuracy | Std | AUC | Threshold | TAR@FAR=1e-3 | Pairs |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for name, res in ctx["eval_results"].items():
            lines.append(
                f'| {name} | **{res.get("accuracy", 0) * 100:.3f}%** '
                f'| {res.get("accuracy_std", 0) * 100:.3f} '
                f'| {_fmt(res.get("auc"))} | {_fmt(res.get("threshold"))} '
                f'| {res.get("tar@far=0.001", 0) * 100:.2f}% | {res.get("num_pairs", 0):,} |'
            )
        lines += [
            "",
            "> Accuracy is the 10-fold mean with the threshold fitted on 9 folds and",
            "> applied to the held-out fold. Validation identities were excluded from",
            "> training, so this is verification of unseen people, not memorisation.",
        ]

    if ctx["records"]:
        keys = [k for k in ctx["records"][-1] if k != "epoch"]
        lines += [
            "",
            "## Per-epoch history",
            "",
            "| epoch | " + " | ".join(keys) + " |",
            "|---:|" + "---:|" * len(keys),
        ]
        for rec in ctx["records"]:
            cells = " | ".join(_fmt(rec.get(k), ".4f") for k in keys)
            lines.append(f'| {rec["epoch"]} | {cells} |')

    return "\n".join(lines) + "\n"


def write_report(
    output_dir: str | Path,
    cfg: Any,
    history: Any,
    dataset_stats: dict[str, Any],
    best_metrics: dict[str, Any] | None = None,
    eval_results: dict[str, Any] | None = None,
    formats: list[str] | None = None,
) -> list[str]:
    """Render and write the report. Returns the paths written."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    formats = formats or ["html", "md"]

    ctx = _build_sections(cfg, history, dataset_stats, best_metrics or {}, eval_results)
    ctx["config"] = cfg

    written: list[str] = []
    if "html" in formats:
        figures = build_training_figures(history, dataset_stats)
        path = output_dir / "report.html"
        path.write_text(render_html(ctx, figures), encoding="utf-8")
        written.append(str(path))
    if "md" in formats:
        path = output_dir / "report.md"
        path.write_text(render_markdown(ctx), encoding="utf-8")
        written.append(str(path))
    return written
