"""Self-contained HTML report for walk-forward results (no JS dependencies)."""

from __future__ import annotations

import html
from pathlib import Path

from ml_trading.backtest.walkforward import FoldOutcome, summarize

_STYLE = """
body{font-family:system-ui,sans-serif;margin:2rem;background:#0f1115;color:#e6e6e6}
h1{font-size:1.4rem} table{border-collapse:collapse;margin:1rem 0;width:100%}
th,td{border:1px solid #333;padding:.4rem .7rem;text-align:right;font-variant-numeric:tabular-nums}
th{background:#1a1d24} tr:nth-child(even){background:#14171d}
.pos{color:#4caf7d}.neg{color:#e05555}
svg{background:#14171d;border:1px solid #333;margin:.5rem 0}
"""


def _sparkline(values, width: int = 800, height: int = 120) -> str:
    if len(values) < 2:
        return ""
    lo, hi = float(min(values)), float(max(values))
    span = (hi - lo) or 1.0
    pts = " ".join(
        f"{i / (len(values) - 1) * width:.1f},{height - (v - lo) / span * (height - 10) - 5:.1f}"
        for i, v in enumerate(values)
    )
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        f'<polyline fill="none" stroke="#4caf7d" stroke-width="1.5" points="{pts}"/></svg>'
    )


def render_report(outcomes: list[FoldOutcome], title: str, out_path: str | Path) -> Path:
    summary = summarize(outcomes)
    rows = []
    for o in outcomes:
        m = o.metrics.as_dict()
        cells = "".join(
            f'<td class="{"pos" if v >= 0 else "neg"}">{v:,.3f}</td>'
            if isinstance(v, float)
            else f"<td>{v}</td>"
            for v in m.values()
        )
        rows.append(f"<tr><td>fold {o.fold}</td>{cells}</tr>")
    header = "".join(f"<th>{html.escape(k)}</th>" for k in (outcomes[0].metrics.as_dict() if outcomes else {}))
    summary_cells = "".join(f"<td>{v:,.3f}</td>" for v in summary.values())
    curves = "".join(
        f"<h3>fold {o.fold} equity</h3>{_sparkline(o.result.equity_curve.tolist())}" for o in outcomes
    )

    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>{html.escape(title)}</title><style>{_STYLE}</style></head><body>
<h1>{html.escape(title)}</h1>
<table><tr><th>fold</th>{header}</tr>{''.join(rows)}
<tr><td><b>mean</b></td>{summary_cells}</tr></table>
{curves}
</body></html>"""

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc)
    return path
