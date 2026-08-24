"""Reviewer UI. A decision packet is rendered to a single self-contained
static HTML file: no server, no build step, safe to email or open from a
shared drive. Shows the decision, the criteria table (which passed, which
failed, which are still open), score and completeness, and the cited
guideline text -- since no page-image files are available offline, the
exact quoted source text stands in for the page image the spec asks for,
clearly labeled as a stand-in rather than hidden.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

try:
    from .scoring_model import is_genuinely_failed
except ImportError:
    from scoring_model import is_genuinely_failed

_DECISION_COLORS = {
    "approve": "#1a7f37", "pend": "#9a6700", "request_more_information": "#9a6700",
    "deny": "#b91c1c", "not_covered": "#6e7781", "no_authorization_needed": "#6e7781",
}
_GOOD, _BAD, _UNKNOWN, _NA = "#1a7f37", "#b91c1c", "#9a6700", "#6e7781"


def _display_verdict(c: dict[str, Any]) -> tuple[str, str, str]:
    """(color, symbol, label) accounting for `exclusion` criteria's
    inverted polarity -- see scoring_model.is_genuinely_failed. Without
    this, an exclusion that correctly did NOT match would render as a red
    cross, which reads backwards to a reviewer."""
    verdict = c["verdict"]
    if verdict in ("insufficient", "not_applicable"):
        return (_UNKNOWN, "?", verdict) if verdict == "insufficient" else (_NA, "&ndash;", verdict)
    if is_genuinely_failed(c["type"], verdict):
        return _BAD, "&cross;", "does not meet"
    return _GOOD, "&check;", "meets"


def _esc(value: Any) -> str:
    return html.escape(str(value)) if value is not None else ""


def _criteria_rows(criteria: list[dict[str, Any]]) -> str:
    rows = []
    for c in criteria:
        color, symbol, label = _display_verdict(c)
        conf = f" (conf {c['confidence']:.2f})" if c.get("confidence") is not None else ""
        rows.append(f"""
        <tr>
          <td><code>{_esc(c['criterion_id'])}</code></td>
          <td>{_esc(c['type'])}</td>
          <td>{_esc(c['weight'])}</td>
          <td style="color:{color};font-weight:700;text-align:center">{symbol}</td>
          <td style="color:{color};font-weight:600">{_esc(label)}{conf}</td>
          <td>{_esc(c['text'])}</td>
          <td>p.{_esc(c['citation'].get('page'))} &mdash; {_esc(c['review_status'])}</td>
        </tr>""")
    return "".join(rows)


def _citation_rows(citations: list[dict[str, Any]]) -> str:
    rows = []
    for c in citations[:12]:
        rows.append(f"""
        <div class="citation">
          <div class="citation-head">{_esc(c['condition'])} &middot; {_esc(c['chunk_type'])} &middot; p.{_esc(c['page'])}
            <span class="tag">{_esc(c['retrieval'])}</span></div>
          <pre>{_esc(c['quote'])}</pre>
        </div>""")
    return "".join(rows)


def _requested_info_rows(items: list[dict[str, Any]]) -> str:
    if not items:
        return "<p><em>None.</em></p>"
    rows = [f"<li><strong>{_esc(i['item'])}</strong> ({_esc(i['criticality'])}, p.{_esc(i.get('page'))}) &mdash; {_esc(i['why'])}</li>" for i in items]
    return f"<ul>{''.join(rows)}</ul>"


def render_html(packet: dict[str, Any]) -> str:
    decision = packet.get("decision", "unknown")
    color = _DECISION_COLORS.get(decision, "#333")
    doc = packet.get("pillars", {}).get("documentation", {})
    elig = packet.get("pillars", {}).get("eligibility", {})
    tally = packet.get("tally", {})

    tally_html = " &middot; ".join(f"{k}: <strong>{v}</strong>" for k, v in tally.items()) or "&ndash;"
    drivers_html = "".join(
        f"<li>{_esc(d.get('factor'))}: <strong>{_esc(d.get('contribution'))}</strong></li>"
        for d in packet.get("top_score_drivers", [])
    ) or "<li><em>No individual driver crossed the reporting threshold.</em></li>"

    return f"""<title>Prior Auth Review &mdash; {_esc(packet.get('request_id'))}</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; max-width: 960px; margin: 2rem auto; padding: 0 1rem;
          background: Canvas; color: CanvasText; line-height: 1.5; }}
  h1, h2 {{ font-weight: 600; }}
  .badge {{ display:inline-block; padding: 0.3em 0.9em; border-radius: 999px; color: white; font-weight: 700;
            background: {color}; text-transform: uppercase; letter-spacing: 0.03em; font-size: 0.9rem; }}
  .scores {{ display:flex; gap: 2rem; margin: 1rem 0 2rem; }}
  .score-card {{ border: 1px solid color-mix(in srgb, CanvasText 20%, transparent); border-radius: 10px; padding: 1rem 1.5rem; flex:1; }}
  .score-card .value {{ font-size: 2rem; font-weight: 700; }}
  .score-card .band {{ opacity: 0.7; }}
  table {{ width:100%; border-collapse: collapse; margin: 1rem 0; }}
  th, td {{ text-align:left; padding: 0.5em 0.6em; border-bottom: 1px solid color-mix(in srgb, CanvasText 15%, transparent); font-size: 0.92rem; vertical-align: top; }}
  th {{ opacity: 0.7; font-weight: 600; }}
  code {{ font-size: 0.85em; }}
  .citation {{ border-left: 3px solid color-mix(in srgb, CanvasText 25%, transparent); padding: 0.4em 0.9em; margin: 0.7em 0; }}
  .citation-head {{ font-size: 0.85rem; opacity: 0.75; margin-bottom: 0.3em; }}
  .tag {{ background: color-mix(in srgb, CanvasText 10%, transparent); border-radius: 4px; padding: 0.1em 0.5em; margin-left: 0.5em; font-size: 0.75rem; }}
  pre {{ white-space: pre-wrap; margin: 0; font-family: inherit; font-size: 0.92rem; }}
  section {{ margin-bottom: 2rem; }}
  .rationale {{ background: color-mix(in srgb, CanvasText 6%, transparent); padding: 1rem; border-radius: 8px; }}
  details summary {{ cursor: pointer; opacity: 0.7; }}
</style>

<h1>Prior Authorization Review</h1>
<p>Request <code>{_esc(packet.get('request_id'))}</code> &middot; evaluated {_esc(packet.get('evaluated_at'))} &middot;
   guideline {_esc(packet.get('guideline_version'))} &middot; rule table {_esc(packet.get('rule_table_version'))}</p>

<p><span class="badge">{_esc(decision.replace('_', ' '))}</span></p>

<section class="scores">
  <div class="score-card">
    <div>Score</div>
    <div class="value">{_esc(packet.get('score', '&ndash;'))}</div>
    <div class="band">{_esc(packet.get('score_reason') or '')}</div>
  </div>
  <div class="score-card">
    <div>Completeness</div>
    <div class="value">{_esc(packet.get('completeness', '&ndash;'))}%</div>
    <div class="band">{tally_html}</div>
  </div>
</section>

<section>
  <h2>Rationale</h2>
  <div class="rationale">{_esc(packet.get('clinical_rationale'))}</div>
</section>

<section>
  <h2>Top score drivers</h2>
  <ul>{drivers_html}</ul>
</section>

<section>
  <h2>Eligibility (Pillar 1)</h2>
  <p>{_esc(elig.get('reason'))}</p>
</section>

<section>
  <h2>Criteria evaluated</h2>
  <table>
    <tr><th>Criterion</th><th>Type</th><th>Weight</th><th></th><th>Verdict</th><th>Text</th><th>Source</th></tr>
    {_criteria_rows(packet.get('criteria', []))}
  </table>
</section>

<section>
  <h2>Documentation on file</h2>
  <p>Documentation completeness (informational, not scored): <strong>{_esc(doc.get('score'))}</strong></p>
  <p>Missing (essential): {_esc(', '.join(doc.get('required_missing', [])) or 'none')}</p>
  <h3>Requested information</h3>
  {_requested_info_rows(packet.get('requested_information', []))}
</section>

<section>
  <h2>Citations</h2>
  <p><em>No scanned page image is available in this offline build; the exact quoted guideline text is shown
     as the closest available stand-in. A production deployment would show the source page image here.</em></p>
  {_citation_rows(packet.get('citations', []))}
</section>

<details>
  <summary>Raw decision packet (JSON)</summary>
  <pre>{_esc(json.dumps(packet, indent=2))}</pre>
</details>
"""


def write_report(packet: dict[str, Any], destination: Path) -> None:
    destination.write_text(render_html(packet), encoding="utf-8")
