"""Two LLM passes. No rule table, no hardcoded rules, no canned text.

  route()    the model reads the context-header index for all 53 conditions
             and decides which record(s) the case actually concerns.
  analyse()  the model receives those records' COMPLETE knowledge-base content
             -- every criterion, investigation, procedure, drug, dose,
             frequency, care level, referral threshold and exclusion the STW
             states -- extracts the requirements it finds there, and checks the
             patient's input against each one.

Nothing in this file enumerates a clinical rule. The rules come out of the
knowledge base at request time, so a condition with rich guidance yields many
findings and a thin one yields few, without any code change. Every string the
caller reads is model-generated, including the explanation given when no
record matches at all.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from reasoner.catalog import Catalog
from reasoner.gemini import LLMUnavailable, Usage, generate

# ---------------------------------------------------------------------------
# Pass 1 -- routing over the context headers
# ---------------------------------------------------------------------------

ROUTER_SYSTEM = """You route clinical cases to the correct Standard Treatment \
Workflow record(s) from the Indian ICMR STW, Volume 1.

You are given an index where each line is one record's context header:
    record_id | Condition | Specialty | ICD-10 codes | page | covers: <sections>

Read the case and decide which record_id(s) it concerns. Judge on the clinical \
picture, not on keyword overlap: a case may name no condition at all and still \
clearly be one, and a case may mention a condition in passing that is not what \
is being asked about.

Select more than one record only when the case genuinely spans them (a \
comorbidity that changes management, or a requested service documented under a \
different condition). Select none when no record fits -- a wrong route is worse \
than an honest miss, because the next stage would then check the patient \
against requirements that do not apply to them.

Reply with JSON only:
{"record_ids": ["..."], "reasoning": "why these records, or if none, what the \
case appears to be and why no record in the index covers it", "confidence": \
"high|medium|low"}"""


@dataclass
class Routing:
    record_ids: list[str]
    reasoning: str
    confidence: str
    usage: Usage


def _extract_json(text: str) -> Any:
    """Models sometimes wrap JSON in prose or a fenced block, and a long
    response can be cut off mid-object by the output limit. Try clean parse,
    then the outermost span, then repair a truncated tail."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    span = re.search(r"[\{\[].*[\}\]]", text, re.S)
    if span:
        try:
            return json.loads(span.group(0))
        except json.JSONDecodeError:
            pass

    balanced = _balanced_prefix(text)
    if balanced is not None:
        return balanced

    repaired = _repair_truncated(text)
    if repaired is not None:
        return repaired
    raise json.JSONDecodeError("could not parse or repair model output", text[:200], 0)


def _balanced_prefix(text: str) -> Any | None:
    """Return the first balanced object, ignoring anything after it. Handles the
    surplus-closer case -- a model emitting one `}` too many makes the whole
    response unparseable even though the object itself is complete."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i, ch in enumerate(text[start:], start):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _repair_truncated(text: str) -> Any | None:
    """Close a response that was cut off mid-structure. Walks the text tracking
    string state and bracket depth, rewinds to the last complete element, and
    closes what is still open. Salvages the rules already emitted rather than
    discarding the whole response over a missing tail."""
    start = text.find("{")
    if start == -1:
        return None
    body = text[start:]

    stack: list[str] = []
    in_string = False
    escaped = False
    last_safe = None          # index just past the last completed element

    for i, ch in enumerate(body):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if stack:
                stack.pop()
            if len(stack) <= 1:
                last_safe = i + 1
        elif ch == "," and len(stack) <= 1:
            last_safe = i

    for candidate in (len(body), last_safe):
        if candidate is None:
            continue
        fragment = body[:candidate].rstrip().rstrip(",")
        depth: list[str] = []
        in_s = False
        esc = False
        for ch in fragment:
            if in_s:
                if esc: esc = False
                elif ch == "\\": esc = True
                elif ch == '"': in_s = False
                continue
            if ch == '"': in_s = True
            elif ch in "{[": depth.append("}" if ch == "{" else "]")
            elif ch in "}]" and depth: depth.pop()
        if in_s:
            fragment += '"'
        try:
            return json.loads(fragment + "".join(reversed(depth)))
        except json.JSONDecodeError:
            continue
    return None


def route(case: str, catalog: Catalog) -> Routing:
    prompt = (
        f"INDEX OF AVAILABLE RECORDS\n{catalog.index_text()}\n\n"
        f"CASE\n{case.strip()}\n\n"
        "Which record_id(s) does this case concern?"
    )
    raw, usage = generate(
        prompt,
        ROUTER_SYSTEM,
        temperature=0.1,          # routing is identification, not judgement
        response_mime_type="application/json",
    )
    try:
        parsed = _extract_json(raw)
    except (json.JSONDecodeError, ValueError):
        return Routing([], raw.strip()[:600], "low", usage)

    ids = parsed.get("record_ids") or []
    if isinstance(ids, str):
        ids = [ids]
    return Routing(
        record_ids=catalog.resolve([str(i) for i in ids]),
        reasoning=str(parsed.get("reasoning", "")),
        confidence=str(parsed.get("confidence", "medium")),
        usage=usage,
    )


# ---------------------------------------------------------------------------
# Pass 2 -- extract the KB's requirements, check the patient against each
# ---------------------------------------------------------------------------

ANALYST_SYSTEM = """You are a clinical reviewer checking one patient against the Indian ICMR Standard Treatment Workflows (Volume 1, 2019).

You receive the COMPLETE knowledge-base record for the matched condition -- not excerpts. Each block starts with its context header:
    [Condition | Specialty | ICD-10 | section type | ICMR STW 2019 Vol 1 p.N]

STEP 1 -- EXTRACT THE RULES

Read the record and identify every requirement it actually states: diagnostic thresholds and cut-off values, essential and desirable investigations, indications and contraindications, procedures and their preconditions, drugs with doses and frequency periods, care-level and facility requirements, referral, admission and discharge triggers, consent and documentation obligations, and any workflow step. Take them from the text in front of you. Never import a requirement from your own medical knowledge, and never invent one the record does not state.

For each rule, judge how binding the record makes it. The source marks this in its own words -- "essential", "mandatory", "MUST", "should", "desirable", "optional", "preferred", "consider". Record that as the obligation.

REPORT AT THE LEVEL THE GUIDELINE DECIDES AT

One rule per decision, not one per line of source text. Where the record states
a set that stands or falls together, that set is ONE rule: "Essential
investigations: Hb, CBC, TFT, USG" is a single rule, not four. A list of
interchangeable drug options is one rule about the therapy, not one rule per
agent. A list of red flags is one rule about whether any red flag is present,
not one rule per flag. Name the individual items inside `rule` and say in
`explanation` which ones are met and which are not.

Splitting a set into separate rules buries the few that decide the case under
many that do not. Consolidate.

MARK WHETHER A RULE GATES THE DECISION

Set `gating: true` when the rule bears on whether the requested service can be
authorised -- indications, contraindications, thresholds, mandatory workup,
required approvals, care level, preconditions for the procedure.

Set `gating: false` for material that belongs in the record but does not decide
this request: lifestyle and dietary advice, long-term prevention targets,
follow-up counselling, patient education. Report these when the case speaks to
them, but they must never be FAIL -- a prevention target the patient has not
met is not a reason to refuse authorisation. Use PASS or NOT_APPLICABLE and let
`gating: false` keep them out of the decision.

STEP 2 -- MATCH BY MEANING, NEVER BY KEYWORD

This is the part reviewers get wrong. The application and the guideline will use different words for the same thing. You must resolve them by clinical meaning:
  "BP 128/78" satisfies a rule asking for blood pressure.
  "haemogram" and "CBC" and "complete blood count" are the same investigation.
  "Hb 8.1 g/dL" satisfies a haemoglobin requirement.
  "USG" and "ultrasound" and "sonography" are the same.
  "uterus 16 weeks size" answers a rule phrased in centimetres if the record
  itself equates them, and a 15.8 cm uterus is a 16-week uterus.
  "norethisterone" IS a progestogen. "apixaban" IS an oral anticoagulant.
  "cath lab on site" satisfies "PCI capable centre".
A requirement is only unmet if the clinical fact is genuinely absent -- not because the application used a different word, an abbreviation, a brand name, a drug class, or a different unit. When you do resolve a term across wordings, say so in `matched_via` so the match can be audited.

STEP 3 -- CHECK THE PATIENT AGAINST EACH RULE

  PASS            the input satisfies it. Quote the patient value that shows it.
  FAIL            the input contradicts it, breaches a stated threshold, or
                  triggers a contraindication or explicit non-indication.
                  Say which value breaches it, and by how much where numeric.
  MISSING         the record requires or expects this and the input is genuinely
                  silent, after you have searched it for equivalent wordings.
  NOT_APPLICABLE  the rule exists but does not govern this patient (wrong age
                  band, wrong care setting, different arm of a conditional).
                  Say what makes it inapplicable.

WATCH THE POLARITY

Some rules describe something that must be PRESENT (an indication, a required
investigation). Others describe something that must be ABSENT (a
contraindication, an exclusion, a red flag, a non-indication).

Always state the verdict from the patient's side, against what the guideline
wants:
  A required finding that is present            -> PASS
  A required finding that is absent             -> MISSING, or FAIL if the
                                                   application states it was
                                                   not done
  A red flag or contraindication that is ABSENT -> PASS. The patient clears it.
  A red flag or contraindication that is PRESENT-> FAIL. Say what triggered it.

Never write PASS when what you mean is "this rule does not apply to this
patient" -- that is NOT_APPLICABLE. And never write FAIL for an adverse feature
being absent. Getting this backwards inverts the decision, so state in
`explanation` which way the rule points before giving the verdict.

STEP 4 -- THE CONFIDENCE SCALE

Every confidence number you produce -- per rule and overall -- uses this one
scale. Apply it consistently so the same evidential situation always scores in
the same band.

  90-100  CERTAIN. The record states this explicitly and the application
          answers it directly. No inference was needed beyond resolving
          wording. A different reviewer would reach the same verdict.
  75-89   CONFIDENT. The record is explicit, but the application answers it
          indirectly -- an equivalent term, a derived value, a fact implied by
          another. The bridge is sound and stated in matched_via.
  60-74   PROBABLE. Either the record is loosely worded, or the application is
          partial and the verdict rests on reasonable clinical inference. A
          careful reviewer could disagree at the margin.
  40-59   UNCERTAIN. Significant interpretation was required -- the record does
          not squarely address this situation, or the application is too thin
          to judge without assuming.
  0-39    SPECULATIVE. Little in the record or the application supports a firm
          verdict. Say so rather than dressing it up.

Score what the EVIDENCE supports, not how good or bad the case looks. A clean
MISSING verdict on an explicit mandatory rule is 90-100 confident -- you are
certain it is absent. Confidence is not approval.

STEP 5 -- FLAG, SCORE, EXPLAIN

`flagged` is your judgement of what a human reviewer must act on before this case can move, most serious first. A breached contraindication outranks a missing routine test. Decide what belongs there and why -- do not simply copy every non-PASS rule across.

`confidence.score` (0-100) is your confidence in the assessment as a whole, on
the same scale. It is not an average of the per-rule numbers: weigh how
completely the record covered this presentation, how much the application left
unsaid, and how much of your reasoning rested on inference rather than stated
fact. `confidence.band` is the band name from the scale above.

`confidence.basis` explains the number: name the specific things that support
it, the specific things that hold it down, and what would raise it. This is the
explanation a reviewer reads to decide whether to trust the assessment, so be
concrete -- refer to actual rules and actual values from this case, never to
generalities.

`overall_explanation` comes last and is prose for the human reading this: what is established, what blocks the decision, and what resolves it. Write it as a clinician would, not as a filled-in form.

How many rules you report is set by the record, not by any quota. Cite the page for every rule.

Reply with JSON only:
{
  "condition_assessment": "what this case is, clinically, in your own words",
  "rules": [
    {
      "rule": "the requirement as the record states it",
      "kb_section": "which section type it came from",
      "page": 38,
      "obligation": "mandatory|recommended|optional|conditional",
      "gating": true,
      "status": "PASS|FAIL|MISSING|NOT_APPLICABLE",
      "patient_evidence": "the value or finding from the input, or null",
      "matched_via": "how you resolved the input wording to the rule wording, or null if worded alike",
      "explanation": "why this status, referring to both the rule and the input",
      "why_it_matters": "clinical consequence -- required for FAIL and MISSING, else null",
      "confidence": 0,
      "confidence_band": "the band whose range contains that exact score"
    }
  ],
  "flagged": [
    {"rule": "...", "page": 0, "severity": "critical|major|minor",
     "issue": "what is wrong or absent", "action_required": "what must happen next"}
  ],
  "confidence": {
    "score": 0,
    "band": "certain|confident|probable|uncertain|speculative",
    "basis": "why this number, referring to actual rules and values in this case",
    "raised_by": ["specific things supporting the score"],
    "lowered_by": ["specific things holding it down"],
    "would_raise": "what evidence would move it into a higher band"
  },
  "overall_explanation": "prose for the human reviewer -- comes last"
}"""


@dataclass
class Analysis:
    case: str
    conditions: list[dict[str, Any]]
    condition_assessment: str
    rules: list[dict[str, Any]]
    flagged: list[dict[str, Any]]
    confidence: dict[str, Any]
    overall_explanation: str
    routing_reasoning: str
    routing_confidence: str
    matched: bool
    citations: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)

    def tally(self) -> dict[str, int]:
        """Statuses of the rules the model marked as gating the decision.
        Background material (diet, prevention, counselling) is counted
        separately so it cannot inflate a FAIL count the caller routes on."""
        counts: dict[str, int] = {}
        for r in self.gating_rules():
            status = str(r.get("status", "UNKNOWN")).upper()
            counts[status] = counts.get(status, 0) + 1
        return counts

    def gating_rules(self) -> list[dict[str, Any]]:
        return [r for r in self.reported_rules() if r.get("gating") is not False]

    def background_rules(self) -> list[dict[str, Any]]:
        return [r for r in self.reported_rules() if r.get("gating") is False]

    def reported_rules(self) -> list[dict[str, Any]]:
        """Rules that actually engaged for this patient. NOT_APPLICABLE ones are
        withheld: the rule exists in the record but does not govern this case,
        so it is neither a finding nor an action. `not_applicable_count` still
        reports how many were set aside, so nothing vanishes silently."""
        return [r for r in self.rules
                if str(r.get("status", "")).upper() != "NOT_APPLICABLE"]

    def not_applicable_count(self) -> int:
        return len(self.rules) - len(self.reported_rules())

    def missing_mandatory(self) -> list[dict[str, Any]]:
        """Rules the record makes binding that the input does not answer. The
        model decides both the obligation and the status; this only selects."""
        return [
            r for r in self.rules
            if str(r.get("status", "")).upper() == "MISSING"
            and str(r.get("obligation", "")).lower() in ("mandatory", "essential")
        ]


_VALID = {"PASS", "FAIL", "MISSING", "NOT_APPLICABLE"}

# The scale published in ANALYST_SYSTEM. The model chooses the NUMBER -- that is
# the clinical judgement. Naming the band the number falls into is arithmetic,
# not judgement, and the model gets it wrong often enough (labelling 95
# "confident" when 95 is "certain") that deriving it here is the only way the
# published scale actually holds.
_BANDS = ((90, "certain"), (75, "confident"), (60, "probable"), (40, "uncertain"), (0, "speculative"))


def band_for(score: Any) -> str | None:
    if not isinstance(score, (int, float)):
        return None
    return next((name for floor, name in _BANDS if score >= floor), "speculative")


def _clean(rules: Any) -> list[dict[str, Any]]:
    """Keep the model's content verbatim; only normalise status casing and drop
    non-objects. Nothing here judges clinical content."""
    out: list[dict[str, Any]] = []
    if not isinstance(rules, list):
        return out
    for item in rules:
        if not isinstance(item, dict):
            continue
        status = str(item.get("status", "")).strip().upper().replace(" ", "_")
        item["status"] = status if status in _VALID else "UNKNOWN"
        derived = band_for(item.get("confidence"))
        if derived is not None:
            item["confidence_band"] = derived
        out.append(item)
    return out


def analyse(case: str, catalog: Catalog, *, temperature: float = 0.3) -> Analysis:
    routing = route(case, catalog)

    if not routing.record_ids:
        # No canned text: the router's own explanation is what the caller gets.
        return Analysis(
            case=case, conditions=[], condition_assessment=routing.reasoning,
            rules=[], flagged=[],
            confidence={"score": None, "band": None, "basis": routing.reasoning,
                        "raised_by": [], "lowered_by": [],
                        "would_raise": None},
            overall_explanation=routing.reasoning,
            routing_reasoning=routing.reasoning,
            routing_confidence=routing.confidence, matched=False,
            usage={"route": routing.usage.__dict__, "analyse": None},
        )

    separator = "\n\n" + ("=" * 70) + "\n\n"
    body = separator.join(catalog.dossier(rid) for rid in routing.record_ids)
    prompt = (
        f"KNOWLEDGE-BASE RECORD(S)\n\n{body}\n\n"
        + ("=" * 70)
        + f"\n\nPATIENT APPLICATION\n\n{case.strip()}\n\n"
        "Extract the rules this record states, then check the patient "
        "application against each one. Resolve wording differences by clinical "
        "meaning, not by keyword."
    )

    raw, usage = generate(
        prompt, ANALYST_SYSTEM,
        temperature=temperature,
        response_mime_type="application/json",
        max_output_tokens=8192,   # a rich record can need well past the default
    )
    try:
        parsed = _extract_json(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        # Returning 200 with zero rules would read as "nothing to report" when
        # the truth is "the response never arrived intact".
        raise LLMUnavailable(
            f"model returned unparseable output ({usage.output_tokens} output tokens; "
            f"likely truncated). First 200 chars: {raw.strip()[:200]}"
        ) from exc

    conditions = [
        {
            "record_id": rid,
            "condition": catalog.conditions[rid].condition,
            "specialty": catalog.conditions[rid].specialty,
            "icd10": catalog.conditions[rid].icd10,
            "page": catalog.conditions[rid].page,
            "sections_reviewed": catalog.conditions[rid].chunk_types,
        }
        for rid in routing.record_ids
    ]

    flagged = parsed.get("flagged")
    conf = parsed.get("confidence")
    if not isinstance(conf, dict):
        conf = {"score": None, "band": None, "basis": "", "raised_by": [],
                "lowered_by": [], "would_raise": None}
    derived = band_for(conf.get("score"))
    if derived is not None:
        conf["band"] = derived
    return Analysis(
        case=case,
        conditions=conditions,
        condition_assessment=str(parsed.get("condition_assessment", "")),
        rules=_clean(parsed.get("rules")),
        flagged=flagged if isinstance(flagged, list) else [],
        confidence=conf,
        overall_explanation=str(parsed.get("overall_explanation", "")),
        routing_reasoning=routing.reasoning,
        routing_confidence=routing.confidence,
        matched=True,
        citations=catalog.citations(routing.record_ids),
        usage={"route": routing.usage.__dict__, "analyse": usage.__dict__},
    )
