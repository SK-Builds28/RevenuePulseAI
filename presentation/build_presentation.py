"""Generate the 11-slide RevenuePulse AI final deck as a .pptx file.

Usage from the project root::

    python presentation/build_presentation.py

Requires python-pptx (optional dependency; the Markdown deck in
``Final_Presentation.md`` is the source of truth). All figures are synthetic-data
results copied from ``artifacts/evaluation/test_results.json``.
"""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

OUTPUT = Path(__file__).resolve().parent / "RevenuePulse_AI_Final_Presentation.pptx"

NAVY = RGBColor(0x14, 0x2B, 0x45)
TEAL = RGBColor(0x0E, 0x7C, 0x86)
GREY = RGBColor(0x55, 0x60, 0x6B)
LIGHT = RGBColor(0xF7, 0xF9, 0xFB)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)


def _background(slide, colour: RGBColor = LIGHT) -> None:
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = colour


def _textbox(slide, left, top, width, height, text, *, size=18, colour=NAVY, bold=False):
    box = slide.shapes.add_textbox(left, top, width, height)
    frame = box.text_frame
    frame.word_wrap = True
    paragraph = frame.paragraphs[0]
    paragraph.text = text
    paragraph.alignment = PP_ALIGN.LEFT
    for run in paragraph.runs:
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = colour
        run.font.name = "Calibri"
    return box


def _bullets(slide, left, top, width, height, items, *, size=15, colour=NAVY):
    box = slide.shapes.add_textbox(left, top, width, height)
    frame = box.text_frame
    frame.word_wrap = True
    for index, item in enumerate(items):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.text = item
        paragraph.space_after = Pt(7)
        for run in paragraph.runs:
            run.font.size = Pt(size)
            run.font.color.rgb = colour
            run.font.name = "Calibri"


def _accent(slide, top: float = 1.05) -> None:
    bar = slide.shapes.add_shape(1, Inches(0.7), Inches(top), Inches(1.4), Inches(0.06))
    bar.fill.solid()
    bar.fill.fore_color.rgb = TEAL
    bar.line.fill.background()


def _footer(slide, number: int) -> None:
    _textbox(
        slide,
        Inches(0.7),
        Inches(7.05),
        Inches(8.0),
        Inches(0.35),
        "RevenuePulse AI · synthetic data · decision support only",
        size=9,
        colour=GREY,
    )
    _textbox(
        slide,
        Inches(11.9),
        Inches(7.05),
        Inches(0.8),
        Inches(0.35),
        f"{number}/11",
        size=9,
        colour=GREY,
    )


def content_slide(prs, title: str, subtitle: str, items, number: int, *, size=15):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _background(slide)
    _textbox(slide, Inches(0.7), Inches(0.4), Inches(11.9), Inches(0.6), title, size=27, bold=True)
    _accent(slide)
    _textbox(slide, Inches(0.7), Inches(1.25), Inches(11.9), Inches(0.45), subtitle, size=15, colour=TEAL)
    _bullets(slide, Inches(0.7), Inches(1.85), Inches(11.9), Inches(4.9), items, size=size)
    _footer(slide, number)
    return slide


def build() -> Path:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # Slide 1 - title
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _background(slide, WHITE)
    band = slide.shapes.add_shape(1, Inches(0), Inches(0), prs.slide_width, Inches(0.35))
    band.fill.solid()
    band.fill.fore_color.rgb = TEAL
    band.line.fill.background()
    _textbox(slide, Inches(0.9), Inches(1.9), Inches(11.4), Inches(1.0), "RevenuePulse AI", size=48, bold=True)
    _textbox(
        slide,
        Inches(0.9),
        Inches(2.95),
        Inches(11.4),
        Inches(0.7),
        "AI-assisted account prioritization for revenue-risk review",
        size=22,
        colour=TEAL,
    )
    _textbox(
        slide,
        Inches(0.9),
        Inches(3.9),
        Inches(11.4),
        Inches(0.8),
        "Recommendation: ITERATE toward a controlled real-data paid pilot. Do not scale.",
        size=18,
        colour=NAVY,
        bold=True,
    )
    _textbox(
        slide,
        Inches(0.9),
        Inches(4.8),
        Inches(11.4),
        Inches(1.2),
        "MBA ZG583 capstone · Suhani Kanwar\n"
        "Decision support only · human review mandatory · all data synthetic",
        size=13,
        colour=GREY,
    )

    # Slide 2
    content_slide(
        prs,
        "Problem and business stakes",
        "The review window is smaller than the book",
        [
            "A manager owns 150-400 accounts and can review roughly 50 per quarter.",
            "History is fragmented across billing, CRM and product tools; dashboards show the past.",
            "Rules that flag declining revenue fire after the decline is visible.",
            "Review effort follows recency and familiarity, not risk.",
            "Declines surface at renewal instead of at an intervention point.",
            "Two managers would prioritise the same book differently, and could not explain why.",
            "No market-size or loss figures are claimed; none were measured.",
        ],
        number=2,
    )

    # Slide 3
    content_slide(
        prs,
        "Target user and current workflow",
        "Sales managers, key account managers, RevOps analysts",
        [
            "Today: export or open data from several systems.",
            "Manually scan revenue, engagement, activity and support history.",
            "Form a judgement about which accounts look risky.",
            "Decide outreach or no action, then record a follow-up.",
            "Failure modes: inconsistent ordering, no shared definition of material decline,",
            "and accounts that are simply never looked at.",
            "Diagram: diagrams/initial_workflow.puml",
        ],
        number=3,
    )

    # Slide 4
    content_slide(
        prs,
        "Why AI / why not rules alone",
        "Ranking at review capacity, not another dashboard",
        [
            "Dashboard: shows history, does not rank the future.",
            "Hand-written rule: transparent, but encodes existing beliefs and misses interactions.",
            "Learned model: combines weak signals into a calibrated ordering.",
            "Locked synthetic test at 50 reviews (12.6% prevalence):",
            "    Scorecard P@50 0.140 | Lift 1.11 | PR-AUC 0.136",
            "    Logistic regression P@50 0.480 | Lift 3.82 | PR-AUC 0.350",
            "    Gradient boosting P@50 0.540 | Lift 4.30 | PR-AUC 0.382",
            "27 of 94 future decliners in 50 review slots versus 7 for the rule.",
            "Synthetic data only; the rule baseline is weak, so this is not proof ML is required.",
        ],
        number=4,
        size=14,
    )

    # Slide 5
    content_slide(
        prs,
        "Product concept + MVP",
        "A quarterly queue, evidence and a human decision",
        [
            "1. Executive Dashboard - review counts by risk band, model in use, synthetic warning.",
            "2. Review Queue - ranked accounts with probability, band, revenue, trend, engagement,",
            "    activity recency, primary reason and review status; filters and CSV export.",
            "3. Account Detail - revenue history, commercial signals, reason codes, next-best-action.",
            "4. Human Review - accept / edit / dismiss with note; SQLite audit log; CSV export.",
            "Boundary: no customer contact, pricing, offer, email or CRM write. Human approval required.",
            "Diagram: diagrams/final_workflow.puml",
        ],
        number=5,
        size=14,
    )

    # Slide 6
    content_slide(
        prs,
        "Data, label and model design",
        "Synthetic data, transparent label, leakage-safe features",
        [
            "400 synthetic accounts x 36 months; quarter-end snapshots; 3,921 labelled rows.",
            "Label: next 3-month mean <= 80% of trailing 3-month mean; >= 6 months history.",
            "It is a revenue-decline proxy, not confirmed churn.",
            "22 allow-listed point-in-time features; account_id never a feature.",
            "Preprocessing fitted on train only; short-history nulls imputed, not dropped.",
            "Logistic regression and histogram gradient boosting; selection on validation only.",
            "No test-set tuning. No class weighting (validation showed no gain, worse calibration).",
        ],
        number=6,
    )

    # Slide 7
    content_slide(
        prs,
        "Evaluation results",
        "748 rows, 380 accounts, 94 positives, 12.6% prevalence",
        [
            "Scorecard: P@10 0.200 | P@50 0.140 | Recall@50 0.074 | Lift 1.11 | PR-AUC 0.136 | Brier 0.243",
            "Logistic regression: P@10 0.500 | P@50 0.480 | Recall 0.255 | Lift 3.82 | PR-AUC 0.350 | Brier 0.097",
            "Gradient boosting: P@10 0.600 | P@25 0.720 | P@50 0.540 | Recall 0.287 | Lift 4.30",
            "Gradient boosting: PR-AUC 0.382 | ROC-AUC 0.756 | Brier 0.094 | false alarm@50 0.460",
            "HGB P@50 0.540 [0.400, 0.700] (account-level bootstrap, 1,000 iterations)",
            "HGB minus scorecard P@50 +0.400 [+0.240, +0.560]",
            "Calibration acceptable without a calibrator: mean bin gap 0.030.",
            "Recall@50 0.287: most decliners are outside the queue.",
            "SYNTHETIC DATA - NOT REAL-WORLD PERFORMANCE. Dominant driver is a generator artefact.",
        ],
        number=7,
        size=13,
    )

    # Slide 8
    content_slide(
        prs,
        "Prototype walkthrough",
        "From queue to audited decision",
        [
            "Dashboard: 370 accounts, 6 high / 29 medium / 335 low for 2025-09-30.",
            "Queue: filtered to Enterprise and high/medium, top 25; high-risk rows highlighted.",
            "Account: probability, band, revenue history, signals, reason codes.",
            "Suggestion: rule-based action with the exact values that triggered it.",
            "Decision: accept / edit / dismiss, optional note, stored with timestamp and score.",
            "Audit: runtime/review_feedback.db, exportable to CSV.",
            "Fallback: missing data or model stops the page; unscorable rows are excluded.",
            "Run: streamlit run prototype/app.py",
        ],
        number=8,
        size=14,
    )

    # Slide 9
    content_slide(
        prs,
        "Commercial roadmap + paid pilot",
        "Diagnostic, then pilot, then product",
        [
            "Phase 0 - Diagnostic sprint: 12-16 hours, Rs 50k-75k.",
            "Phase 1 - Paid pilot: 6-8 weeks, Rs 1.5L-2.5L, one segment.",
            "Phase 2 - Productised SaaS: after pilot evidence; recurring subscription.",
            "Phase 3 - Scale: CRM integration, multi-user controls, multi-tenancy.",
            "Pilot terms: fixed fee, 50% upfront, 50% at readout, two-week termination,",
            "no auto-renewal, no success fee in the first pilot.",
            "Data: account IDs and monthly aggregates only, encrypted transfer, deleted after 30 days.",
            "Success criteria agreed first: triage time, acceptance, Precision@K vs client rule,",
            "false-alarm tolerance, one documented intervention. No product-market fit claimed.",
        ],
        number=9,
        size=13,
    )

    # Slide 10
    content_slide(
        prs,
        "Risk, governance and human oversight",
        "What could go wrong, and what controls it",
        [
            "Synthetic-to-real gap -> real-data pilot with pre-agreed stop conditions.",
            "Counter-intuitive driver -> disclosed in the model card and queue context.",
            "False alarms and missed declines -> Precision@K, lift and Recall@50 disclosed.",
            "Automation bias -> mandatory accept / edit / dismiss; overrides logged.",
            "Privacy -> aggregate-only data, no personal identifiers, deletion policy.",
            "Unauthorised automation -> no outbound integration in code; guardrail test.",
            "Drift -> monitoring plan: drift, calibration, Precision@K per window, rollback.",
            "Missing auth in MVP -> documented limitation; role-based access is a pilot blocker.",
            "STOP if no lift, unacceptable false alarms, poor acceptance, or failing real-data performance.",
        ],
        number=10,
        size=13,
    )

    # Slide 11
    content_slide(
        prs,
        "Recommendation and ask",
        "Iterate, with a controlled real-data pilot",
        [
            "The evidence supports the method: leakage-safe features, validation-only selection,",
            "capacity-aware evaluation and account-level uncertainty.",
            "On synthetic data the model materially outranks a transparent rule (P@50 0.540 vs 0.140).",
            "It does not support any claim about real accounts, churn, retention or revenue.",
            "Ask: approve controlled real-data pilot exploration, not scale.",
            "Phase 0 diagnostic sprint (Rs 50k-75k, 12-16 hours) to validate data access,",
            "label materiality and the current workflow.",
            "Then a 6-8 week fixed-fee pilot with pre-agreed criteria and a real stop option.",
            "Thank you. Questions welcome on evaluation design, governance and the prototype.",
        ],
        number=11,
        size=14,
    )

    prs.save(OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    path = build()
    print(f"Wrote {path}")
