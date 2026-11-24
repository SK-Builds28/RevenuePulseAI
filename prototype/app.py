"""RevenuePulse AI - Streamlit decision-support prototype.

Run from the project root:

    streamlit run prototype/app.py

Four pages: Executive Dashboard, Review Queue, Account Detail and Human Review.
All scores come from the frozen Phase 2 model artifact. The app is decision
support only: it never contacts a customer, changes a price or writes to a CRM.

Every number shown is derived from synthetic data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.product.feedback import ReviewFeedbackStore
from src.product.recommendations import ALLOWED_ACTIONS
from src.product.review import (
    RISK_BANDS,
    DataUnavailableError,
    ModelUnavailableError,
    account_detail,
    build_queue,
    list_snapshot_dates,
    load_model_bundle,
    load_monthly_history,
    risk_summary,
)
from src.utils.config import load_config

CONFIG_PATH = PROJECT_ROOT / "config.yaml"

PAGES = (
    "Executive Dashboard",
    "Review Queue",
    "Account Detail",
    "Human Review",
)

DISCLAIMER = (
    "This score is a prioritization signal, not a statement that the customer "
    "will churn. All data in this prototype is synthetic, so no real-world "
    "performance is implied."
)


@st.cache_resource(show_spinner=False)
def get_config(config_path: str):
    return load_config(config_path)


@st.cache_data(show_spinner=False)
def get_dates(config_path: str) -> list[str]:
    config = load_config(config_path)
    return [value.date().isoformat() for value in list_snapshot_dates(config)]


def load_queue(config, snapshot: str) -> pd.DataFrame:
    return build_queue(
        config,
        snapshot_date=pd.Timestamp(snapshot),
        store=ReviewFeedbackStore(config.feedback_db_path),
    )


def _format_percentage(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value) * 100:+.1f}%"


def _format_probability(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.3f}"


def queue_table(queue: pd.DataFrame) -> pd.DataFrame:
    """Return the reviewer-facing column set with readable labels."""

    table = pd.DataFrame(
        {
            "Rank": queue["risk_rank"].astype(int),
            "Account": queue["account_id"].astype(str),
            "Segment": queue["segment"].astype(str),
            "Risk probability": queue["risk_probability"].round(3),
            "Risk band": queue["risk_band"].astype(str),
            "Recent revenue (3m mean)": queue["revenue_mean_3m"].round(0),
            "Revenue trend (3m)": queue["revenue_trend_3m_pct"].map(_format_percentage),
            "Engagement (3m mean)": queue["engagement_mean_3m"].round(1),
            "Days since activity": queue["days_since_activity_last"].round(1),
            "Primary reason": queue["primary_reason"].astype(str),
            "Review status": queue["review_status"].astype(str),
        }
    )
    return table


def _band_style(row: pd.Series) -> list[str]:
    band = str(row.get("Risk band", ""))
    colour = {
        "high": "background-color: #f8d7da",
        "medium": "background-color: #fff3cd",
        "low": "background-color: #e8f5e9",
    }.get(band, "")
    return [colour] * len(row)


def guard(function, *args, **kwargs):
    """Run a product call and render a controlled error instead of crashing."""

    try:
        return function(*args, **kwargs)
    except (DataUnavailableError, ModelUnavailableError) as exc:
        st.error(str(exc))
        st.stop()
    except FileNotFoundError as exc:
        st.error(f"Required file is missing: {exc}")
        st.stop()


def sidebar(config) -> tuple[str, str]:
    st.sidebar.title("RevenuePulse AI")
    st.sidebar.caption("Decision support only - human review is mandatory.")
    page = st.sidebar.radio("Page", PAGES, key="page")

    dates = guard(get_dates, str(CONFIG_PATH))
    if not dates:
        st.sidebar.error("No review dates are available in the snapshot data.")
        st.stop()
    default_index = 0
    snapshot = st.sidebar.selectbox(
        "Review date (snapshot)",
        options=dates,
        index=default_index,
        key="snapshot_date",
        help="Quarter-end snapshot date. Features use only data up to this date.",
    )
    st.sidebar.divider()
    st.sidebar.caption(
        "Synthetic data demo. Scores come from the frozen Phase 2 "
        "histogram gradient boosting model."
    )
    return page, snapshot


# --------------------------------------------------------------------------
# Page 1 - Executive Dashboard
# --------------------------------------------------------------------------
def page_dashboard(config, snapshot: str) -> None:
    st.title("RevenuePulse AI")
    st.subheader("AI-assisted account prioritization for revenue-risk review")
    st.warning(DISCLAIMER, icon="⚠️")

    queue = guard(load_queue, config, snapshot)
    summary = risk_summary(queue)
    bundle = guard(load_model_bundle, config)
    model_label = str(bundle.get("selected_model", "unknown")).replace("_", " ").title()

    columns = st.columns(5)
    columns[0].metric("Accounts reviewed", summary["reviewed"])
    columns[1].metric("High risk", summary["high"])
    columns[2].metric("Medium risk", summary["medium"])
    columns[3].metric("Low risk", summary["low"])
    columns[4].metric("Model used", model_label)

    st.caption(
        f"Risk bands use the configured probability thresholds: high ≥ "
        f"{config.risk_high:.2f}, medium ≥ {config.risk_medium:.2f}. "
        f"Average predicted probability: {summary['average_probability']:.3f}."
    )
    if summary["skipped"]:
        st.info(
            f"{summary['skipped']} snapshot row(s) were excluded from the queue "
            "because their core account fields were missing; no score was invented."
        )

    st.subheader("Risk distribution")
    distribution = pd.DataFrame(
        {
            "Risk band": ["high", "medium", "low"],
            "Accounts": [summary["high"], summary["medium"], summary["low"]],
        }
    ).set_index("Risk band")
    st.bar_chart(distribution, height=280)

    st.subheader("How to use this review")
    st.write(
        "RevenuePulse AI helps sales teams prioritize which accounts deserve "
        "review. It provides decision support only; all interventions require "
        "human approval. Start with the Review Queue, open an account to inspect "
        "the evidence, then record an accept / edit / dismiss decision on the "
        "Human Review page."
    )
    st.info(
        "Model performance on synthetic data: Precision@50 = 0.540, Lift@50 = 4.30, "
        "PR-AUC = 0.382 (748 snapshot rows, 380 accounts, 94 positives). These are "
        "synthetic-data results, not real-world evidence."
    )


# --------------------------------------------------------------------------
# Page 2 - Review Queue
# --------------------------------------------------------------------------
def page_queue(config, snapshot: str) -> None:
    st.title("Review Queue")
    st.caption(
        "Accounts ranked from highest to lowest model risk score for the selected "
        "review date. This is the core prioritization screen."
    )
    st.warning(DISCLAIMER, icon="⚠️")

    queue = guard(load_queue, config, snapshot)
    summary = risk_summary(queue)

    filter_columns = st.columns(4)
    segments = sorted(queue["segment"].astype(str).unique())
    selected_segments = filter_columns[0].multiselect(
        "Segment", options=segments, default=segments
    )
    selected_bands = filter_columns[1].multiselect(
        "Risk band", options=list(RISK_BANDS), default=list(RISK_BANDS)
    )
    maximum = max(1, len(queue))
    top_n = filter_columns[2].number_input(
        "Top N accounts", min_value=1, max_value=maximum, value=min(50, maximum)
    )
    filter_columns[3].metric("High risk this date", summary["high"])

    filtered = queue[
        queue["segment"].astype(str).isin(selected_segments)
        & queue["risk_band"].isin(selected_bands)
    ].head(int(top_n))

    if filtered.empty:
        st.info("No accounts match the current filters.")
        return

    st.caption(
        f"Showing {len(filtered)} of {len(queue)} scored accounts. "
        "High-risk rows are highlighted; medium-risk rows are advisory "
        "('Review recommended'), not definitive."
    )
    table = queue_table(filtered)
    st.dataframe(
        table.style.apply(_band_style, axis=1),
        use_container_width=True,
        hide_index=True,
        height=460,
    )

    st.divider()
    accounts = filtered["account_id"].astype(str).tolist()
    chosen = st.selectbox("Select an account to inspect", options=accounts)
    if st.button("Open account detail", type="primary"):
        st.session_state["selected_account"] = chosen
        st.session_state["page"] = "Account Detail"
        st.rerun()

    st.download_button(
        "Download this queue as CSV",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name=f"revenuepulse_queue_{snapshot}.csv",
        mime="text/csv",
    )


# --------------------------------------------------------------------------
# Page 3 - Account Detail
# --------------------------------------------------------------------------
def page_detail(config, snapshot: str) -> None:
    st.title("Account Detail")
    st.warning(DISCLAIMER, icon="⚠️")

    queue = guard(load_queue, config, snapshot)
    if queue.empty:
        st.info("No scored accounts are available for this review date.")
        return

    accounts = queue["account_id"].astype(str).tolist()
    default = st.session_state.get("selected_account")
    index = accounts.index(default) if default in accounts else 0
    account_id = st.selectbox("Account", options=accounts, index=index)
    st.session_state["selected_account"] = account_id

    detail = account_detail(config, queue, account_id)
    if detail is None:
        st.error("The selected account is not in the queue for this date.")
        return
    row = detail["row"]

    first = st.columns(4)
    first[0].metric("Risk probability", _format_probability(row["risk_probability"]))
    first[1].metric("Risk band", str(row["risk_band"]).title())
    first[2].metric("Segment", str(row["segment"]))
    first[3].metric("Review status", str(row["review_status"]).title())

    second = st.columns(4)
    second[0].metric("Recent revenue (3m mean)", f"{row['revenue_mean_3m']:,.0f}")
    second[1].metric("Revenue trend (3m)", _format_percentage(row["revenue_trend_3m_pct"]))
    second[2].metric("Engagement (3m mean)", f"{row['engagement_mean_3m']:.1f}")
    second[3].metric(
        "Days since activity",
        "n/a" if pd.isna(row["days_since_activity_last"]) else f"{row['days_since_activity_last']:.1f}",
    )

    st.subheader("Monthly revenue history")
    history = guard(load_monthly_history, config, account_id, snapshot)
    if history.empty:
        st.info("No monthly history is available for this account.")
    else:
        chart = history.set_index("month")[["monthly_revenue"]].rename(
            columns={"monthly_revenue": "Monthly revenue"}
        )
        st.line_chart(chart, height=260)
        st.caption(
            f"{len(history)} monthly observations up to {snapshot}. "
            "Revenue is in synthetic currency units."
        )

    st.subheader("Commercial signals")
    signals = pd.DataFrame(
        {
            "Signal": [
                "Revenue trend (3m vs prior 3m)",
                "Revenue trend (6m vs prior 6m)",
                "Transaction frequency trend (3m)",
                "Engagement level (3m mean)",
                "Engagement trend (3m)",
                "Days since last activity",
                "Active products",
                "Revenue volatility (6m)",
            ],
            "Value": [
                _format_percentage(row.get("revenue_trend_3m_pct")),
                _format_percentage(row.get("revenue_trend_6m_pct")),
                _format_percentage(row.get("transaction_trend_3m_pct")),
                f"{row.get('engagement_mean_3m'):.1f}" if pd.notna(row.get("engagement_mean_3m")) else "n/a",
                _format_percentage(row.get("engagement_trend_3m_pct")),
                "n/a" if pd.isna(row.get("days_since_activity_last")) else f"{row.get('days_since_activity_last'):.1f}",
                f"{row.get('product_count_last'):.0f}" if pd.notna(row.get("product_count_last")) else "n/a",
                f"{row.get('revenue_volatility_6m'):.3f}" if pd.notna(row.get("revenue_volatility_6m")) else "n/a",
            ],
        }
    )
    st.dataframe(signals, use_container_width=True, hide_index=True)

    st.subheader("Factors associated with this risk score")
    st.write(
        "The reason codes below are checkable rules evaluated on this account's "
        "own features. They describe association, not cause."
    )
    reasons = detail["reason_list"]
    if reasons:
        for reason in reasons:
            st.markdown(f"- {reason}")
    else:
        st.write(
            "No configured reason-code threshold fired for this account. The score "
            "comes from the model's combined feature pattern rather than a single "
            "rule."
        )

    recommendation = detail["recommendation"]
    st.subheader("Next-best-action suggestion")
    st.success(f"**{recommendation['action']}**")
    st.write(recommendation["rationale"])
    st.markdown("**Evidence from this account**")
    for item in recommendation["evidence"]:
        st.markdown(f"- {item}")
    st.caption(
        "Rule-based suggestion, not an LLM output and not an automated action. "
        "A reviewer decides whether it is justified."
    )


# --------------------------------------------------------------------------
# Page 4 - Human Review
# --------------------------------------------------------------------------
def page_review(config, snapshot: str) -> None:
    st.title("Human Review")
    st.caption(
        "Record the reviewer decision for the AI suggestion. This is the "
        "human-in-the-loop record of the prototype."
    )
    st.warning(DISCLAIMER, icon="⚠️")

    queue = guard(load_queue, config, snapshot)
    if queue.empty:
        st.info("No scored accounts are available for this review date.")
        return

    accounts = queue["account_id"].astype(str).tolist()
    default = st.session_state.get("selected_account")
    index = accounts.index(default) if default in accounts else 0
    account_id = st.selectbox("Account", options=accounts, index=index)
    st.session_state["selected_account"] = account_id

    detail = account_detail(config, queue, account_id)
    if detail is None:
        st.error("The selected account is not in the queue for this date.")
        return
    row = detail["row"]
    recommendation = detail["recommendation"]
    bundle = guard(load_model_bundle, config)

    left, right = st.columns([3, 2])
    with left:
        st.subheader("AI suggested action")
        st.info(f"**{recommendation['action']}**")
        st.write(recommendation["rationale"])
        st.caption("Evidence: " + " | ".join(recommendation["evidence"]))
        st.caption(
            f"Risk probability {row['risk_probability']:.3f} "
            f"({str(row['risk_band']).title()} band), "
            f"review status: {row['review_status']}."
        )
    with right:
        st.metric("Risk probability", f"{row['risk_probability']:.3f}")
        st.metric("Risk band", str(row["risk_band"]).title())
        st.metric("Account", account_id)

    st.divider()
    decision = st.radio(
        "Reviewer decision",
        options=["Accept", "Edit", "Dismiss"],
        horizontal=True,
    )
    edited_text = ""
    if decision == "Edit":
        edited_text = st.text_area(
            "Amended recommendation",
            value=recommendation["action"],
            help="Replace the AI suggestion with the action you consider appropriate.",
        )
    elif decision == "Dismiss":
        st.caption(
            "Dismissing records that the suggestion was not actionable. "
            "The account remains in the queue."
        )
    note = st.text_input("Reviewer note (optional)")

    if st.button("Save review decision", type="primary"):
        if decision == "Edit" and not edited_text.strip():
            st.error("Enter an amended recommendation before saving an edit.")
        else:
            store = ReviewFeedbackStore(config.feedback_db_path)
            row_id = store.record(
                account_id=account_id,
                snapshot_date=snapshot,
                risk_score=float(row["risk_probability"]),
                risk_band=str(row["risk_band"]),
                model_name=str(bundle.get("selected_model", "")),
                ai_recommendation=recommendation["action"],
                decision=decision.lower(),
                edited_recommendation=edited_text.strip() or None,
                reviewer_note=note.strip() or None,
                reviewer="demo-reviewer",
            )
            st.success(
                f"Decision saved (record #{row_id}). The review log is local to "
                f"this machine: {config.feedback_db_path}"
            )
            st.rerun()

    st.divider()
    st.subheader("Review log")
    store = ReviewFeedbackStore(config.feedback_db_path)
    log = store.fetch_all()
    if log.empty:
        st.info("No review decisions have been recorded yet.")
    else:
        st.dataframe(log.tail(25), use_container_width=True, hide_index=True)
        st.download_button(
            "Export review log to CSV",
            data=log.to_csv(index=False).encode("utf-8"),
            file_name="revenuepulse_review_log.csv",
            mime="text/csv",
        )
        counts = log["decision"].value_counts().to_dict()
        st.caption("Decision counts: " + ", ".join(f"{k}: {v}" for k, v in counts.items()))

    st.caption(
        "Prototype limitation: there is no authentication, no role-based access "
        "and no multi-user separation. A production pilot would require all three."
    )


def main() -> None:
    st.set_page_config(
        page_title="RevenuePulse AI",
        page_icon="📊",
        layout="wide",
    )
    config = guard(get_config, str(CONFIG_PATH))
    page, snapshot = sidebar(config)
    st.sidebar.caption(
        "Allowed suggested actions: " + ", ".join(ALLOWED_ACTIONS)
    )

    if page == "Executive Dashboard":
        page_dashboard(config, snapshot)
    elif page == "Review Queue":
        page_queue(config, snapshot)
    elif page == "Account Detail":
        page_detail(config, snapshot)
    else:
        page_review(config, snapshot)


if __name__ == "__main__":
    main()
