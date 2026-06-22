"""Streamlit interface for the Form 5500 Lead Discovery Platform."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from logic import (
    AnalysisError,
    AnalysisSettings,
    analyze_csv,
    analyze_zip,
    file_sha256,
    inspect_zip_archive,
    write_uploaded_file_to_temp,
)
from schema import SchemaError


st.set_page_config(
    page_title="Form 5500 Lead Discovery Platform",
    page_icon="📊",
    layout="wide",
)

US_STATE_OPTIONS = [
    "AA",
    "AE",
    "AK",
    "AL",
    "AP",
    "AR",
    "AS",
    "AZ",
    "CA",
    "CO",
    "CT",
    "DC",
    "DE",
    "FL",
    "GA",
    "GU",
    "HI",
    "IA",
    "ID",
    "IL",
    "IN",
    "KS",
    "KY",
    "LA",
    "MA",
    "MD",
    "ME",
    "MI",
    "MN",
    "MO",
    "MP",
    "MS",
    "MT",
    "NC",
    "ND",
    "NE",
    "NH",
    "NJ",
    "NM",
    "NV",
    "NY",
    "OH",
    "OK",
    "OR",
    "PA",
    "PR",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VA",
    "VI",
    "VT",
    "WA",
    "WI",
    "WV",
    "WY",
]


@st.dialog("Upload Form 5500 Dataset")
def require_dataset_upload() -> None:
    """Require a dataset upload before showing the application controls."""

    st.write("Upload a DOL Form 5500 or Form 5500-SF CSV or ZIP archive to begin.")
    st.markdown(
        "You can download the source files from the "
        "[Department of Labor Form 5500 datasets page]"
        "(https://www.dol.gov/agencies/ebsa/about-ebsa/our-activities/public-disclosure/foia/form-5500-datasets)."
    )
    st.file_uploader(
        "Upload Form 5500 or Form 5500-SF CSV or ZIP Archive",
        type=["csv", "zip"],
        key="uploaded_dataset",
    )


def main() -> None:
    """Render the Streamlit application."""

    uploaded = st.session_state.get("uploaded_dataset")
    if uploaded is None:
        require_dataset_upload()
        st.stop()

    st.title("Form 5500 Lead Discovery Platform")
    st.write(
        "Filter by recently established Cash Balance and Defined Benefit plans, as well "
        "as plan participant count."
    )

    with st.sidebar:
        uploaded = st.file_uploader(
            "Upload Form 5500 or Form 5500-SF CSV or ZIP Archive",
            type=["csv", "zip"],
            key="uploaded_dataset",
        )
        min_participants = st.number_input("Minimum Participants", min_value=0, value=10, step=1)
        max_participants = st.number_input("Maximum Participants", min_value=0, value=100, step=1)
        max_plan_age_months = st.slider(
            "Max Plan Age (Months)",
            min_value=1,
            max_value=120,
            value=24,
            step=1,
            help=(
                "Calculated relative to the most recent reliable date found inside the uploaded dataset. "
                "Used when Require Recent Plan Effective Date is enabled."
            ),
        )
        require_recent = st.checkbox(
            "Require Recent Plan Effective Date",
            value=False,
            help=(
                "When enabled, only plans with PLAN_EFF_DATE inside the selected age window are returned. "
                "Leave off to find filings that report Cash Balance or pay-related Defined Benefit codes "
                "even when the underlying plan effective date is older."
            ),
        )
        include_frozen = st.checkbox("Include Frozen Plans", value=False)
        include_unknown = st.checkbox("Include Initial Filings With Missing Effective Dates", value=False)
        selected_states = st.multiselect(
            "State Filter",
            options=US_STATE_OPTIONS,
            default=[],
            placeholder="All States",
            help="Leave empty to include all sponsor states.",
        )
        signal_filter_mode = st.radio(
            "Signal Filter Mode",
            options=["All companies in participant range", "Only companies with selected plan signals"],
            index=0,
            help=(
                "Choose all companies to ignore Cash Balance / Defined Benefit signal codes. "
                "Choose selected plan signals to narrow results to 1A and/or 1C filings."
            ),
        )
        selected_signal_types: list[str] = []
        if signal_filter_mode == "Only companies with selected plan signals":
            selected_signal_types = st.multiselect(
                "Signal Type Filter",
                options=["Cash Balance", "Pay-Related Defined Benefit", "Both"],
                default=["Cash Balance", "Pay-Related Defined Benefit"],
                help="Select one or more signal types to require.",
            )
        combine_selected = st.checkbox("Combine Selected CSV Files", value=False)
        run = st.button("Run Analysis", type="primary")

    suffix = Path(uploaded.name).suffix.lower()
    if suffix not in {".csv", ".zip"}:
        st.error("Unsupported file type. Upload a .csv or .zip file.")
        return

    settings = AnalysisSettings(
        min_participants=int(min_participants),
        max_participants=int(max_participants),
        max_plan_age_months=int(max_plan_age_months),
        require_recent_effective_date=require_recent,
        include_frozen_plans=include_frozen,
        include_unknown_effective_initial_filings=include_unknown,
        states=tuple(selected_states),
        signal_types=tuple(selected_signal_types),
    )

    selected_members: list[str] | None = None
    temp_path: str | None = None
    try:
        temp_path = write_uploaded_file_to_temp(uploaded, suffix=suffix)
        if suffix == ".zip":
            zip_info = inspect_zip_archive(temp_path)
            candidates = zip_info["candidates"]
            if len(candidates) > 1:
                st.subheader("Supported CSV Files in Archive")
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "Archive Filename": item.filename,
                                "Detected Form Type": item.detected_form_type,
                                "Compressed Size": item.compressed_size,
                                "Uncompressed Size": item.uncompressed_size,
                            }
                            for item in candidates
                        ]
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
                selected_members = st.multiselect(
                    "Select CSV member(s) to analyze",
                    options=[item.filename for item in candidates],
                    default=[],
                )
            elif len(candidates) == 1:
                selected_members = [candidates[0].filename]
                st.caption(f"Selected archive member: {candidates[0].filename}")

        if not run:
            return

        with st.spinner("Analyzing Form 5500 records..."):
            result = run_cached_analysis(
                temp_path,
                suffix,
                file_sha256(temp_path),
                settings,
                selected_members,
                combine_selected,
            )
    except (AnalysisError, SchemaError) as exc:
        st.error(str(exc))
        return
    finally:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)

    render_results(result)


@st.cache_data(max_entries=3, ttl="2h", show_spinner=False)
def run_cached_analysis(
    _temp_path: str,
    suffix: str,
    file_hash: str,
    settings: AnalysisSettings,
    selected_members: list[str] | None,
    combine_selected: bool,
):
    """Cache compact deterministic analysis results, not raw uploaded data."""

    del file_hash
    if suffix == ".csv":
        return analyze_csv(_temp_path, settings)
    return analyze_zip(
        _temp_path,
        settings,
        selected_members=selected_members,
        combine_selected=combine_selected,
    )


def render_results(result) -> None:
    """Display analysis results."""

    metadata = result.metadata
    for warning in result.warnings:
        st.warning(warning)

    st.subheader("Results Summary")
    metric_cols = st.columns(5)
    metric_cols[0].metric("Raw Records Parsed", f"{metadata.get('raw_record_count', 0):,}")
    metric_cols[1].metric("Unique Plan Filings Analyzed", f"{metadata.get('deduplicated_record_count', 0):,}")
    metric_cols[2].metric("Anchor Date", str(metadata.get("anchor_date", "")))
    metric_cols[3].metric("Qualified Companies", f"{metadata.get('qualified_company_count', 0):,}")
    metric_cols[4].metric("Qualified Plans", f"{metadata.get('qualified_plan_count', 0):,}")

    st.caption(
        f"Anchor Source: {metadata.get('anchor_source', '')} | "
        f"Records Removed as Duplicates or Superseded Filings: "
        f"{metadata.get('duplicate_record_count', 0):,}"
    )

    diagnostics = metadata.get("filter_diagnostics")
    if diagnostics:
        with st.expander("Filter Diagnostics"):
            diagnostic_labels = {
                "unique_plan_filings_analyzed": "Unique Plan Filings Analyzed",
                "participant_range_count": "Inside Participant Range",
                "target_signal_including_frozen_count": "With 1A or 1C Signal, Including Frozen",
                "frozen_target_count": "Frozen Target Plans",
                "after_signal_filter_count": "After Signal-Type Filter",
                "target_after_frozen_setting_count": "After Frozen-Plan Setting",
                "target_after_state_and_signal_filters_count": "After State and Signal-Type Filters",
                "qualified_plan_count": "Qualified Plans Returned",
                "qualified_company_count": "Qualified Companies Returned",
            }
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Stage": label, "Count": diagnostics.get(key, 0)}
                        for key, label in diagnostic_labels.items()
                    ]
                ),
                width="stretch",
                hide_index=True,
            )

    if metadata.get("uploaded_file_type") == "ZIP":
        with st.expander("ZIP Analysis Metadata"):
            st.json(
                {
                    key: value
                    for key, value in metadata.items()
                    if key
                    in {
                        "uploaded_file_type",
                        "archive_member_count",
                        "csv_candidate_count",
                        "selected_csv_members",
                        "detected_form_type_per_member",
                        "compressed_size_per_member",
                        "uncompressed_size_per_member",
                        "records_parsed_per_member",
                        "qualified_records_per_member",
                    }
                }
            )

    if result.company_leads.empty:
        st.info("No qualifying records matched the selected filters.")
        return

    company_tab, plan_tab = st.tabs(["Company Leads", "Qualifying Plan Details"])
    with company_tab:
        st.download_button(
            "Download Company Leads CSV",
            data=result.company_leads.to_csv(index=False).encode("utf-8"),
            file_name="company_leads.csv",
            mime="text/csv",
        )
        st.dataframe(result.company_leads, use_container_width=True, hide_index=True)
    with plan_tab:
        st.download_button(
            "Download Qualifying Plan Details CSV",
            data=result.plan_details.to_csv(index=False).encode("utf-8"),
            file_name="qualifying_plan_details.csv",
            mime="text/csv",
        )
        st.dataframe(result.plan_details, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
