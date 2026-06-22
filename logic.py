"""Business logic for the Form 5500 Retirement-Plan Signal Finder."""

from __future__ import annotations

import hashlib
import io
import os
import re
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable

import pandas as pd

from schema import OPTIONAL_FIELDS, SchemaError, SchemaResolution, resolve_schema


class AnalysisError(ValueError):
    """Raised for controlled analysis failures suitable for display."""


CHUNKSIZE = 100_000
MAX_ARCHIVE_MEMBERS = 100
MAX_UNCOMPRESSED_CSV_SIZE = 2 * 1024**3
MAX_COMBINED_UNCOMPRESSED_SIZE = 4 * 1024**3
MAX_COMPRESSION_RATIO = 200

CSV_METADATA_NAMES = {".DS_STORE", "THUMBS.DB"}
CONFIDENCE_ORDER = {"Lower": 1, "Medium": 2, "High": 3}
STATUS_GOOD_TOKENS = ("RECEIVED", "ACCEPTED", "SUCCESS", "VALID", "LATEST")
STATUS_BAD_TOKENS = ("ERROR", "REJECT", "STOP", "INVALID", "FAILED", "FAILURE")

PLAN_EXPORT_COLUMNS = [
    "Company Name",
    "EIN",
    "City",
    "State",
    "NAICS Code",
    "Participant Count",
    "Plan Number",
    "Form Year",
    "Pension Codes",
    "Signal Type",
    "Plan Effective Date",
    "Date Received",
    "Is Initial Filing?",
    "Is Frozen?",
    "Recency Basis",
    "Confidence",
    "Confidence Reason",
]

COMPANY_EXPORT_COLUMNS = [
    "Company Name",
    "EIN",
    "City",
    "State",
    "NAICS Code",
    "Participant Count",
    "Qualifying Plan Count",
    "Signal Types",
    "Most Recent Plan Effective Date",
    "Most Recent Filing Received Date",
    "Any Initial Filing?",
    "Highest Confidence",
    "Confidence Reason",
    "Plan Numbers",
    "Pension Codes",
]


@dataclass(frozen=True)
class AnalysisSettings:
    """User-selected filters for an analysis run."""

    min_participants: int = 10
    max_participants: int = 100
    max_plan_age_months: int = 24
    require_recent_effective_date: bool = False
    include_frozen_plans: bool = False
    include_unknown_effective_initial_filings: bool = False
    states: tuple[str, ...] = ()
    signal_types: tuple[str, ...] = ()


@dataclass(frozen=True)
class CsvMemberCandidate:
    """Supported CSV member discovered inside a ZIP archive."""

    filename: str
    detected_form_type: str
    compressed_size: int
    uncompressed_size: int
    crc: int


@dataclass(frozen=True)
class AnalysisResult:
    """Structured output from an analysis run."""

    company_leads: pd.DataFrame
    plan_details: pd.DataFrame
    metadata: dict[str, object]
    warnings: list[str]


def validate_settings(settings: AnalysisSettings) -> None:
    """Validate user-controlled filter settings."""

    if settings.min_participants < 0:
        raise AnalysisError("Minimum Participants must be zero or greater.")
    if settings.max_participants < settings.min_participants:
        raise AnalysisError("Maximum Participants must be greater than or equal to Minimum Participants.")
    if settings.max_plan_age_months < 1 or settings.max_plan_age_months > 120:
        raise AnalysisError("Max Plan Age must be between 1 and 120 months.")


def normalize_ein(series: pd.Series) -> pd.Series:
    """Normalize EIN values as strings while preserving leading zeros."""

    return (
        series.astype("string")
        .fillna("")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def normalize_boolean_flag(series: pd.Series) -> pd.Series:
    """Normalize common truthy filing flags to booleans."""

    truthy = {"1", "Y", "YES", "TRUE", "T"}
    return series.astype("string").fillna("").str.strip().str.upper().isin(truthy)


def parse_dates(series: pd.Series) -> pd.Series:
    """Parse dates without crashing on malformed values."""

    return pd.to_datetime(series, errors="coerce")


def tokenize_pension_codes(series: pd.Series) -> pd.Series:
    """Tokenize pension benefit codes on non-alphanumeric separators."""

    def tokenize(value: object) -> list[str]:
        if pd.isna(value):
            return []
        text = str(value).strip().upper()
        if not text:
            return []
        raw_tokens = [token for token in re.split(r"[^A-Z0-9]+", text) if token]
        tokens: list[str] = []
        for token in raw_tokens:
            if _is_concatenated_dol_code_string(token):
                tokens.extend(token[index : index + 2] for index in range(0, len(token), 2))
            else:
                tokens.append(token)
        return tokens

    return series.apply(tokenize)


def _is_concatenated_dol_code_string(token: str) -> bool:
    """Return true for compact DOL code strings such as 1A1I3D."""

    if len(token) < 4 or len(token) % 2:
        return False
    return all(re.fullmatch(r"[0-9][A-Z]", token[index : index + 2]) for index in range(0, len(token), 2))


def normalize_chunk(
    chunk: pd.DataFrame,
    resolution: SchemaResolution,
    source_member_name: str | None = None,
) -> pd.DataFrame:
    """Normalize a raw chunk into canonical columns and typed helper fields."""

    data: dict[str, pd.Series] = {}
    for canonical, source in resolution.mapping.items():
        data[canonical] = chunk[source]
    for optional in OPTIONAL_FIELDS:
        if optional not in data:
            data[optional] = pd.Series([""] * len(chunk), index=chunk.index)

    df = pd.DataFrame(data)
    text_fields = [
        "company_name",
        "city",
        "state",
        "naics_code",
        "pension_codes",
        "plan_number",
        "form_year",
        "filing_status",
        "amended_indicator",
        "ack_id",
    ]
    for field in text_fields:
        df[field] = df[field].astype("string").fillna("").str.strip()

    df["ein"] = normalize_ein(df["ein"])
    df["ein_normalized"] = df["ein"].str.replace(r"\D", "", regex=True)
    df["participant_count"] = pd.to_numeric(df["participant_count"], errors="coerce")
    df["plan_effective_date"] = parse_dates(df["plan_effective_date"])
    df["date_received"] = parse_dates(df["date_received"])
    df["initial_filing"] = normalize_boolean_flag(df["initial_filing"])
    df["is_amended"] = normalize_boolean_flag(df["amended_indicator"])
    df["source_form_type"] = resolution.source_form_type
    df["archive_member_name"] = source_member_name or ""
    return df


def filter_filing_status(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Retain valid filing statuses when status information is available."""

    if "filing_status" not in df.columns or df["filing_status"].astype("string").str.strip().eq("").all():
        return df.copy(), False

    normalized = df["filing_status"].astype("string").fillna("").str.strip().str.upper()
    bad = normalized.apply(lambda value: any(token in value for token in STATUS_BAD_TOKENS))
    good = normalized.apply(lambda value: value == "" or any(token in value for token in STATUS_GOOD_TOKENS))
    return df.loc[good & ~bad].copy(), True


def determine_anchor_date(df: pd.DataFrame) -> tuple[pd.Timestamp, str]:
    """Determine the dataset-relative recency anchor date."""

    if "date_received" in df.columns:
        valid = df["date_received"].dropna()
        if not valid.empty:
            return valid.max().normalize(), "DATE_RECEIVED"

    if "form_year" in df.columns:
        form_dates = _parse_form_year_dates(df["form_year"])
        valid = form_dates.dropna()
        if not valid.empty:
            return valid.max().normalize(), "FORM_YEAR"

    if "plan_effective_date" in df.columns:
        valid = df["plan_effective_date"].dropna()
        if not valid.empty:
            return valid.max().normalize(), "PLAN_EFF_DATE"

    raise AnalysisError("No valid anchor date could be derived from the uploaded dataset.")


def _parse_form_year_dates(series: pd.Series) -> pd.Series:
    """Parse form-year or tax-period values into dates."""

    text = series.astype("string").fillna("").str.strip()
    year_values = text.str.extract(r"^(\d{4})$")[0]
    dates = pd.to_datetime(year_values + "-12-31", errors="coerce")
    fallback = pd.to_datetime(text, errors="coerce")
    return dates.fillna(fallback)


def deduplicate_filings(df: pd.DataFrame) -> tuple[pd.DataFrame, int, str | None]:
    """Remove duplicate or superseded plan filings using the strongest key."""

    if df.empty:
        return df.copy(), 0, None

    key_fields = ["ein_normalized", "plan_number", "form_year"]
    fallback_message = None
    if df["form_year"].astype("string").str.strip().eq("").all():
        key_fields = ["ein_normalized", "plan_number"]
        fallback_message = "Form year was unavailable, so deduplication used EIN and plan number."
    if df["plan_number"].astype("string").str.strip().eq("").all():
        key_fields = ["ein_normalized", "form_year"]
        fallback_message = "Plan number was unavailable, so deduplication used EIN and form year."
    if df["form_year"].astype("string").str.strip().eq("").all() and df["plan_number"].astype("string").str.strip().eq("").all():
        key_fields = ["ein_normalized"]
        fallback_message = "Plan number and form year were unavailable, so deduplication used EIN only."

    sortable = df.copy()
    sortable["_status_rank"] = 1
    sortable["_date_rank"] = sortable["date_received"].fillna(pd.Timestamp.min)
    sortable["_amended_rank"] = sortable["is_amended"].astype(int)
    sortable["_ack_rank"] = sortable["ack_id"].astype("string").fillna("")
    sortable = sortable.sort_values(
        key_fields + ["_status_rank", "_date_rank", "_amended_rank", "_ack_rank"],
        ascending=[True] * len(key_fields) + [True, True, True, True],
        kind="mergesort",
    )
    deduped = sortable.drop_duplicates(subset=key_fields, keep="last")
    deduped = deduped.drop(columns=["_status_rank", "_date_rank", "_amended_rank", "_ack_rank"])
    removed = len(df) - len(deduped)
    return deduped.reset_index(drop=True), removed, fallback_message


def classify_signal(df: pd.DataFrame) -> pd.DataFrame:
    """Create pension-code booleans and human-readable signal labels."""

    result = df.copy()
    tokens = tokenize_pension_codes(result["pension_codes"])
    result["pension_code_tokens"] = tokens
    result["has_code_1a"] = tokens.apply(lambda values: "1A" in values)
    result["has_code_1c"] = tokens.apply(lambda values: "1C" in values)
    result["has_code_1i"] = tokens.apply(lambda values: "1I" in values)

    def label(row: pd.Series) -> str:
        parts: list[str] = []
        if row["has_code_1c"]:
            parts.append("Cash Balance")
        if row["has_code_1a"]:
            parts.append("Pay-Related Defined Benefit")
        signal = " + ".join(parts)
        if row["has_code_1i"]:
            signal = f"{signal} Frozen" if signal else "Frozen"
        return signal

    result["Signal Type"] = result.apply(label, axis=1)
    return result


def calculate_recency(
    df: pd.DataFrame,
    anchor_date: pd.Timestamp,
    max_age_months: int,
    future_tolerance_days: int = 31,
) -> pd.DataFrame:
    """Evaluate recency using calendar-month subtraction."""

    result = df.copy()
    cutoff_date = anchor_date - pd.DateOffset(months=max_age_months)
    latest_allowed = anchor_date + pd.Timedelta(days=future_tolerance_days)
    valid_effective = result["plan_effective_date"].notna()
    result["passes_recency"] = (
        valid_effective
        & (result["plan_effective_date"] >= cutoff_date)
        & (result["plan_effective_date"] <= latest_allowed)
    )
    result["future_effective_outlier"] = valid_effective & (result["plan_effective_date"] > latest_allowed)
    result["Recency Basis"] = ""
    result.loc[result["passes_recency"], "Recency Basis"] = "Dated Recent Plan"
    result.loc[valid_effective & ~result["passes_recency"], "Recency Basis"] = "Target Plan Filing"
    result.loc[result["plan_effective_date"].isna() & result["initial_filing"], "Recency Basis"] = (
        "Initial Filing; Effective Date Unknown"
    )
    result.loc[result["plan_effective_date"].isna() & ~result["initial_filing"], "Recency Basis"] = (
        "Effective Date Unknown"
    )
    result.attrs["cutoff_date"] = cutoff_date
    return result


def assign_confidence(df: pd.DataFrame) -> pd.DataFrame:
    """Assign transparent confidence classes and reasons."""

    result = df.copy()
    if result.empty:
        result["Confidence"] = pd.Series(dtype="string")
        result["Confidence Reason"] = pd.Series(dtype="string")
        result["confidence_rank"] = pd.Series(dtype="float")
        return result

    def classify(row: pd.Series) -> tuple[str, str]:
        if row["has_code_1c"] and row["passes_recency"] and row["initial_filing"] and not row["has_code_1i"]:
            return "High", "Cash Balance code 1C, recent effective date, initial filing, and not frozen."
        if row["passes_recency"] and row["has_code_1c"]:
            return "Medium", "Cash Balance code 1C with a valid recent effective date."
        if row["passes_recency"] and row["has_code_1a"]:
            return "Medium", "Pay-related Defined Benefit code 1A with a valid recent effective date."
        if row["initial_filing"] and pd.isna(row["plan_effective_date"]):
            return "Lower", "Target pension code with initial filing flag and unknown effective date."
        if row["has_code_1c"]:
            return "Lower", "Cash Balance code 1C reported, but plan effective date is outside the recent-date window."
        if row["has_code_1a"]:
            return "Lower", "Pay-related Defined Benefit code 1A reported, but plan effective date is outside the recent-date window."
        return "Lower", "Target pension code with limited supporting recency information."

    classified = result.apply(classify, axis=1, result_type="expand")
    result["Confidence"] = classified[0]
    result["Confidence Reason"] = classified[1]
    result["confidence_rank"] = result["Confidence"].map(CONFIDENCE_ORDER).fillna(0)
    return result


def aggregate_company_leads(plan_details: pd.DataFrame) -> pd.DataFrame:
    """Aggregate qualifying plan records into company-level leads."""

    if plan_details.empty:
        return pd.DataFrame(columns=COMPANY_EXPORT_COLUMNS)

    rows: list[dict[str, object]] = []
    for _, group in plan_details.groupby("ein_normalized", dropna=False):
        ordered = group.sort_values(
            ["plan_effective_date", "date_received"], ascending=[True, True], na_position="first"
        )
        most_recent = ordered.iloc[-1]
        top_confidence = group.sort_values(
            ["confidence_rank", "plan_effective_date"], ascending=[False, False], na_position="last"
        ).iloc[0]
        rows.append(
            {
                "Company Name": _latest_non_empty(group, "company_name"),
                "EIN": _latest_non_empty(group, "ein"),
                "City": _latest_non_empty(group, "city"),
                "State": _latest_non_empty(group, "state"),
                "NAICS Code": _latest_non_empty(group, "naics_code"),
                "Participant Count": group["participant_count"].max(),
                "Qualifying Plan Count": group["plan_number"].replace("", pd.NA).nunique(dropna=True)
                or len(group),
                "Signal Types": _join_distinct(group["Signal Type"]),
                "Most Recent Plan Effective Date": most_recent["plan_effective_date"],
                "Most Recent Filing Received Date": group["date_received"].max(),
                "Any Initial Filing?": bool(group["initial_filing"].any()),
                "Highest Confidence": top_confidence["Confidence"],
                "Confidence Reason": top_confidence["Confidence Reason"],
                "Plan Numbers": _join_distinct(group["plan_number"]),
                "Pension Codes": _join_distinct(group["pension_codes"]),
            }
        )

    leads = pd.DataFrame(rows)
    leads["_confidence_rank"] = leads["Highest Confidence"].map(CONFIDENCE_ORDER).fillna(0)
    leads = leads.sort_values(
        ["_confidence_rank", "Most Recent Plan Effective Date", "Participant Count"],
        ascending=[False, False, False],
        na_position="last",
    ).drop(columns=["_confidence_rank"])
    return leads.reset_index(drop=True)


def _latest_non_empty(group: pd.DataFrame, column: str) -> str:
    """Return the most recent non-empty value in a group."""

    ordered = group.sort_values(["date_received", "plan_effective_date"], ascending=[True, True], na_position="first")
    values = ordered[column].astype("string").fillna("").str.strip()
    values = values[values != ""]
    return "" if values.empty else str(values.iloc[-1])


def _join_distinct(series: pd.Series) -> str:
    """Join distinct non-empty string values."""

    values = [str(value).strip() for value in series.dropna().tolist() if str(value).strip()]
    return "; ".join(dict.fromkeys(values))


def format_plan_export(df: pd.DataFrame) -> pd.DataFrame:
    """Format plan details for display and CSV export."""

    if df.empty:
        return pd.DataFrame(columns=PLAN_EXPORT_COLUMNS)
    export = pd.DataFrame(
        {
            "Company Name": df["company_name"],
            "EIN": df["ein"].astype("string"),
            "City": df["city"],
            "State": df["state"],
            "NAICS Code": df["naics_code"],
            "Participant Count": df["participant_count"],
            "Plan Number": df["plan_number"],
            "Form Year": df["form_year"],
            "Pension Codes": df["pension_codes"],
            "Signal Type": df["Signal Type"],
            "Plan Effective Date": _format_date_series(df["plan_effective_date"]),
            "Date Received": _format_date_series(df["date_received"]),
            "Is Initial Filing?": df["initial_filing"],
            "Is Frozen?": df["has_code_1i"],
            "Recency Basis": df["Recency Basis"],
            "Confidence": df["Confidence"],
            "Confidence Reason": df["Confidence Reason"],
        }
    )
    if "archive_member_name" in df.columns and df["archive_member_name"].astype("string").str.strip().ne("").any():
        export["Archive Member Name"] = df["archive_member_name"]
    return export


def format_company_export(df: pd.DataFrame) -> pd.DataFrame:
    """Format company leads for display and CSV export."""

    if df.empty:
        return pd.DataFrame(columns=COMPANY_EXPORT_COLUMNS)
    export = df.copy()
    for column in ["Most Recent Plan Effective Date", "Most Recent Filing Received Date"]:
        export[column] = _format_date_series(export[column])
    export["EIN"] = export["EIN"].astype("string")
    return export[COMPANY_EXPORT_COLUMNS]


def analyze_csv(
    file_path: str | Path,
    settings: AnalysisSettings,
    source_member_name: str | None = None,
) -> AnalysisResult:
    """Analyze one CSV file using a two-pass chunked workflow."""

    validate_settings(settings)
    file_path = Path(file_path)
    resolution = _resolve_csv_file_schema(file_path)
    usecols = list(dict.fromkeys(resolution.mapping.values()))

    raw_count = 0
    anchor_frames: list[pd.DataFrame] = []
    status_frames: list[pd.DataFrame] = []
    status_available = False
    post_status_count = 0
    duplicate_count = 0
    dedupe_warning: str | None = None

    try:
        for chunk in pd.read_csv(file_path, dtype=str, usecols=usecols, chunksize=CHUNKSIZE):
            if chunk.empty:
                continue
            normalized = normalize_chunk(chunk, resolution, source_member_name)
            raw_count += len(normalized)
            anchor_frames.append(normalized[["date_received", "form_year", "plan_effective_date"]].copy())
    except pd.errors.EmptyDataError as exc:
        raise AnalysisError("The uploaded CSV file is empty.") from exc
    except ValueError as exc:
        raise AnalysisError(f"The uploaded CSV could not be read: {exc}") from exc

    if raw_count == 0:
        raise AnalysisError("The uploaded CSV file contains no records.")

    anchor_input = pd.concat(anchor_frames, ignore_index=True)
    anchor_date, anchor_source = determine_anchor_date(anchor_input)

    try:
        for chunk in pd.read_csv(file_path, dtype=str, usecols=usecols, chunksize=CHUNKSIZE):
            normalized = normalize_chunk(chunk, resolution, source_member_name)
            status_filtered, chunk_status_available = filter_filing_status(normalized)
            status_available = status_available or chunk_status_available
            post_status_count += len(status_filtered)
            chunk_deduped, chunk_duplicate_count, chunk_dedupe_warning = deduplicate_filings(status_filtered)
            duplicate_count += chunk_duplicate_count
            dedupe_warning = dedupe_warning or chunk_dedupe_warning
            if not chunk_deduped.empty:
                status_frames.append(chunk_deduped)
    except pd.errors.ParserError as exc:
        raise AnalysisError(f"The uploaded CSV could not be parsed: {exc}") from exc

    status_records = _concat_or_empty(status_frames)
    if status_records.empty:
        warnings = _metadata_warnings(status_available, anchor_source, dedupe_warning)
        metadata = _base_metadata(
            raw_count,
            post_status_count,
            0,
            duplicate_count,
            resolution,
            anchor_date,
            anchor_source,
            warnings,
        )
        return AnalysisResult(
            company_leads=pd.DataFrame(columns=COMPANY_EXPORT_COLUMNS),
            plan_details=pd.DataFrame(columns=PLAN_EXPORT_COLUMNS),
            metadata=metadata,
            warnings=warnings,
        )

    deduped, final_duplicate_count, final_dedupe_warning = deduplicate_filings(status_records)
    duplicate_count += final_duplicate_count
    dedupe_warning = dedupe_warning or final_dedupe_warning
    filter_diagnostics = calculate_filter_diagnostics(deduped, settings)
    filtered = _prefilter_candidates(deduped, settings)
    final_plans = _finalize_candidates(filtered, settings, anchor_date)
    company_leads = aggregate_company_leads(final_plans)
    plan_export = format_plan_export(final_plans)
    company_export = format_company_export(company_leads)
    warnings = _metadata_warnings(status_available, anchor_source, dedupe_warning)
    metadata = _base_metadata(
        raw_count,
        post_status_count,
        len(deduped),
        duplicate_count,
        resolution,
        anchor_date,
        anchor_source,
        warnings,
    )
    metadata["qualified_plan_count"] = len(final_plans)
    metadata["qualified_company_count"] = len(company_export)
    filter_diagnostics["qualified_plan_count"] = len(final_plans)
    filter_diagnostics["qualified_company_count"] = len(company_export)
    metadata["filter_diagnostics"] = filter_diagnostics
    if source_member_name:
        metadata["selected_csv_members"] = [source_member_name]
        metadata["records_parsed_per_member"] = {source_member_name: raw_count}
        metadata["qualified_records_per_member"] = {source_member_name: len(final_plans)}
    return AnalysisResult(company_export, plan_export, metadata, warnings)


def _prefilter_candidates(df: pd.DataFrame, settings: AnalysisSettings) -> pd.DataFrame:
    """Apply filters that do not require deduplication or final recency."""

    if df.empty:
        return df.copy()
    filtered = df[
        (df["participant_count"] >= settings.min_participants)
        & (df["participant_count"] <= settings.max_participants)
    ].copy()
    if filtered.empty:
        return filtered
    filtered = classify_signal(filtered)
    filtered = filtered[filtered["has_code_1a"] | filtered["has_code_1c"]].copy()
    if not settings.include_frozen_plans:
        filtered = filtered[~filtered["has_code_1i"]].copy()
    if settings.states:
        selected = {state.upper() for state in settings.states}
        filtered = filtered[filtered["state"].astype("string").str.upper().isin(selected)].copy()
    if settings.signal_types:
        filtered = _filter_signal_types(filtered, settings.signal_types)
    return filtered


def calculate_filter_diagnostics(df: pd.DataFrame, settings: AnalysisSettings) -> dict[str, int]:
    """Return count diagnostics for the main filtering stages."""

    if df.empty:
        return {
            "unique_plan_filings_analyzed": 0,
            "participant_range_count": 0,
            "target_signal_including_frozen_count": 0,
            "frozen_target_count": 0,
            "target_after_frozen_setting_count": 0,
            "target_after_state_and_signal_filters_count": 0,
            "qualified_plan_count": 0,
            "qualified_company_count": 0,
        }

    classified = classify_signal(df)
    participant_match = classified[
        classified["participant_count"].notna()
        & (classified["participant_count"] >= settings.min_participants)
        & (classified["participant_count"] <= settings.max_participants)
    ].copy()
    target = participant_match[participant_match["has_code_1a"] | participant_match["has_code_1c"]].copy()
    frozen_target = target[target["has_code_1i"]]
    after_frozen = target if settings.include_frozen_plans else target[~target["has_code_1i"]]
    after_optional = after_frozen
    if settings.states:
        selected = {state.upper() for state in settings.states}
        after_optional = after_optional[after_optional["state"].astype("string").str.upper().isin(selected)]
    if settings.signal_types:
        after_optional = _filter_signal_types(after_optional, settings.signal_types)

    return {
        "unique_plan_filings_analyzed": len(classified),
        "participant_range_count": len(participant_match),
        "target_signal_including_frozen_count": len(target),
        "frozen_target_count": len(frozen_target),
        "target_after_frozen_setting_count": len(after_frozen),
        "target_after_state_and_signal_filters_count": len(after_optional),
        "qualified_plan_count": 0,
        "qualified_company_count": 0,
    }


def _filter_signal_types(df: pd.DataFrame, signal_types: Iterable[str]) -> pd.DataFrame:
    """Apply selected signal-type filters."""

    selected = set(signal_types)
    mask = pd.Series(False, index=df.index)
    if "Cash Balance" in selected:
        mask |= df["has_code_1c"]
    if "Pay-Related Defined Benefit" in selected:
        mask |= df["has_code_1a"]
    if "Both" in selected:
        mask |= df["has_code_1a"] & df["has_code_1c"]
    return df[mask].copy()


def _finalize_candidates(
    df: pd.DataFrame,
    settings: AnalysisSettings,
    anchor_date: pd.Timestamp,
) -> pd.DataFrame:
    """Apply final recency/fallback rules and confidence ranking."""

    if df.empty:
        return df.copy()
    recent = calculate_recency(df, anchor_date, settings.max_plan_age_months)
    include_unknown = settings.include_unknown_effective_initial_filings
    if settings.require_recent_effective_date:
        mask = recent["passes_recency"].copy()
        if include_unknown:
            mask |= recent["plan_effective_date"].isna() & recent["initial_filing"]
    else:
        mask = pd.Series(True, index=recent.index)
        if not include_unknown:
            mask &= recent["plan_effective_date"].notna()
    mask &= ~recent["future_effective_outlier"]
    qualified = recent[mask].copy()
    if qualified.empty:
        return assign_confidence(qualified)
    return assign_confidence(qualified).sort_values(
        ["confidence_rank", "plan_effective_date", "participant_count"],
        ascending=[False, False, False],
        na_position="last",
    )


def _base_metadata(
    raw_count: int,
    post_status_count: int,
    deduplicated_count: int,
    duplicate_count: int,
    resolution: SchemaResolution,
    anchor_date: pd.Timestamp,
    anchor_source: str,
    warnings: list[str],
) -> dict[str, object]:
    """Build common metadata for analysis output."""

    return {
        "raw_record_count": raw_count,
        "post_status_record_count": post_status_count,
        "deduplicated_record_count": deduplicated_count,
        "duplicate_record_count": duplicate_count,
        "qualified_plan_count": 0,
        "qualified_company_count": 0,
        "anchor_date": anchor_date.strftime("%Y-%m-%d"),
        "anchor_source": anchor_source,
        "schema_mapping": resolution.mapping,
        "source_form_type": resolution.source_form_type,
        "warnings": warnings,
    }


def _metadata_warnings(
    status_available: bool,
    anchor_source: str,
    dedupe_warning: str | None,
) -> list[str]:
    """Return non-fatal analysis warnings."""

    warnings: list[str] = []
    if not status_available:
        warnings.append("Filing status was unavailable, so invalid filing records could not be excluded.")
    if anchor_source != "DATE_RECEIVED":
        warnings.append(f"The anchor date was derived from {anchor_source} because DATE_RECEIVED was unavailable.")
    if dedupe_warning:
        warnings.append(dedupe_warning)
    return warnings


def _resolve_csv_file_schema(file_path: Path) -> SchemaResolution:
    """Read only the header and resolve the CSV schema."""

    try:
        header = pd.read_csv(file_path, dtype=str, nrows=0)
    except pd.errors.EmptyDataError as exc:
        raise AnalysisError("The uploaded CSV file is empty.") from exc
    except UnicodeDecodeError as exc:
        raise AnalysisError("The uploaded CSV encoding could not be read as text.") from exc
    return resolve_schema(header.columns)


def _concat_or_empty(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate frames or return an empty frame."""

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _format_date_series(series: pd.Series) -> pd.Series:
    """Format date-like values as YYYY-MM-DD strings."""

    dates = pd.to_datetime(series, errors="coerce")
    return dates.dt.strftime("%Y-%m-%d").fillna("")


def inspect_zip_archive(file_path: str | Path) -> dict[str, object]:
    """Validate a ZIP archive and return supported CSV candidates."""

    path = Path(file_path)
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            validate_zip_infos(infos, path.stat().st_size)
            candidates = discover_csv_members(archive)
    except zipfile.BadZipFile as exc:
        raise AnalysisError("The uploaded file is not a readable ZIP archive.") from exc
    return {
        "uploaded_file_type": "ZIP",
        "archive_member_count": len(infos),
        "csv_candidate_count": len(candidates),
        "candidates": candidates,
    }


def validate_zip_infos(infos: list[zipfile.ZipInfo], archive_size: int) -> None:
    """Validate ZIP metadata before any member is processed."""

    if len(infos) > MAX_ARCHIVE_MEMBERS:
        raise AnalysisError(f"ZIP archive exceeds the maximum member count of {MAX_ARCHIVE_MEMBERS}.")

    combined_size = 0
    for info in infos:
        name = info.filename
        if _is_ignored_zip_member(name):
            continue
        if info.flag_bits & 0x1:
            raise AnalysisError("Encrypted or password-protected ZIP archives are not supported.")
        if _is_unsafe_zip_path(name):
            raise AnalysisError(f"ZIP archive contains an unsafe path: {name}.")
        if name.lower().endswith(".zip"):
            raise AnalysisError("Nested ZIP archives are not supported.")
        if info.is_dir():
            continue
        combined_size += info.file_size
        if info.file_size > MAX_UNCOMPRESSED_CSV_SIZE and name.lower().endswith(".csv"):
            raise AnalysisError("A CSV member exceeds the maximum uncompressed size limit.")
        if info.compress_size > 0 and info.file_size / info.compress_size > MAX_COMPRESSION_RATIO:
            raise AnalysisError("A ZIP member exceeds the maximum compression ratio limit.")
        if info.compress_size == 0 and info.file_size > 0:
            raise AnalysisError("A ZIP member has an unsafe compression ratio.")
    if combined_size > MAX_COMBINED_UNCOMPRESSED_SIZE:
        raise AnalysisError("ZIP archive exceeds the maximum combined uncompressed size limit.")
    if archive_size <= 0:
        raise AnalysisError("The uploaded ZIP archive is empty.")


def discover_csv_members(archive: zipfile.ZipFile) -> list[CsvMemberCandidate]:
    """Find supported Form 5500 CSV files in a validated ZIP archive."""

    csv_infos = [
        info
        for info in archive.infolist()
        if not info.is_dir()
        and not _is_ignored_zip_member(info.filename)
        and info.filename.lower().endswith(".csv")
    ]
    if not csv_infos:
        raise AnalysisError("The ZIP archive contains no CSV files.")

    candidates: list[CsvMemberCandidate] = []
    for info in csv_infos:
        resolution = detect_member_schema(archive, info)
        if resolution is None:
            continue
        candidates.append(
            CsvMemberCandidate(
                filename=info.filename,
                detected_form_type=resolution.source_form_type,
                compressed_size=info.compress_size,
                uncompressed_size=info.file_size,
                crc=info.CRC,
            )
        )
    if not candidates:
        raise AnalysisError("The ZIP archive contains no supported Form 5500 CSV files.")
    return candidates


def detect_member_schema(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
) -> SchemaResolution | None:
    """Inspect a CSV member header and return a schema resolution if supported."""

    try:
        with archive.open(info) as member:
            header_bytes = member.readline(256 * 1024)
        header_text = header_bytes.decode("utf-8-sig")
        columns = pd.read_csv(io.StringIO(header_text), nrows=0).columns
        return resolve_schema(columns)
    except (SchemaError, UnicodeDecodeError, pd.errors.ParserError):
        return None
    except RuntimeError as exc:
        if "password" in str(exc).lower():
            raise AnalysisError("Encrypted or password-protected ZIP archives are not supported.") from exc
        raise


def copy_zip_member_to_tempfile(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> str:
    """Copy one validated ZIP CSV member to a managed temporary file."""

    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    try:
        with temp:
            with archive.open(info) as source:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    temp.write(chunk)
        return temp.name
    except RuntimeError as exc:
        temp.close()
        _remove_tempfile(temp.name)
        if "password" in str(exc).lower():
            raise AnalysisError("Encrypted or password-protected ZIP archives are not supported.") from exc
        raise AnalysisError(f"CSV member could not be read: {info.filename}.") from exc


def analyze_zip(
    file_path: str | Path,
    settings: AnalysisSettings,
    selected_members: list[str] | None = None,
    combine_selected: bool = False,
) -> AnalysisResult:
    """Analyze one or more supported CSV members inside a ZIP archive."""

    validate_settings(settings)
    path = Path(file_path)
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            validate_zip_infos(infos, path.stat().st_size)
            candidates = discover_csv_members(archive)
            candidate_by_name = {candidate.filename: candidate for candidate in candidates}
            if selected_members is None:
                if len(candidates) == 1:
                    selected_members = [candidates[0].filename]
                else:
                    raise AnalysisError("Multiple supported CSV files were found. Select one before running analysis.")
            if not selected_members:
                raise AnalysisError("No supported CSV member was selected.")
            if len(selected_members) > 1 and not combine_selected:
                raise AnalysisError("Select a single CSV member or enable Combine Selected CSV Files.")

            selected_infos = []
            for member_name in selected_members:
                if member_name not in candidate_by_name:
                    raise AnalysisError(f"Selected CSV member is not supported: {member_name}.")
                selected_infos.append(archive.getinfo(member_name))

            member_results: list[AnalysisResult] = []
            temp_paths: list[str] = []
            try:
                for info in selected_infos:
                    temp_path = copy_zip_member_to_tempfile(archive, info)
                    temp_paths.append(temp_path)
                    member_results.append(analyze_csv(temp_path, settings, source_member_name=info.filename))
            finally:
                for temp_path in temp_paths:
                    _remove_tempfile(temp_path)
    except zipfile.BadZipFile as exc:
        raise AnalysisError("The uploaded file is not a readable ZIP archive.") from exc

    if len(member_results) == 1:
        result = member_results[0]
        metadata = dict(result.metadata)
        metadata.update(_zip_metadata(path, infos, candidates, selected_members))
        metadata["compressed_size_per_member"] = {
            name: candidate_by_name[name].compressed_size for name in selected_members
        }
        metadata["uncompressed_size_per_member"] = {
            name: candidate_by_name[name].uncompressed_size for name in selected_members
        }
        return AnalysisResult(result.company_leads, result.plan_details, metadata, result.warnings)

    return _combine_member_results(path, infos, candidates, selected_members, member_results, settings)


def _combine_member_results(
    path: Path,
    infos: list[zipfile.ZipInfo],
    candidates: list[CsvMemberCandidate],
    selected_members: list[str],
    member_results: list[AnalysisResult],
    settings: AnalysisSettings,
) -> AnalysisResult:
    """Combine normalized candidate rows from multiple compatible ZIP members."""

    candidate_by_name = {candidate.filename: candidate for candidate in candidates}
    raw = sum(int(result.metadata["raw_record_count"]) for result in member_results)
    post_status = sum(int(result.metadata["post_status_record_count"]) for result in member_results)
    frames = []
    for result in member_results:
        if not result.plan_details.empty:
            frames.append(result.plan_details)

    if not frames:
        first = member_results[0]
        metadata = dict(first.metadata)
        metadata.update(_zip_metadata(path, infos, candidates, selected_members))
        metadata["raw_record_count"] = raw
        metadata["post_status_record_count"] = post_status
        return AnalysisResult(first.company_leads, first.plan_details, metadata, first.warnings)

    plan_export = pd.concat(frames, ignore_index=True)
    raw_plans = _plan_export_to_internal(plan_export)
    deduped, duplicate_count, dedupe_warning = deduplicate_filings(raw_plans)
    anchor_date, anchor_source = determine_anchor_date(deduped)
    final_plans = _finalize_candidates(deduped, settings, anchor_date)
    company_export = format_company_export(aggregate_company_leads(final_plans))
    final_plan_export = format_plan_export(final_plans)
    warnings = list(dict.fromkeys(warning for result in member_results for warning in result.warnings))
    if dedupe_warning:
        warnings.append(dedupe_warning)
    metadata = {
        "raw_record_count": raw,
        "post_status_record_count": post_status,
        "deduplicated_record_count": len(deduped),
        "duplicate_record_count": duplicate_count,
        "qualified_plan_count": len(final_plan_export),
        "qualified_company_count": len(company_export),
        "anchor_date": anchor_date.strftime("%Y-%m-%d"),
        "anchor_source": anchor_source,
        "source_form_type": "Multiple",
        "warnings": warnings,
    }
    metadata.update(_zip_metadata(path, infos, candidates, selected_members))
    metadata["compressed_size_per_member"] = {
        name: candidate_by_name[name].compressed_size for name in selected_members
    }
    metadata["uncompressed_size_per_member"] = {
        name: candidate_by_name[name].uncompressed_size for name in selected_members
    }
    metadata["records_parsed_per_member"] = {
        name: int(result.metadata["raw_record_count"]) for name, result in zip(selected_members, member_results)
    }
    metadata["qualified_records_per_member"] = {
        name: int(result.metadata["qualified_plan_count"]) for name, result in zip(selected_members, member_results)
    }
    return AnalysisResult(company_export, final_plan_export, metadata, warnings)


def _plan_export_to_internal(plan_export: pd.DataFrame) -> pd.DataFrame:
    """Convert formatted plan export rows back to internal columns for combined dedupe."""

    df = pd.DataFrame(
        {
            "company_name": plan_export["Company Name"],
            "ein": plan_export["EIN"],
            "city": plan_export["City"],
            "state": plan_export["State"],
            "naics_code": plan_export["NAICS Code"],
            "participant_count": plan_export["Participant Count"],
            "plan_number": plan_export["Plan Number"],
            "form_year": plan_export["Form Year"],
            "pension_codes": plan_export["Pension Codes"],
            "plan_effective_date": parse_dates(plan_export["Plan Effective Date"]),
            "date_received": parse_dates(plan_export["Date Received"]),
            "initial_filing": plan_export["Is Initial Filing?"].astype(bool),
            "has_code_1i": plan_export["Is Frozen?"].astype(bool),
            "archive_member_name": plan_export.get("Archive Member Name", ""),
            "filing_status": "",
            "amended_indicator": "",
            "ack_id": "",
            "is_amended": False,
            "source_form_type": "",
        }
    )
    df["ein_normalized"] = normalize_ein(df["ein"]).str.replace(r"\D", "", regex=True)
    df = classify_signal(df)
    return assign_confidence(calculate_recency(df, determine_anchor_date(df)[0], 120))


def _zip_metadata(
    path: Path,
    infos: list[zipfile.ZipInfo],
    candidates: list[CsvMemberCandidate],
    selected_members: list[str],
) -> dict[str, object]:
    """Return ZIP-specific display metadata."""

    candidate_by_name = {candidate.filename: candidate for candidate in candidates}
    return {
        "uploaded_file_type": "ZIP",
        "archive_member_count": len(infos),
        "csv_candidate_count": len(candidates),
        "selected_csv_members": selected_members,
        "detected_form_type_per_member": {
            name: candidate_by_name[name].detected_form_type for name in selected_members
        },
        "uploaded_zip_hash": file_sha256(path),
    }


def file_sha256(file_path: str | Path) -> str:
    """Return a stable SHA-256 hash for cache keys and ZIP identity."""

    digest = hashlib.sha256()
    with open(file_path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_uploaded_file_to_temp(uploaded: BinaryIO, suffix: str) -> str:
    """Write an uploaded file-like object to a managed temporary file."""

    temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        with temp:
            uploaded.seek(0)
            while True:
                chunk = uploaded.read(1024 * 1024)
                if not chunk:
                    break
                temp.write(chunk)
        return temp.name
    except OSError as exc:
        temp.close()
        _remove_tempfile(temp.name)
        raise AnalysisError("The uploaded file could not be saved for analysis.") from exc


def _remove_tempfile(path: str) -> None:
    """Remove a temporary file if it exists."""

    try:
        os.unlink(path)
    except FileNotFoundError:
        return


def _is_ignored_zip_member(name: str) -> bool:
    """Return true for OS metadata entries."""

    clean = name.strip("/")
    return clean.startswith("__MACOSX/") or Path(clean).name.upper() in CSV_METADATA_NAMES


def _is_unsafe_zip_path(name: str) -> bool:
    """Reject absolute paths and traversal inside ZIP archives."""

    path = Path(name)
    return path.is_absolute() or ".." in path.parts
