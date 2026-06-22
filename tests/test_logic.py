from __future__ import annotations

import os
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from logic import (
    AnalysisError,
    AnalysisSettings,
    MAX_COMPRESSION_RATIO,
    analyze_csv,
    analyze_zip,
    calculate_recency,
    classify_signal,
    copy_zip_member_to_tempfile,
    determine_anchor_date,
    discover_csv_members,
    discover_state_options_csv,
    discover_state_options_zip,
    inspect_zip_archive,
    normalize_boolean_flag,
    normalize_chunk,
    normalize_ein,
    parse_dates,
    tokenize_pension_codes,
    validate_zip_infos,
)
from schema import SchemaError, resolve_schema


FIXTURES = Path(__file__).parent / "fixtures"
ALL_SIGNAL_TYPES = ("Cash Balance", "Pay-Related Defined Benefit")


def test_form_5500_aliases_resolve() -> None:
    resolution = resolve_schema(pd.read_csv(FIXTURES / "sample_form_5500.csv", nrows=0).columns)
    assert resolution.mapping["company_name"] == "SPONSOR_DFE_NAME"
    assert resolution.mapping["ein"] == "SPONS_DFE_EIN"
    assert resolution.source_form_type == "Form 5500"


def test_2025_all_layout_aliases_resolve() -> None:
    columns = [
        "ACK_ID",
        "FORM_TAX_PRD",
        "INITIAL_FILING_IND",
        "AMENDED_IND",
        "SPONS_DFE_PN",
        "PLAN_EFF_DATE",
        "SPONSOR_DFE_NAME",
        "SPONS_DFE_EIN",
        "SPONS_DFE_LOC_US_CITY",
        "SPONS_DFE_LOC_US_STATE",
        "BUSINESS_CODE",
        "TOT_ACTIVE_PARTCP_CNT",
        "TYPE_PENSION_BNFT_CODE",
        "FILING_STATUS",
        "DATE_RECEIVED",
    ]
    resolution = resolve_schema(columns)
    assert resolution.mapping["plan_number"] == "SPONS_DFE_PN"
    assert resolution.mapping["form_year"] == "FORM_TAX_PRD"


def test_form_5500_sf_aliases_resolve() -> None:
    resolution = resolve_schema(pd.read_csv(FIXTURES / "sample_form_5500_sf.csv", nrows=0).columns)
    assert resolution.mapping["company_name"] == "SF_SPONSOR_NAME"
    assert resolution.mapping["ein"] == "SF_SPONS_EIN"
    assert resolution.source_form_type == "Form 5500-SF"


def test_state_options_are_discovered_from_csv() -> None:
    states = discover_state_options_csv(FIXTURES / "sample_form_5500.csv")
    assert {"AZ", "CO", "ID", "NV", "OR", "TX", "UT", "WA"}.issubset(states)


def test_ein_normalization_preserves_leading_zero_and_cleans_dot_zero() -> None:
    values = normalize_ein(pd.Series(["012345678", "000123456.0"]))
    assert values.tolist() == ["012345678", "000123456"]


def test_invalid_participants_become_missing_and_do_not_pass_filter() -> None:
    df = pd.DataFrame({"count": ["abc", "", "10"]})
    parsed = pd.to_numeric(df["count"], errors="coerce")
    assert parsed.isna().tolist() == [True, True, False]
    assert ((parsed >= 10) & (parsed <= 100)).tolist() == [False, False, True]


def test_participant_boundaries_are_inclusive() -> None:
    result = analyze_csv(FIXTURES / "sample_form_5500_sf.csv", AnalysisSettings(10, 12, 24))
    assert set(result.company_leads["Participant Count"].astype(int)) == {10, 12}


def test_exact_pension_code_matching() -> None:
    tokens = tokenize_pension_codes(pd.Series(["1A,1C", "1A 1I", "1C; 2E", "11A", "1A1I3D"]))
    assert tokens.tolist() == [["1A", "1C"], ["1A", "1I"], ["1C", "2E"], ["11A"], ["1A", "1I", "3D"]]
    classified = classify_signal(pd.DataFrame({"pension_codes": ["1A", "1C", "11A", "1A1I3D"]}))
    assert classified["has_code_1a"].tolist() == [True, False, False, True]
    assert classified["has_code_1c"].tolist() == [False, True, False, False]
    assert classified["has_code_1i"].tolist() == [False, False, False, True]


def test_frozen_excluded_by_default_and_included_when_enabled() -> None:
    default = analyze_csv(
        FIXTURES / "sample_form_5500.csv",
        AnalysisSettings(signal_types=ALL_SIGNAL_TYPES),
    )
    included = analyze_csv(
        FIXTURES / "sample_form_5500.csv",
        AnalysisSettings(include_frozen_plans=True, signal_types=ALL_SIGNAL_TYPES),
    )
    assert "Beta Tooling" not in set(default.company_leads["Company Name"])
    assert "Beta Tooling" in set(included.company_leads["Company Name"])


def test_malformed_dates_become_nat() -> None:
    parsed = parse_dates(pd.Series(["not-a-date", "2024-01-01"]))
    assert pd.isna(parsed.iloc[0])
    assert parsed.iloc[1] == pd.Timestamp("2024-01-01")


def test_anchor_date_prefers_date_received_and_falls_back() -> None:
    df = pd.DataFrame(
        {
            "date_received": pd.to_datetime(["2025-01-01", "2025-02-01"]),
            "form_year": ["2024", "2023"],
            "plan_effective_date": pd.to_datetime(["2024-01-01", "2024-02-01"]),
        }
    )
    assert determine_anchor_date(df) == (pd.Timestamp("2025-02-01"), "DATE_RECEIVED")
    fallback = df.assign(date_received=pd.NaT)
    assert determine_anchor_date(fallback) == (pd.Timestamp("2024-12-31"), "FORM_YEAR")


def test_future_outliers_are_excluded() -> None:
    result = analyze_csv(
        FIXTURES / "sample_form_5500.csv",
        AnalysisSettings(signal_types=ALL_SIGNAL_TYPES),
    )
    assert "Future Co" not in set(result.company_leads["Company Name"])


def test_recent_effective_date_filter_is_optional() -> None:
    relaxed = analyze_csv(
        FIXTURES / "sample_form_5500.csv",
        AnalysisSettings(max_plan_age_months=12, signal_types=ALL_SIGNAL_TYPES),
    )
    strict = analyze_csv(
        FIXTURES / "sample_form_5500.csv",
        AnalysisSettings(
            max_plan_age_months=12,
            require_recent_effective_date=True,
            signal_types=ALL_SIGNAL_TYPES,
        ),
    )
    assert "Delta Labs" in set(relaxed.company_leads["Company Name"])
    assert "Delta Labs" not in set(strict.company_leads["Company Name"])
    row = relaxed.plan_details[relaxed.plan_details["Company Name"] == "Delta Labs"].iloc[0]
    assert row["Recency Basis"] == "Target Plan Filing"


def test_calendar_month_subtraction_is_used() -> None:
    df = pd.DataFrame(
        {
            "plan_effective_date": pd.to_datetime(["2024-02-29", "2024-03-01"]),
            "initial_filing": [False, False],
        }
    )
    result = calculate_recency(df, pd.Timestamp("2024-03-31"), 1)
    assert result["passes_recency"].tolist() == [True, True]


def test_initial_filing_flag_variants() -> None:
    values = normalize_boolean_flag(pd.Series([1, "1", "Y", "YES", "TRUE", True, "N"]))
    assert values.tolist() == [True, True, True, True, True, True, False]


def test_missing_date_initial_filings_default_and_enabled() -> None:
    default = analyze_csv(
        FIXTURES / "sample_form_5500.csv",
        AnalysisSettings(signal_types=ALL_SIGNAL_TYPES),
    )
    enabled = analyze_csv(
        FIXTURES / "sample_form_5500.csv",
        AnalysisSettings(
            include_unknown_effective_initial_filings=True,
            signal_types=ALL_SIGNAL_TYPES,
        ),
    )
    assert "Missing Date Co" not in set(default.company_leads["Company Name"])
    assert "Missing Date Co" in set(enabled.company_leads["Company Name"])
    row = enabled.plan_details[enabled.plan_details["Company Name"] == "Missing Date Co"].iloc[0]
    assert row["Recency Basis"] == "Initial Filing; Effective Date Unknown"
    assert row["Confidence"] == "Lower"


def test_filing_error_records_are_excluded() -> None:
    result = analyze_csv(FIXTURES / "sample_form_5500.csv", AnalysisSettings())
    assert "Error Co" not in set(result.company_leads["Company Name"])


def test_duplicate_filings_latest_received_wins() -> None:
    result = analyze_csv(FIXTURES / "sample_form_5500.csv", AnalysisSettings())
    alpha = result.plan_details[result.plan_details["Company Name"] == "Alpha Dental"].iloc[0]
    assert alpha["Date Received"] == "2025-02-10"
    assert result.metadata["duplicate_record_count"] == 1


def test_deduplication_runs_before_signal_filtering(tmp_path: Path) -> None:
    csv = tmp_path / "superseded_signal.csv"
    csv.write_text(
        "SPONSOR_DFE_NAME,SPONS_DFE_EIN,TOT_ACTIVE_PARTCP_CNT,TYPE_PENSION_BNFT_CODE,"
        "PLAN_EFF_DATE,INITIAL_FILING_IND,DATE_RECEIVED,SPONS_DFE_PN,FORM_TAX_PRD,FILING_STATUS,AMENDED_IND,ACK_ID\n"
        "Old Signal,123456789,25,1C,2024-01-01,Y,2025-01-01,001,2024-12-31,FILING_RECEIVED,0,A1\n"
        "Old Signal,123456789,25,2E,2024-01-01,Y,2025-02-01,001,2024-12-31,FILING_RECEIVED,1,A2\n"
    )
    result = analyze_csv(csv, AnalysisSettings(signal_types=ALL_SIGNAL_TYPES))
    assert result.plan_details.empty
    assert result.metadata["deduplicated_record_count"] == 1
    assert result.metadata["duplicate_record_count"] == 1


def test_empty_signal_filter_returns_all_participant_range_companies() -> None:
    result = analyze_csv(FIXTURES / "sample_form_5500.csv", AnalysisSettings())
    companies = set(result.company_leads["Company Name"])
    assert {"Alpha Dental", "Beta Tooling", "Delta Labs", "Gamma Holdings"}.issubset(companies)


def test_multiple_plans_under_one_ein_remain_distinct_before_aggregation(tmp_path: Path) -> None:
    csv = tmp_path / "multi.csv"
    csv.write_text(
        Path(FIXTURES / "sample_form_5500_sf.csv").read_text()
        + "SF Cash Co,098765432,Boulder,CO,541330,30,1A,2024-06-01,1,2025-02-04,003,2024,RECEIVED,N,S3\n"
    )
    result = analyze_csv(csv, AnalysisSettings(10, 100, 24))
    cash = result.company_leads[result.company_leads["EIN"] == "098765432"].iloc[0]
    assert cash["Qualifying Plan Count"] == 2


def test_company_aggregation_does_not_sum_participants() -> None:
    result = analyze_csv(FIXTURES / "sample_form_5500.csv", AnalysisSettings(include_frozen_plans=True))
    assert result.company_leads["Participant Count"].max() <= 100


def test_empty_file_controlled_error(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("")
    with pytest.raises(AnalysisError, match="empty"):
        analyze_csv(path, AnalysisSettings())


def test_missing_required_columns_controlled_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("A,B\n1,2\n")
    with pytest.raises(SchemaError, match="Missing required column"):
        analyze_csv(path, AnalysisSettings())


def test_ambiguous_mapping_controlled_error() -> None:
    columns = [
        "SPONSOR_DFE_NAME",
        "SPONSOR_DFE_NAME",
        "SPONS_DFE_EIN",
        "TOT_ACTIVE_PARTCP_CNT",
        "TYPE_PENSION_BNFT_CODE",
        "PLAN_EFF_DATE",
        "INITIAL_FILING_IND",
    ]
    with pytest.raises(SchemaError, match="Ambiguous"):
        resolve_schema(columns)


def test_no_valid_anchor_date_controlled_error(tmp_path: Path) -> None:
    path = tmp_path / "no_anchor.csv"
    text = Path(FIXTURES / "sample_form_5500_sf.csv").read_text()
    text = text.replace("2025-02-02", "").replace("2025-02-03", "")
    text = text.replace("2024-04-01", "").replace("2024-05-01", "")
    text = text.replace("2024", "")
    path.write_text(text)
    with pytest.raises(AnalysisError, match="No valid anchor date"):
        analyze_csv(path, AnalysisSettings())


def test_csv_exports_preserve_ein_strings() -> None:
    result = analyze_csv(FIXTURES / "sample_form_5500.csv", AnalysisSettings())
    csv_text = result.company_leads.to_csv(index=False)
    assert "012345678" in csv_text


def test_no_result_returns_empty_structured_dataframes() -> None:
    result = analyze_csv(FIXTURES / "sample_form_5500.csv", AnalysisSettings(999, 1000, 24))
    assert result.company_leads.empty
    assert list(result.company_leads.columns)
    assert result.plan_details.empty
    assert list(result.plan_details.columns)


def _make_zip(path: Path, members: dict[str, str]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in members.items():
            archive.writestr(name, content)


def test_zip_single_form_5500_processed(tmp_path: Path) -> None:
    path = tmp_path / "one.zip"
    _make_zip(path, {"main.csv": Path(FIXTURES / "sample_form_5500.csv").read_text()})
    result = analyze_zip(path, AnalysisSettings())
    assert result.metadata["uploaded_file_type"] == "ZIP"
    assert "Alpha Dental" in set(result.company_leads["Company Name"])


def test_zip_single_form_5500_sf_processed(tmp_path: Path) -> None:
    path = tmp_path / "sf.zip"
    _make_zip(path, {"sf.CSV": Path(FIXTURES / "sample_form_5500_sf.csv").read_text()})
    result = analyze_zip(path, AnalysisSettings())
    assert set(result.company_leads["Company Name"]) == {"SF Cash Co", "SF Pay Co"}


def test_csv_member_matching_case_insensitive(tmp_path: Path) -> None:
    path = tmp_path / "case.zip"
    _make_zip(path, {"UPPER.CSV": Path(FIXTURES / "sample_form_5500_sf.csv").read_text()})
    info = inspect_zip_archive(path)
    assert info["csv_candidate_count"] == 1


def test_state_options_are_discovered_from_zip(tmp_path: Path) -> None:
    path = tmp_path / "states.zip"
    _make_zip(path, {"main.csv": Path(FIXTURES / "sample_form_5500_sf.csv").read_text()})
    assert discover_state_options_zip(path) == ["CO", "WY"]


def test_zip_multiple_supported_files_return_selectable_candidates(tmp_path: Path) -> None:
    path = tmp_path / "many.zip"
    _make_zip(
        path,
        {
            "a.csv": Path(FIXTURES / "sample_form_5500.csv").read_text(),
            "b.csv": Path(FIXTURES / "sample_form_5500_sf.csv").read_text(),
        },
    )
    info = inspect_zip_archive(path)
    assert len(info["candidates"]) == 2
    with pytest.raises(AnalysisError, match="Multiple supported"):
        analyze_zip(path, AnalysisSettings())


def test_unsupported_csv_members_are_ignored(tmp_path: Path) -> None:
    path = tmp_path / "mixed.zip"
    _make_zip(
        path,
        {
            "notes.csv": "A,B\n1,2\n",
            "main.csv": Path(FIXTURES / "sample_form_5500_sf.csv").read_text(),
        },
    )
    info = inspect_zip_archive(path)
    assert [candidate.filename for candidate in info["candidates"]] == ["main.csv"]


def test_zip_no_csv_files_controlled_error(tmp_path: Path) -> None:
    path = tmp_path / "none.zip"
    _make_zip(path, {"readme.txt": "hello"})
    with pytest.raises(AnalysisError, match="no CSV"):
        inspect_zip_archive(path)


def test_corrupt_zip_controlled_error(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.zip"
    path.write_text("not zip")
    with pytest.raises(AnalysisError, match="not a readable ZIP"):
        inspect_zip_archive(path)


def test_encrypted_zip_rejected() -> None:
    info = zipfile.ZipInfo("data.csv")
    info.flag_bits |= 0x1
    with pytest.raises(AnalysisError, match="Encrypted"):
        validate_zip_infos([info], 10)


def test_unsafe_member_paths_rejected() -> None:
    info = zipfile.ZipInfo("../data.csv")
    with pytest.raises(AnalysisError, match="unsafe path"):
        validate_zip_infos([info], 10)


def test_nested_zip_archives_rejected() -> None:
    info = zipfile.ZipInfo("inner.zip")
    with pytest.raises(AnalysisError, match="Nested ZIP"):
        validate_zip_infos([info], 10)


def test_excessive_decompressed_size_rejected() -> None:
    info = zipfile.ZipInfo("big.csv")
    info.file_size = 3 * 1024**3
    info.compress_size = info.file_size
    with pytest.raises(AnalysisError, match="uncompressed size"):
        validate_zip_infos([info], 10)


def test_excessive_compression_ratio_rejected() -> None:
    info = zipfile.ZipInfo("ratio.csv")
    info.file_size = (MAX_COMPRESSION_RATIO + 1) * 100
    info.compress_size = 100
    with pytest.raises(AnalysisError, match="compression ratio"):
        validate_zip_infos([info], 10)


def test_combined_zip_members_deduplicate_across_files(tmp_path: Path) -> None:
    path = tmp_path / "combine.zip"
    data = Path(FIXTURES / "sample_form_5500.csv").read_text()
    _make_zip(path, {"a.csv": data, "b.csv": data})
    result = analyze_zip(
        path,
        AnalysisSettings(),
        selected_members=["a.csv", "b.csv"],
        combine_selected=True,
    )
    alpha_rows = result.plan_details[result.plan_details["Company Name"] == "Alpha Dental"]
    assert len(alpha_rows) == 1


def test_source_archive_member_names_retained(tmp_path: Path) -> None:
    path = tmp_path / "source.zip"
    _make_zip(path, {"folder/main.csv": Path(FIXTURES / "sample_form_5500_sf.csv").read_text()})
    result = analyze_zip(path, AnalysisSettings())
    assert "Archive Member Name" in result.plan_details.columns
    assert set(result.plan_details["Archive Member Name"]) == {"folder/main.csv"}


def test_temporary_extracted_files_removed_after_success_and_failure(tmp_path: Path) -> None:
    path = tmp_path / "temp.zip"
    _make_zip(path, {"main.csv": Path(FIXTURES / "sample_form_5500_sf.csv").read_text()})
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo("main.csv")
        temp_path = copy_zip_member_to_tempfile(archive, info)
    assert os.path.exists(temp_path)
    os.unlink(temp_path)
    analyze_zip(path, AnalysisSettings())
    leftovers = list(Path(tempfile_dir()).glob("*.csv"))
    assert temp_path not in [str(item) for item in leftovers]


def tempfile_dir() -> str:
    import tempfile

    return tempfile.gettempdir()
