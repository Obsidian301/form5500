"""Schema resolution for Form 5500 and Form 5500-SF datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


class SchemaError(ValueError):
    """Raised when an uploaded file cannot be mapped to the canonical schema."""


REQUIRED_FIELDS = {
    "company_name",
    "ein",
    "participant_count",
    "pension_codes",
    "plan_effective_date",
    "initial_filing",
}

OPTIONAL_FIELDS = {
    "city",
    "state",
    "naics_code",
    "date_received",
    "plan_number",
    "form_year",
    "filing_status",
    "amended_indicator",
    "ack_id",
}

CANONICAL_FIELDS = tuple(sorted(REQUIRED_FIELDS | OPTIONAL_FIELDS))

ALIASES: dict[str, list[str]] = {
    "company_name": ["SPONSOR_DFE_NAME", "SPONS_DFE_NAME", "SF_SPONSOR_NAME"],
    "ein": ["SPONS_DFE_EIN", "SPONSOR_DFE_EIN", "SF_SPONS_EIN"],
    "city": [
        "SPONS_DFE_LOC_US_CITY",
        "SPONSOR_DFE_LOC_US_CITY",
        "SF_SPONS_US_CITY",
        "SF_SPONSOR_US_CITY",
    ],
    "state": [
        "SPONS_DFE_LOC_US_STATE",
        "SPONSOR_DFE_LOC_US_STATE",
        "SF_SPONS_US_STATE",
        "SF_SPONSOR_US_STATE",
    ],
    "naics_code": ["BUSINESS_CODE", "SF_BUSINESS_CODE"],
    "participant_count": [
        "TOT_ACTIVE_PARTCP_CNT",
        "TOT_ACT_PARTCP_EOY_CNT",
        "SF_TOT_ACT_PARTCP_EOY_CNT",
    ],
    "pension_codes": ["TYPE_PENSION_BNFT_CODE", "SF_TYPE_PENSION_BNFT_CODE"],
    "plan_effective_date": ["PLAN_EFF_DATE", "SF_PLAN_EFF_DATE"],
    "initial_filing": [
        "INITIAL_FILING_IND",
        "FIRST_RTN_RPT_IND",
        "SF_INITIAL_FILING_IND",
    ],
    "date_received": ["DATE_RECEIVED", "SF_DATE_RECEIVED"],
    "plan_number": ["PLAN_NUM", "PLAN_NUMBER", "SPONS_DFE_PN", "SF_PLAN_NUM"],
    "form_year": [
        "FORM_YEAR",
        "SF_FORM_YEAR",
        "FORM_TAX_PRD",
        "TAX_PRD",
        "TAX_PERIOD",
        "FORM_PLAN_YEAR_BEGIN_DATE",
    ],
    "filing_status": ["FILING_STATUS", "SF_FILING_STATUS"],
    "amended_indicator": ["AMENDED_IND", "SF_AMENDED_IND"],
    "ack_id": ["ACK_ID", "SF_ACK_ID"],
}


@dataclass(frozen=True)
class SchemaResolution:
    """Resolved canonical mapping for an uploaded dataset."""

    mapping: dict[str, str]
    source_form_type: str
    available_columns: list[str]


def normalize_column_name(column: object) -> str:
    """Normalize a source column name for case-insensitive matching."""

    return str(column).strip().upper()


def normalize_column_names(columns: Iterable[object]) -> list[str]:
    """Return normalized header names in their original order."""

    return [normalize_column_name(column) for column in columns]


def resolve_schema(columns: Iterable[object]) -> SchemaResolution:
    """Resolve source columns to canonical field names.

    Matching is case-insensitive, strips surrounding whitespace, and uses only
    explicit aliases. If multiple source columns match one canonical field, the
    alias order in ``ALIASES`` is the documented priority. Duplicate aliases of
    the same priority are treated as ambiguous.
    """

    original_columns = [str(column).strip() for column in columns]
    normalized_to_originals: dict[str, list[str]] = {}
    for original in original_columns:
        normalized_to_originals.setdefault(normalize_column_name(original), []).append(original)

    mapping: dict[str, str] = {}
    ambiguous: dict[str, list[str]] = {}

    for canonical, aliases in ALIASES.items():
        for alias in aliases:
            matches = normalized_to_originals.get(normalize_column_name(alias), [])
            if len(matches) > 1:
                ambiguous[canonical] = matches
                break
            if len(matches) == 1:
                mapping[canonical] = matches[0]
                break

    if ambiguous:
        details = "; ".join(
            f"{field}: {', '.join(matches)}" for field, matches in sorted(ambiguous.items())
        )
        raise SchemaError(f"Ambiguous schema mapping for {details}.")

    missing_required = sorted(REQUIRED_FIELDS - mapping.keys())
    if missing_required:
        field = missing_required[0]
        accepted = ", ".join(ALIASES[field])
        available = ", ".join(original_columns)
        raise SchemaError(
            "Missing required column for canonical field "
            f"'{field}'. Accepted aliases: {accepted}. "
            f"Available columns: {available}."
        )

    source_form_type = _detect_form_type(mapping)
    return SchemaResolution(mapping=mapping, source_form_type=source_form_type, available_columns=original_columns)


def _detect_form_type(mapping: dict[str, str]) -> str:
    """Infer the source form type from resolved aliases."""

    if any(str(source).strip().upper().startswith("SF_") for source in mapping.values()):
        return "Form 5500-SF"
    return "Form 5500"
