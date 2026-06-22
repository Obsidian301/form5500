# Form 5500 Retirement-Plan Signal Finder

This internal Streamlit application analyzes Department of Labor Form 5500 and Form 5500-SF CSV datasets to find companies with retirement-plan signals that may justify additional M&A or business-transition research. It focuses on recently established Cash Balance plans and pay-related Defined Benefit plan features in a selected participant range.

The output is a prospecting signal, not proof that an owner is retiring, preparing to sell, or pursuing a transaction. Participant counts are plan-size proxies, not exact company headcount.

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\activate
```

## Testing

```bash
pytest
```

## Docker

```bash
docker build -t form5500-signal-finder .
docker run --rm -p 8501:8501 form5500-signal-finder
```

## Input Requirements

Upload a DOL Form 5500 or Form 5500-SF dataset as `.csv` or as a `.zip` containing one or more CSV files. The app supports DOL "Latest" datasets and "All" datasets when filing status and filing date fields are sufficient for filtering and deduplication. "Latest" files are preferred because they usually reduce duplicate and superseded filings before upload.

ZIP archives are inspected before processing. The app rejects encrypted archives, unsafe paths, nested archives, excessive decompressed sizes, and excessive compression ratios. It does not use `extractall()`.

The 2025 Form 5500 "All" layout uses fields such as `SPONS_DFE_PN` for plan number and `FORM_TAX_PRD` for the tax period; these are included in the schema aliases. Pension benefit codes may appear as separated values like `1A,1C` or compact DOL strings like `1A1I3D`; compact strings are split into exact two-character codes.

By default, the app returns filings that report target codes (`1A` or `1C`) within the selected participant range, even if the plan effective date is older. Enable **Require Recent Plan Effective Date** when you want to restrict results to plans whose `PLAN_EFF_DATE` falls inside the selected recency window. This distinction matters because `PLAN_EFF_DATE` is the plan's effective date and may not be the date a Cash Balance feature was adopted.

## Streamlit Community Cloud Deployment

1. Push this repository to GitHub.
2. Sign in to Streamlit Community Cloud.
3. Create a new application.
4. Select the repository and branch.
5. Set `app.py` as the entry point.
6. Deploy the application.
7. Open the generated public URL.

No secrets, passwords, or user-account configuration is required.

## Deployment Limitations

`.streamlit/config.toml` raises Streamlit's upload size to 1024 MB, but deployment providers may impose lower request-size, memory, or execution-time limits. For repeated processing of files around 500 MB or larger, Docker deployment on infrastructure with more memory is recommended.

## Data Caveats

Participant count is a plan-size proxy. Code `1A` indicates a pay-related Defined Benefit characteristic. Code `1C` identifies Cash Balance or similar plan features. Plan creation does not prove retirement intent. Initial filings with missing effective dates are lower-confidence signals. Duplicate and amended filings require careful plan-level deduplication before company-level aggregation.
