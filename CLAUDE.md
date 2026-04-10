# Resident Scheduler 2

## Overview
Streamlit web app for managing surgery resident rotation schedules at Conemaugh Surgery Department. Handles yearly/daily schedule generation, compliance checks, ACGME case log tracking, calendar subscriptions, and Amion-compatible exports.

## Tech Stack
- **Framework:** Streamlit (multi-page app)
- **Language:** Python 3.10+
- **Key Libraries:** pandas, numpy, openpyxl, xlsxwriter, pdfplumber, pytesseract, PuLP (optional ILP optimizer), altair, matplotlib
- **System Dependencies:** tesseract-ocr, poppler-utils

## Project Structure
```
app.py                          # Main application (~4,600 lines, 6 tabs)
pages/1_Calendar_Feed.py        # ICS calendar subscription endpoint
imports/case_logs/              # ACGME case log uploads
.streamlit/config.toml          # Streamlit config (dark theme, headless)
.streamlit/secrets.toml         # APP_PASSWORD (not committed)
requirements.txt                # Python dependencies
packages.txt                    # System packages for Streamlit Cloud
```

## Commands
- **Run locally:** `streamlit run app.py`
- **Install deps:** `pip install -r requirements.txt`
- **Access:** http://localhost:8501

## Architecture
- **Single-file app** (`app.py`) with 6 tabs: Yearly, Daily Blocks, Checks & Balance, Export, Case Logs, Weekly Schedule
- **Schedule generation** uses a greedy algorithm (`auto_generate_yearly()`) with optional ILP polish (`polish_with_optimizer()`)
- **State management** via `st.session_state` with pickle-based caching (`.schedule_cache.pkl`, `.case_logs_cache.pkl`)
- **Calendar feeds** served via the `pages/1_Calendar_Feed.py` page using ICS format

## Key Functions
| Function | Purpose |
|----------|---------|
| `auto_generate_yearly()` | Greedy schedule generation across 13 blocks |
| `build_dailies_from_yearly()` | Expand yearly to daily assignments |
| `auto_assign_weekend_call()` | Fair call distribution |
| `compute_checks()` | Validate compliance and fairness |
| `generate_ics_for_resident()` | Create ICS calendar feeds |
| `parse_acgme_pdf()` | Extract case logs from PDFs |
| `to_excel()` / `to_amion_csv()` | Export functions |

## Coding Conventions
- All UI logic uses Streamlit widgets and `st.session_state`
- Color palette is centrally defined (Gold=#FFD966, Nights=#000000, etc.)
- Schedule data stored as pandas DataFrames
- 13 academic blocks per year
- PGY levels (1-5) and Prelim status affect rotation eligibility
- Pittsburgh rotation: PGY-3 only, 3 consecutive blocks, excludes late January

## Deployment
- **Production:** Streamlit Cloud (auto-deploys on git push)
- **Secrets:** Configured via Streamlit Cloud dashboard or `.streamlit/secrets.toml` locally

## Important Notes
- `app.py` is large (~4,600 lines) - be cautious with full-file reads
- No test suite exists yet
- Authentication is password-based via `st.secrets["APP_PASSWORD"]`
- `.schedule_cache.pkl` must be accessible for calendar feeds to work
